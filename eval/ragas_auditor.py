"""Gate 8: continuous RAGAS evaluation as an async post-hook (design.md §1.3 step 9, KPI-20/21).

The request path only calls `submit`, which never blocks: a sampled response goes on a bounded queue
(dropped when full) and a background worker scores it. Scores land in a rolling window (default 5 minutes)
that the Gate 11 "hallucination surge" alert reads. The offline release gate stays in eval/run_ragas.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from observability.tracing import remote_context, span

log = logging.getLogger("eval.ragas_auditor")
METRICS = ("faithfulness", "answer_relevancy")


@dataclass(frozen=True)
class Sample:
    trace_id: str
    question: str
    response: str
    contexts: tuple[str, ...]
    traceparent: str | None = None  # the request's span, so the async score shows up in the same trace


class Scorer(Protocol):
    async def score(self, s: Sample) -> dict[str, float]: ...


class RagasScorer:
    """Same judge, embeddings and metric settings as the Gate 3 release gate (eval/run_ragas.py)."""

    def __init__(self) -> None:
        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness, ResponseRelevancy

        from eval.run_ragas import BgeEmbeddings, ClaudeLLM  # lazy: pulls in ragas and the judge client

        llm = ClaudeLLM()
        self._sample = SingleTurnSample
        self._metrics = {"faithfulness": Faithfulness(llm=llm),
                         "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=BgeEmbeddings(), strictness=1)}

    async def score(self, s: Sample) -> dict[str, float]:
        sample = self._sample(user_input=s.question, response=s.response, retrieved_contexts=list(s.contexts))
        return {name: float(await m.single_turn_ascore(sample)) for name, m in self._metrics.items()}


def sampled(trace_id: str, rate: float) -> bool:
    """Deterministic per trace, so a retried request is scored or skipped the same way on every worker."""
    return int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


class RagasAuditor:
    def __init__(
        self,
        scorer: Scorer,
        *,
        sample_rate: float = 0.1,
        window_s: float = 300.0,
        queue_size: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scorer, self.sample_rate, self.window_s, self.clock = scorer, sample_rate, window_s, clock
        self._queue: asyncio.Queue[Sample] = asyncio.Queue(maxsize=queue_size)
        self._window: deque[tuple[float, dict[str, float]]] = deque()
        self.dropped = self.failed = 0
        self._worker: asyncio.Task | None = None

    def submit(self, s: Sample) -> bool:
        """Called from the request path. Returns True when the sample was queued. Never blocks or raises."""
        if not s.contexts or not sampled(s.trace_id, self.sample_rate):
            return False
        try:
            self._queue.put_nowait(s)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        return True

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    async def drain(self) -> None:
        """Wait until every queued sample is scored (tests and graceful shutdown)."""
        await self._queue.join()

    async def _run(self) -> None:
        while True:
            s = await self._queue.get()
            try:
                with span("hook.post.ragas", context=remote_context(s.traceparent)):  # joins the request's trace
                    scores = await self.scorer.score(s)
                self._window.append((self.clock(), {k: v for k, v in scores.items() if not math.isnan(v)}))
            except Exception:  # a judge outage must not kill the worker
                self.failed += 1
                log.exception("ragas scoring failed for trace %s", s.trace_id)
            finally:
                self._queue.task_done()

    def window(self, metric: str) -> tuple[float | None, int]:
        """Mean of `metric` over the last `window_s` seconds, and the number of samples behind it."""
        cutoff = self.clock() - self.window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        xs = [scores[metric] for _, scores in self._window if metric in scores]
        return (sum(xs) / len(xs) if xs else None), len(xs)
