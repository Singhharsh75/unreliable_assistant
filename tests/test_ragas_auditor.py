"""Gate 8 — continuous RAGAS post-hook: sampling, non-blocking submit, rolling window, failure isolation."""
import asyncio

import pytest

from eval.ragas_auditor import RagasAuditor, Sample, sampled


class FakeScorer:
    def __init__(self, faith=1.0, fail=False):
        self.faith, self.fail, self.n = faith, fail, 0

    async def score(self, s):
        self.n += 1
        if self.fail:
            raise RuntimeError("judge down")
        return {"faithfulness": self.faith, "answer_relevancy": 0.9}


def s(i):
    return Sample(f"trace{i}", "q", "a", ("ctx",))


def test_sampling_is_deterministic_and_near_the_rate():
    hits = sum(sampled(f"t{i}", 0.1) for i in range(5000))
    assert 400 < hits < 600 and sampled("t1", 0.1) == sampled("t1", 0.1)
    assert all(sampled(f"t{i}", 1.0) for i in range(100)) and not any(sampled(f"t{i}", 0.0) for i in range(100))


def test_submit_never_blocks_and_counts_drops():
    async def go():
        a = RagasAuditor(FakeScorer(), sample_rate=1.0, queue_size=2)  # worker not started: queue fills
        return [a.submit(s(i)) for i in range(5)], a.dropped
    accepted, dropped = asyncio.run(go())
    assert accepted == [True, True, False, False, False] and dropped == 3


def test_no_contexts_is_not_scored():
    a = RagasAuditor(FakeScorer(), sample_rate=1.0)
    assert not a.submit(Sample("t", "q", "a", ()))


def test_window_mean_and_expiry():
    now = [0.0]

    async def go():
        a = RagasAuditor(FakeScorer(faith=0.8), sample_rate=1.0, window_s=300, clock=lambda: now[0])
        a.start()
        for i in range(3):
            a.submit(s(i))
        await a.drain()
        first = a.window("faithfulness")
        now[0] = 301.0
        await a.stop()
        return first, a.window("faithfulness")
    first, later = asyncio.run(go())
    assert first == (pytest.approx(0.8), 3) and later == (None, 0)


def test_scorer_failure_does_not_kill_the_worker():
    async def go():
        sc = FakeScorer(fail=True)
        a = RagasAuditor(sc, sample_rate=1.0)
        a.start()
        a.submit(s(1)); a.submit(s(2))
        await a.drain()
        await a.stop()
        return sc.n, a.failed
    assert asyncio.run(go()) == (2, 2)
