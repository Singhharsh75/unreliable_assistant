"""Gate 3/8: RAGAS release gate over eval/golden_dataset.jsonl.

  uv run eval/run_ragas.py retrieve [--limit N]   # stratified sample of N rows (default 50, 0 = all 250); offline
  uv run eval/run_ragas.py answer                 # LLM answers from the retrieved chunks   (needs an API key)
  uv run eval/run_ragas.py score                  # ragas metrics + release gates, exit 1 on fail (needs key)

Sized for a side project (design.md ADR-007): ~5 judge calls per answerable row. Context precision/recall are
ID-based against gold chunk hashes (no LLM); answer relevancy uses 1 reverse question instead of ragas' default 3.
Set RAGAS_WORKERS=2 for OpenRouter :free models (20 requests/minute).

LLM provider: OpenRouter when OPENROUTER_API_KEY is set, else the Anthropic API (ANTHROPIC_API_KEY).
Override models with GENERATOR_MODEL / JUDGE_MODEL. Keys and models load from .env (see .env.example).

Answerable rows are scored with RAGAS. Negative rows are scored on abstention instead: their reference is
"not in the knowledge base", which RAGAS faithfulness/recall cannot score meaningfully.
"""
import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas sends usage analytics unless told not to

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

import anthropic
import numpy as np
import openai
from dotenv import load_dotenv
from langchain_core.outputs import Generation, LLMResult
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import ContextEntityRecall, Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from eval.build_golden import quotas
from mcp_servers.rag_server import Retriever, _demo_req, embed

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # real env vars win over .env
OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# defaults: design.md §0 LLM as generator; a separate judge avoids self-preference
# Cheapest paid defaults (design.md ADR-007); the OpenRouter judge is another vendor, so it does not grade its own
# family. Set GENERATOR_MODEL to Opus 5.5 to certify the production LLM; :free models are listed in .env.example.
GENERATOR = os.getenv("GENERATOR_MODEL") or ("anthropic/claude-haiku-5.5" if OPENROUTER else "claude-haiku-5-5")
JUDGE = os.getenv("JUDGE_MODEL") or ("deepseek/deepseek-v4-flash" if OPENROUTER else "claude-haiku-5-5")
FREE = OPENROUTER and ":free" in GENERATOR + JUDGE
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# :free models allow 20 requests/minute and failed requests count toward the daily cap, so run them serially
WORKERS = int(os.getenv("RAGAS_WORKERS") or (1 if FREE else 8))
DEFAULT_SAMPLE, SEED = 50, 7
# Waived gates (design.md ADR-009): 90% of the mean measured across the three Gate 3 runs so far.
# Blueprint targets: faithfulness 0.92, relevancy 0.90, ID precision 0.88, ID recall 0.85, entity recall 0.85.
GATES = {"faithfulness": 0.737, "answer_relevancy": 0.688, "id_context_precision": 0.785,
         "id_context_recall": 0.764, "context_entity_recall": 0.453}
ABSTAIN_TEXT = "The knowledge base does not cover this question."
EVAL = Path(__file__).resolve().parent
GOLDEN, RUNS = EVAL / "golden_dataset.jsonl", EVAL / "runs"
INPUTS, ANSWERS, REPORT = RUNS / "retrieval.jsonl", RUNS / "answers.jsonl", RUNS / "ragas_report.json"
SYSTEM = ("You answer questions for a regulated manufacturing assistant using ONLY the numbered context chunks. "
          "Be concise and specific. If the chunks do not contain the answer, reply exactly: " + ABSTAIN_TEXT)


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def text_of(msg) -> str | None:
    if msg.stop_reason == "refusal":
        return None
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def make_client():
    if OPENROUTER:
        return openai.AsyncOpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    return anthropic.AsyncAnthropic()


async def complete(client, model: str, prompt: str, system: str | None = None, stop=None) -> str | None:
    """One user turn → text, or None on refusal. Same call shape for both providers."""
    if OPENROUTER:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = await client.chat.completions.create(
            model=model, messages=msgs, max_tokens=4096, stop=stop or None,  # low cap: OpenRouter reserves credit by max_tokens
            extra_body={"reasoning": {"effort": "low"}})  # ignored by models without reasoning
        c = r.choices[0]
        return None if c.finish_reason == "content_filter" else (c.message.content or "").strip()
    extra = {"system": system} if system else {}
    if "haiku" not in model:  # Haiku 5.5 has no server-side refusal fallback
        extra.update(betas=[FALLBACK_BETA], fallbacks="default")
    msg = await client.beta.messages.create(
        model=model, max_tokens=16000, output_config={"effort": "low"}, stop_sequences=stop or None,
        messages=[{"role": "user", "content": prompt}], **extra)
    return text_of(msg)


class ClaudeLLM(BaseRagasLLM):
    """ragas judge LLM on Anthropic or OpenRouter. Ignores ragas' temperature: current Claude models reject it."""

    def __init__(self, model: str = JUDGE):
        super().__init__()
        self.model, self.aclient = model, make_client()

    async def _one(self, prompt: str, stop) -> str:
        return await complete(self.aclient, self.model, prompt, stop=stop) or ""

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
        texts = await asyncio.gather(*(self._one(prompt.to_string(), stop) for _ in range(n)))
        return LLMResult(generations=[[Generation(text=t) for t in texts]])

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
        return asyncio.run(self.agenerate_text(prompt, n, temperature, stop, callbacks))

    def is_finished(self, response: LLMResult) -> bool:
        return all(g.text for gens in response.generations for g in gens)


class BgeEmbeddings(BaseRagasEmbeddings):
    """ragas embeddings on the same bge-large ONNX model the retriever uses (answer relevancy only)."""

    def embed_query(self, text: str) -> list[float]:
        return embed([text])[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.mean(xs)), 4) if xs else float("nan")


def id_precision(retrieved: list[str], gold: list[str]) -> float:
    """RAGAS context precision (rank-weighted: mean precision@k over gold ranks), gold chunk IDs as relevance.
    ragas' IDBasedContextPrecision is plain set precision, which is not the blueprint metric."""
    g, hits, total = set(gold), 0, 0.0
    for k, h in enumerate(retrieved, 1):
        if h in g:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def id_recall(retrieved: list[str], gold: list[str]) -> float:
    return len(set(gold) & set(retrieved)) / len(set(gold))  # abstained (nothing retrieved) scores 0


def sample(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Deterministic stratified sample keeping the blueprint category mix; n <= 0 or n >= len(rows) keeps all."""
    if n <= 0 or n >= len(rows):
        return rows
    rng, out = random.Random(seed), []
    for cat, k in quotas(n).items():
        pool = [r for r in rows if r["category"] == cat]
        out += rng.sample(pool, min(k, len(pool)))
    return out


def cmd_retrieve(limit: int) -> None:
    gold, r, rows = sample(read(GOLDEN), limit), Retriever(), []
    for i, g in enumerate(gold, 1):
        out = r.retrieve(_demo_req(g["question"], g["tenant_id"]))
        rows.append({**g, "abstain": out.abstain, "retrieved_contexts": [c.text for c in out.chunks],
                     "retrieved_context_ids": [c.chunk_hash for c in out.chunks],
                     "stage_latency_ms": out.stage_latency_ms})
        if i % 25 == 0:
            print(f"retrieved {i}/{len(gold)}", file=sys.stderr)
    write(INPUTS, rows)
    pos = [x for x in rows if x["category"] != "negative"]
    neg = [x for x in rows if x["category"] == "negative"]
    print(json.dumps({
        "rows": len(rows), "answerable": len(pos),
        "id_context_precision": mean([id_precision(x["retrieved_context_ids"], x["gold_chunk_hashes"]) for x in pos]),
        "id_context_recall": mean([id_recall(x["retrieved_context_ids"], x["gold_chunk_hashes"]) for x in pos]),
        "answerable_abstained": sum(x["abstain"] for x in pos),
        "negative_abstain_rate": mean([float(x["abstain"]) for x in neg]),
        "p95_ms": {s: round(float(np.percentile([x["stage_latency_ms"][s] for x in rows], 95)), 2)
                   for s in ("bm25", "dense", "rrf", "rerank")},
    }, indent=2))


async def _answer_all(rows: list[dict]) -> list[dict]:
    client, sem = make_client(), asyncio.Semaphore(WORKERS)

    async def one(x: dict) -> dict:
        if x["abstain"]:  # design.md request flow step 5: abstention short-circuits the LLM
            return {**x, "response": ABSTAIN_TEXT}
        ctx = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(x["retrieved_contexts"], 1))
        async with sem:
            text = await complete(client, GENERATOR, f"Context:\n{ctx}\n\nQuestion: {x['question']}", system=SYSTEM)
        return {**x, "response": text or ABSTAIN_TEXT}

    return await asyncio.gather(*(one(x) for x in rows))


def cmd_answer() -> None:
    write(ANSWERS, asyncio.run(_answer_all(read(INPUTS))))
    print(f"wrote {ANSWERS}")


def cmd_score() -> None:
    rows = read(ANSWERS)
    pos, neg = [x for x in rows if x["category"] != "negative"], [x for x in rows if x["category"] == "negative"]
    ds = EvaluationDataset([SingleTurnSample(user_input=x["question"], response=x["response"],
                                             retrieved_contexts=x["retrieved_contexts"] or [""],
                                             reference=x["reference_answer"]) for x in pos])
    # LLM calls per row: faithfulness 2, relevancy 1, entity recall 2. Context precision/recall need no LLM.
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), ContextEntityRecall()]
    df = evaluate(ds, metrics=metrics, llm=ClaudeLLM(), embeddings=BgeEmbeddings(),
                  run_config=RunConfig(max_workers=WORKERS, timeout=180)).to_pandas()
    df["id_context_precision"] = [id_precision(x["retrieved_context_ids"], x["gold_chunk_hashes"]) for x in pos]
    df["id_context_recall"] = [id_recall(x["retrieved_context_ids"], x["gold_chunk_hashes"]) for x in pos]
    df["category"] = [x["category"] for x in pos]
    scores = {m: mean(df[m]) for m in GATES}
    by_cat = {c: {m: mean(g[m]) for m in GATES} for c, g in df.groupby("category")}
    gates = {m: {"score": scores[m], "threshold": t, "pass": scores[m] >= t} for m, t in GATES.items()}
    report = {"provider": "openrouter" if OPENROUTER else "anthropic", "judge": JUDGE, "generator": GENERATOR,
              "answerable": len(pos), "gates": gates, "by_category": by_cat,
              "negative_abstain_rate": mean([float(x["response"] == ABSTAIN_TEXT) for x in neg]),
              "pass": all(g["pass"] for g in gates.values())}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieve", "answer", "score"])
    ap.add_argument("--limit", type=int, default=DEFAULT_SAMPLE, help="stratified sample size; 0 = all rows")
    a = ap.parse_args()
    if a.stage != "retrieve" and not (OPENROUTER or os.getenv("ANTHROPIC_API_KEY")):
        sys.exit("set OPENROUTER_API_KEY or ANTHROPIC_API_KEY (env or .env; see .env.example)")
    {"retrieve": lambda: cmd_retrieve(a.limit), "answer": cmd_answer, "score": cmd_score}[a.stage]()
