"""Gate 3/8: golden RAG dataset — Claude generates candidates, a second Claude judges them (LLM-as-judge).

  uv run eval/build_golden.py --total 250            # needs ANTHROPIC_API_KEY
  uv run eval/build_golden.py --total 20 --dry-run   # print quotas and corpus size, no API calls

Strata (blueprint Gate 3): factoid 25%, complex reasoning 25%, multi-hop 20%, summary 15%, negative 15%.
Gold chunk hashes come from rag_server.chunk_notes, so they match indexes/rag exactly.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from mcp_servers.rag_server import VAULT, chunk_notes, load_notes

GENERATOR, JUDGE = "claude-opus-5-5", "claude-sonnet-5-5"  # different judge model avoids self-preference bias
FALLBACK_BETA = "server-side-fallback-2026-07-01"
STRATA = {"factoid": 0.25, "complex_reasoning": 0.25, "multi_hop": 0.20, "summary": 0.15, "negative": 0.15}
BRIEF = {
    "factoid": "asks for one fact stated explicitly in ONE chunk (a value, ID, limit, duration, owner). Cite that chunk.",
    "complex_reasoning": "applies a rule from ONE note to a concrete scenario the note does not spell out "
                         "(conditions, exceptions, comparisons). The answer must still follow strictly from the chunk.",
    "multi_hop": "can only be answered by combining facts from TWO OR MORE different notes. Cite every chunk needed.",
    "summary": "asks for an overview of one note's procedure or policy. The reference answer covers all its key points.",
    "negative": "sounds in-domain and plausible, but the corpus does NOT contain the answer. The reference answer says "
                "the knowledge base does not cover it. gold_chunk_ids must be empty.",
}
PER_CALL, MIN_SCORE, DUP_JACCARD, DEFAULT_TENANT = 5, 4, 0.8, "tenant_demo"
OUT_DIR = Path(__file__).resolve().parent


class Candidate(BaseModel):
    question: str
    reference_answer: str
    gold_chunk_ids: list[str] = Field(description='Corpus chunk labels such as "C3"')


class CandidateBatch(BaseModel):
    items: list[Candidate]


class Verdict(BaseModel):
    grounded: int = Field(description="1-5. Reference answer is fully supported by the gold chunks and contradicts "
                                      "nothing in the corpus. For negative: the corpus truly lacks the answer.")
    gold_complete: int = Field(description="1-5. Gold chunks are necessary and sufficient (negative: empty is correct).")
    category_fit: int = Field(description="1-5. The question matches its category definition.")
    standalone: int = Field(description="1-5. Clear, unambiguous, answerable without seeing the chunks.")
    reason: str


def quotas(total: int) -> dict[str, int]:
    q = {s: int(total * p) for s, p in STRATA.items()}
    for s in sorted(STRATA, key=lambda s: -(total * STRATA[s] % 1))[:total - sum(q.values())]:
        q[s] += 1  # largest remainder, so quotas always sum to total
    return q


def corpus_text(chunks: list[dict]) -> str:
    return "\n\n".join(f"<chunk id=\"C{i}\" note=\"{c['note_id']}\" domain=\"{c['domain']}\" "
                       f"tenants=\"{','.join(c['tenants'] or ['all'])}\">\n{c['text']}\n</chunk>"
                       for i, c in enumerate(chunks))


def tokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", q.lower()))


def is_dup(q: str, kept: list[str]) -> bool:
    t = tokens(q)
    return any(len(t & (k := tokens(o))) / max(len(t | k), 1) >= DUP_JACCARD for o in kept)


def check(cat: str, cand: Candidate, chunks: list[dict]) -> tuple[str | None, list[int]]:
    """Deterministic checks before paying for a judge call. Returns (reject_reason, gold chunk indices)."""
    try:
        idx = sorted({int(g.lstrip("Cc")) for g in cand.gold_chunk_ids})
    except ValueError:
        return "bad_chunk_id", []
    if any(i >= len(chunks) for i in idx):
        return "bad_chunk_id", []
    if cat == "negative":
        return ("negative_has_gold" if idx else None), idx
    if not idx:
        return "missing_gold", idx
    if cat == "multi_hop" and len({chunks[i]["note_id"] for i in idx}) < 2:
        return "multi_hop_single_note", idx
    if len({tuple(chunks[i]["tenants"]) for i in idx if chunks[i]["tenants"]}) > 1:
        return "mixed_private_tenants", idx
    return None, idx


class Builder:
    def __init__(self, chunks: list[dict], seed: int):
        self.client = anthropic.Anthropic()
        self.chunks, self.rng = chunks, random.Random(seed)
        # stable corpus prefix shared by every call, cached across generator and judge requests
        self.system = [{"type": "text", "cache_control": {"type": "ephemeral"},
                        "text": "You build and audit a golden evaluation set for a retrieval-augmented assistant. "
                                "The knowledge base is exactly the chunks below; nothing else counts as known.\n\n"
                                + corpus_text(chunks)}]
        self.notes = sorted({c["note_id"] for c in chunks})

    def _call(self, model: str, effort: str, prompt: str, fmt):
        r = self.client.beta.messages.parse(
            model=model, max_tokens=16000, betas=[FALLBACK_BETA], fallbacks="default",
            system=self.system, output_config={"effort": effort}, output_format=fmt,
            messages=[{"role": "user", "content": prompt}])
        return None if r.stop_reason == "refusal" else r.parsed_output

    def generate(self, cat: str, n: int, avoid: list[str]) -> list[Candidate]:
        focus = ", ".join(self.rng.sample(self.notes, min(3, len(self.notes))))
        prompt = (f"Write {n} {cat} questions. A {cat} question {BRIEF[cat]}\n"
                  f"Prefer the notes: {focus} (use others if needed). Phrase questions the way an operator would ask.\n"
                  "Do not repeat or paraphrase these existing questions:\n" + "\n".join(f"- {q}" for q in avoid[-40:]))
        out = self._call(GENERATOR, "medium", prompt, CandidateBatch)
        return out.items if out else []

    def judge(self, cat: str, cand: Candidate) -> Verdict | None:
        prompt = (f"Grade this golden-set item strictly. Category: {cat} — a {cat} question {BRIEF[cat]}\n\n"
                  f"Question: {cand.question}\nReference answer: {cand.reference_answer}\n"
                  f"Gold chunks: {cand.gold_chunk_ids or 'none'}\n\n"
                  "Score each criterion 1-5 (5 = no issues). Use 3 or lower for any factual error, "
                  "unsupported claim, missing gold chunk, or ambiguity.")
        return self._call(JUDGE, "medium", prompt, Verdict)


def admit(cat: str, c: Candidate, idx: list[int], v: Verdict | None, kept: dict, quota: dict, chunks: list[dict],
          rejected: Counter, generator: str = GENERATOR, judge_model: str = JUDGE) -> None:
    """Apply the judge threshold, quota, and dedup; append the row to kept[cat] if it survives."""
    scores = v and {k: max(1, min(5, getattr(v, k))) for k in ("grounded", "gold_complete", "category_fit", "standalone")}
    if not scores:
        rejected["judge_refused"] += 1
    elif min(scores.values()) < MIN_SCORE:
        rejected["judge_" + min(scores, key=scores.get)] += 1
    elif len(kept[cat]) >= quota[cat]:
        rejected["over_quota"] += 1
    elif is_dup(c.question, [k["question"] for k in kept[cat]]):
        rejected["duplicate"] += 1
    else:
        private = [t for i in idx if (t := chunks[i]["tenants"])]
        kept[cat].append({
            "id": f"{cat}-{len(kept[cat]) + 1:03d}", "category": cat, "question": c.question,
            "reference_answer": c.reference_answer,
            "gold_chunk_hashes": [chunks[i]["chunk_hash"] for i in idx],
            "gold_note_ids": sorted({chunks[i]["note_id"] for i in idx}),
            "tenant_id": private[0][0] if private else DEFAULT_TENANT,
            "judge": {**scores, "reason": v.reason}, "generator": generator, "judge_model": judge_model})


def from_files(candidates: Path, verdicts: Path, total: int, generator: str, judge_model: str) -> tuple[list[dict], dict]:
    """Assemble from pre-written candidates + judge verdicts (JSONL keyed by cid), e.g. when no API key is available."""
    chunks = chunk_notes(load_notes(VAULT))
    quota, kept, rejected = quotas(total), {s: [] for s in STRATA}, Counter()
    by_cid = {(r := json.loads(line))["cid"]: r for line in verdicts.read_text().splitlines() if line.strip()}
    for line in candidates.read_text().splitlines():
        r = json.loads(line)
        cat, c = r["category"], Candidate.model_validate(r)
        reason, idx = check(cat, c, chunks)
        if reason:
            rejected[reason] += 1
            continue
        v = by_cid.get(r["cid"])
        admit(cat, c, idx, v and Verdict.model_validate(v), kept, quota, chunks, rejected, generator,
              (v or {}).get("judge_model", judge_model))  # a verdict row may name its own judge
    rows = [r for s in STRATA for r in kept[s]]
    return rows, {"target": quota, "kept": {s: len(kept[s]) for s in STRATA}, "rejected": dict(rejected),
                  "chunks": len(chunks), "notes": len({c["note_id"] for c in chunks}),
                  "source": {"candidates": candidates.name, "verdicts": verdicts.name}}


def build(total: int, seed: int, workers: int, max_rounds: int) -> tuple[list[dict], dict]:
    chunks = chunk_notes(load_notes(VAULT))
    quota, kept, rejected = quotas(total), {s: [] for s in STRATA}, Counter()
    b = Builder(chunks, seed)
    with ThreadPoolExecutor(workers) as pool:
        for rnd in range(1, max_rounds + 1):
            need = {s: quota[s] - len(kept[s]) for s in STRATA if len(kept[s]) < quota[s]}
            if not need:
                break
            # over-generate ~1.5x: the judge rejects a share of candidates
            jobs = [(s, PER_CALL) for s, n in need.items() for _ in range(-(-int(n * 1.5) // PER_CALL))]
            batches = pool.map(lambda j: (j[0], b.generate(j[0], j[1], [k["question"] for k in kept[j[0]]])), jobs)
            cands = []
            for cat, items in batches:
                for c in items:
                    reason, idx = check(cat, c, chunks)
                    if reason:
                        rejected[reason] += 1
                    else:
                        cands.append((cat, c, idx))
            for (cat, c, idx), v in zip(cands, pool.map(lambda t: b.judge(t[0], t[1]), cands)):
                admit(cat, c, idx, v, kept, quota, chunks, rejected)
            print(f"round {rnd}: " + ", ".join(f"{s} {len(kept[s])}/{quota[s]}" for s in STRATA), file=sys.stderr)
    rows = [r for s in STRATA for r in kept[s]]
    report = {"target": quota, "kept": {s: len(kept[s]) for s in STRATA}, "rejected": dict(rejected),
              "chunks": len(chunks), "notes": len(b.notes), "rounds": rnd, "seed": seed}
    return rows, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--candidates", type=Path, help="pre-written candidates JSONL (skips API generation)")
    ap.add_argument("--verdicts", type=Path, help="judge verdicts JSONL keyed by cid (with --candidates)")
    ap.add_argument("--generator-label", default="claude-code-session")
    ap.add_argument("--judge-label", default="claude-code-subagent")
    a = ap.parse_args()
    if a.candidates:
        rows, report = from_files(a.candidates, a.verdicts, a.total, a.generator_label, a.judge_label)
    elif a.dry_run:
        chunks = chunk_notes(load_notes(VAULT))
        print(json.dumps({"quotas": quotas(a.total), "chunks": len(chunks),
                          "corpus_chars": len(corpus_text(chunks))}, indent=2))
        sys.exit()
    else:
        rows, report = build(a.total, a.seed, a.workers, a.max_rounds)
    (OUT_DIR / "golden_dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT_DIR / "golden_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if sum(report["kept"].values()) < a.total:
        sys.exit("quota not met; see golden_report.json (likely corpus too small for distinct questions)")
