"""KPI-24 on RELEASED answers, judged independently of the NLI verifier (ADR-016).

Release path as in triage/orchestrator.py with NLI: a draft is released only if the verifier grounds every claim;
a vetoed draft becomes the abstention text. An LLM judge (JUDGE_MODEL, which reads the whole chunk set and accepts
conclusions that follow from it) then lists the sentences of each released draft that the chunks do not support.
The judge shares nothing with the NLI model, so a verifier blind spot shows up here instead of cancelling out.

  uv run eval/released_judge.py        # writes eval/runs/released_judge.json
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import run_ragas as rr  # noqa: E402
from triage.nli import entailment  # noqa: E402
from triage.verifier import segments, verify, with_question  # noqa: E402

OUT = rr.RUNS / "released_judge.json"
PROMPT = ("You audit an answer for hallucinations. Chunks:\n{ctx}\n\nQuestion: {q}\n\nAnswer sentences:\n{sents}\n\n"
          "A sentence is SUPPORTED if the chunks state it, or it follows from the chunks by plain logic or arithmetic "
          "(a comparison, a count, a date). It is UNSUPPORTED if it adds a fact, name, number or rule that no chunk "
          "provides, or contradicts a chunk. Reply with only JSON: {{\"unsupported\": [sentence numbers]}}")


async def main() -> None:
    rows = [r for r in rr.read(rr.ANSWERS) if r["category"] != "negative" and r["response"] != rr.ABSTAIN_TEXT]
    released, vetoed = [], 0
    for r in rows:
        ctxs = [(str(i), t) for i, t in enumerate(r["retrieved_contexts"])]
        if verify(r["response"], with_question(r["question"], ctxs), entailment).grounded:
            released.append(r)
        else:
            vetoed += 1
    client, sem = rr.make_client(), asyncio.Semaphore(rr.WORKERS)

    async def one(r: dict) -> tuple[int, int, list[str]]:
        sents = [x for x in segments(r["response"]) if x.strip() != rr.ABSTAIN_TEXT]  # an abstention asserts nothing
        ctx = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(r["retrieved_contexts"], 1))
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1))
        async with sem:
            text = await rr.complete(client, rr.JUDGE, PROMPT.format(ctx=ctx, q=r["question"], sents=listing))
        m = re.search(r"\{.*\}", text or "", re.S)
        bad = [int(n) for n in json.loads(m.group(0)).get("unsupported", [])] if m else []
        return len(sents), len(bad), [f"{r['id']}: {sents[n - 1]}" for n in bad if 0 < n <= len(sents)]

    res = await asyncio.gather(*(one(r) for r in released))
    claims, bad = sum(x[0] for x in res), sum(x[1] for x in res)
    rep = {"judge": rr.JUDGE, "answerable_drafts": len(rows), "released": len(released), "vetoed": vetoed,
           "claims": claims, "unsupported": bad, "rate": bad / claims if claims else None,
           "unsupported_sentences": [s for x in res for s in x[2]]}
    OUT.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "unsupported_sentences"})


if __name__ == "__main__":
    asyncio.run(main())
