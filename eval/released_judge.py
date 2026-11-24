"""KPI-24 on RELEASED answers, judged independently of the NLI verifier (ADR-016).

Release path as in triage/orchestrator.py with NLI: a draft is released only if the verifier grounds every claim;
a vetoed draft becomes the abstention text. An LLM judge (JUDGE_MODEL, which reads the whole chunk set and accepts
conclusions that follow from it) then lists the sentences of each released draft that the chunks do not support.
The judge shares nothing with the NLI model, so a verifier blind spot shows up here instead of cancelling out.

  uv run eval/released_judge.py              # offline release path; writes eval/runs/released_judge.json
  uv run eval/released_judge.py --api URL    # answers come from a running API, so the deployed path is what is judged.
                                             # Start it with INCIDENT_RESPONDER=0: a fired runbook changes the answers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import run_ragas as rr  # noqa: E402
from triage.nli import entailment  # noqa: E402
from triage.verifier import segments, verify, with_question  # noqa: E402

OUT = rr.RUNS / "released_judge.json"
PROMPT = ("You audit an answer for hallucinations. Chunks:\n{ctx}\n\nQuestion: {q}\n\nAnswer sentences:\n{sents}\n\n"
          "A sentence is SUPPORTED if the chunks state it, or it follows from the chunks by plain logic or arithmetic "
          "(a comparison, a count, a date). It is UNSUPPORTED if it adds a fact, name, number or rule that no chunk "
          "provides, or contradicts a chunk. Reply with only JSON: {{\"unsupported\": [sentence numbers]}}")


async def through_api(url: str) -> tuple[list[dict], int, str]:
    """(released rows, not released, generator name). The contexts are the recorded retrievals for the same questions."""
    import httpx

    rows = [r for r in rr.read(rr.INPUTS) if r["category"] != "negative" and not r["abstain"]]
    sem, released = asyncio.Semaphore(4), []
    async with httpx.AsyncClient(base_url=url, timeout=60) as c:
        async def ask(r: dict) -> dict:
            t = r.get("tenant_id") or "tenant_demo"
            async with sem:
                resp = await c.post("/api/v1/agent/query", headers={"X-Tenant-ID": t}, json={
                    "tenant_id": t, "user_id": "user_kpi24", "session_id": str(uuid4()), "query": r["question"]})
            resp.raise_for_status()
            return resp.json()

        generator = (await c.get("/metrics")).json()["generator"]
        if generator != "ClaudeGenerator":
            raise SystemExit(f"the API answers with {generator}; KPI-24 is only judged on Claude answers")
        outs = await asyncio.gather(*(ask(r) for r in rows))
    halted = [o for o in outs if o["status"] == "HALTED"]
    if halted:  # an outage must not read as a good score
        raise SystemExit(f"{len(halted)} of {len(outs)} requests halted, first: {halted[0]['answer']}")
    for r, o in zip(rows, outs):
        if o["route"] == "HYBRID_RAG" and not o["abstained"]:
            released.append({**r, "response": o["answer"]})
    return released, len(rows) - len(released), generator


def offline() -> tuple[list[dict], int, str]:
    rows = [r for r in rr.read(rr.ANSWERS) if r["category"] != "negative" and r["response"] != rr.ABSTAIN_TEXT]
    released, vetoed = [], 0
    for r in rows:
        ctxs = [(str(i), t) for i, t in enumerate(r["retrieved_contexts"])]
        if verify(r["response"], with_question(r["question"], ctxs), entailment).grounded:
            released.append(r)
        else:
            vetoed += 1
    return released, vetoed, rr.GENERATOR


async def main(api: str | None) -> None:
    released, vetoed, generator = await through_api(api) if api else offline()
    client, sem = rr.make_client(), asyncio.Semaphore(rr.WORKERS)

    async def one(r: dict) -> tuple[int, int, list[str], bool]:
        sents = [x for x in segments(r["response"]) if x.strip() != rr.ABSTAIN_TEXT]  # an abstention asserts nothing
        ctx = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(r["retrieved_contexts"], 1))
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1))
        async with sem:
            text = await rr.complete(client, rr.JUDGE, PROMPT.format(ctx=ctx, q=r["question"], sents=listing))
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:  # a refusal or a garbled reply is not a clean verdict: its claims are left out and it is counted
            return 0, 0, [], True
        bad = [int(n) for n in json.loads(m.group(0)).get("unsupported", [])]
        return len(sents), len(bad), [f"{r['id']}: {sents[n - 1]}" for n in bad if 0 < n <= len(sents)], False

    res = await asyncio.gather(*(one(r) for r in released))
    claims, bad = sum(x[0] for x in res), sum(x[1] for x in res)
    rep = {"judge": rr.JUDGE, "path": "api" if api else "offline", "generator": generator,
           "answerable_drafts": len(released) + vetoed, "released": len(released), "vetoed": vetoed,
           "claims": claims, "unsupported": bad, "judge_failed": sum(x[3] for x in res), "rate": bad / claims if claims else None,
           "unsupported_sentences": [s for x in res for s in x[2]]}
    OUT.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "unsupported_sentences"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", metavar="URL")
    asyncio.run(main(ap.parse_args().api))
