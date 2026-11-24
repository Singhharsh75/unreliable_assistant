"""KPI-14, KPI-17, KPI-18 from real LLM-planner runs (design.md ADR-017).

Runs every task in eval/planner_tasks.jsonl through the real BoundedLoop with the LLM planner, the execution subagent's
scoped harness and an in-memory StateStore, then reports:
  KPI-14  share of tool calls that validate on the first pass: no regex repair, no schema failure, no unknown tool
  KPI-17  mean ReAct steps (REASONING passes) over tasks that completed (DONE, or a PENDING_APPROVAL record exists)
  KPI-18  mean cosine between the goal and each step (thought + chosen action), bge-large embeddings

  uv run eval/planner_eval.py [--limit N]    # real LLM calls (GENERATOR_MODEL); writes eval/runs/planner_eval.json

The loop budgets are widened for this run (a remote LLM cannot answer inside the blueprint's 250 ms step): ADR-017.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import run_ragas as rr  # noqa: E402
from harness import BoundedLoop, LoopConfig, State, ToolExecutionHarness  # noqa: E402
from harness.errors import AUTO_REPAIR_FAILED, SCHEMA_VALIDATION_FAILED, HarnessError  # noqa: E402
from harness.loop import FinalAnswer  # noqa: E402
from mcp_servers.state_store import StateStore  # noqa: E402
from triage.llm_planner import LLMPlanner  # noqa: E402
from triage.nli import entailment  # noqa: E402
from triage.orchestrator import _tool_output_ok, _VerifiedPlanner  # noqa: E402
from triage.subagents import SUBAGENTS, ScopedHarness  # noqa: E402

TASKS, OUT = rr.EVAL / "planner_tasks.jsonl", rr.RUNS / "planner_eval.json"
TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
TENANT = "tenant_eval"
LOTS = {"LOT-00044": ("QUARANTINE", "SOP-DEV-01"), "LOT-00312": ("HOLD", "SOP-QA-014"), "LOT-88392": ("RELEASED", "SOP-REL-02"),
        "LOT-00007": ("RELEASED", "SOP-REL-02"), "LOT-00001": ("HOLD", "SOP-QA-014")}
CFG = LoopConfig(max_iterations=5, per_step_timeout_s=5.0, total_loop_budget_s=180.0, llm_step_timeout_s=60.0)


class Counting:
    """Wraps the scoped harness and classifies every call the planner makes."""

    def __init__(self, inner: ScopedHarness) -> None:
        self.inner, self.calls = inner, []

    async def call(self, tool: str, payload: dict[str, Any], *, caller_tier):
        rec = {"tool": tool, "first_pass": True}
        self.calls.append(rec)
        if tool not in self.inner.spec.allowed_tools:
            rec["first_pass"] = False
        try:
            res = await self.inner.call(tool, payload, caller_tier=caller_tier)
        except HarnessError as e:
            if e.code in (SCHEMA_VALIDATION_FAILED, AUTO_REPAIR_FAILED):
                rec["first_pass"] = False
            raise
        if res.repairs:
            rec["first_pass"] = False
        return res


class Recording:
    """Records, per REASONING pass, the planner's thought and chosen action for the drift audit."""

    def __init__(self, inner: LLMPlanner) -> None:
        self.inner, self.steps = inner, []

    async def reason(self, prompt):
        action = await self.inner.reason(prompt)
        what = action.text if isinstance(action, FinalAnswer) else f"{action.tool} {json.dumps(action.args, sort_keys=True)}"
        self.steps.append(f"{self.inner.thoughts[-1] if self.inner.thoughts else ''} {what}".strip())
        return action

    async def verify(self, answer, observations):
        return await self.inner.verify(answer, observations)


async def run_task(task: dict, client, sem: asyncio.Semaphore) -> dict:
    store = StateStore()
    for lot, (status, ref) in LOTS.items():
        store.upsert_lot(TENANT, lot, status, ref)
    harness = ToolExecutionHarness(store.specs(pool=None))
    models = {s.name: s.input_model for s in store.specs(pool=None) if s.name in SUBAGENTS["execution"].allowed_tools}

    async def complete(system: str, user: str) -> str | None:
        async with sem:
            return await rr.complete(client, rr.GENERATOR, user, system=system)

    planner = Recording(LLMPlanner(complete, models, entailment, record_thoughts=True))
    scoped = Counting(ScopedHarness(harness, SUBAGENTS["execution"]))
    loop = BoundedLoop(scoped, _VerifiedPlanner(planner, entailment), CFG, observation_gate=_tool_output_ok)  # type: ignore[arg-type]
    base = {"tenant_id": TENANT, "user_id": "user_eval", "session_id": str(uuid4()), "traceparent": TP}
    res = await loop.run(task["goal"], base_ctx=base, caller_tier="TIER_2_COMPUTE")
    pending = any(o["tool"] == "submit_transaction" and o.get("ok") for o in res.observations)
    called = [c["tool"] for c in scoped.calls]
    return {"id": task["id"], "kind": task["kind"], "goal": task["goal"], "state": res.state.value, "halt": res.halt_reason,
            "iterations": res.iterations, "completed": res.state is State.DONE or pending, "pending_approval": pending,
            "tools": called, "expected_ok": all(t in called for t in task["expect_tools"]),
            "calls": len(scoped.calls), "first_pass": sum(c["first_pass"] for c in scoped.calls),
            "steps": planner.steps, "answer": res.answer}


async def main(limit: int) -> None:
    tasks = rr.read(TASKS)[:limit or None]
    client, sem = rr.make_client(), asyncio.Semaphore(4)
    rows = await asyncio.gather(*(run_task(t, client, sem) for t in tasks))
    from mcp_servers.rag_server import embed
    import numpy as np

    sims, vecs = [], {}
    for r in rows:
        if r["steps"]:
            vecs[r["id"]] = v = embed([r["goal"]] + r["steps"])
            r["drift"] = [float(np.dot(v[0], x)) for x in v[1:]]
            sims += r["drift"]
    # control: the same goals against the steps of other tasks (3 per goal, fixed seed) show what "unrelated" scores
    rng, ids, ctl = random.Random(1), list(vecs), []
    for i in ids:
        for _ in range(3):
            j = rng.choice([k for k in ids if k != i])
            ctl += [float(np.dot(vecs[i][0], x)) for x in vecs[j][1:]]
    first = [r["drift"][0] for r in rows if r.get("drift")]
    last = [r["drift"][-1] for r in rows if r.get("drift")]
    done = [r for r in rows if r["completed"]]
    calls, ok = sum(r["calls"] for r in rows), sum(r["first_pass"] for r in rows)
    rep = {"generator": rr.GENERATOR, "tasks": len(rows), "completed": len(done), "expected_tools_used": sum(r["expected_ok"] for r in rows),
           "kpi14_first_pass": ok / calls if calls else None, "tool_calls": calls,
           "kpi17_mean_steps": sum(r["iterations"] for r in done) / len(done) if done else None,
           "kpi18_mean_cosine": float(np.mean(sims)) if sims else None, "steps_scored": len(sims),
           "kpi18_unrelated_baseline": float(np.mean(ctl)) if ctl else None,
           "kpi18_first_step": float(np.mean(first)) if first else None, "kpi18_last_step": float(np.mean(last)) if last else None,
           "rows": rows}
    OUT.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "rows"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args().limit))
