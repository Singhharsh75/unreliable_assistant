"""KPI-14, KPI-17, KPI-18 from real LLM-planner runs (design.md ADR-017).

Runs every task in eval/planner_tasks.jsonl through the real BoundedLoop with the LLM planner, the execution subagent's
scoped harness and an in-memory StateStore, then reports:
  KPI-14  share of tool calls that validate on the first pass: no regex repair, no schema failure, no unknown tool
  KPI-17  mean ReAct steps (REASONING passes) over tasks that completed (DONE, or a PENDING_APPROVAL record exists)
  KPI-18  chance (AUC) that a step is closer to its own goal than another task's step is, bge-large embeddings (ADR-019);
          the mean goal cosine and the unrelated-task baseline are reported next to it

  uv run eval/planner_eval.py [--limit N]    # real LLM calls (GENERATOR_MODEL); writes eval/runs/planner_eval.json

  uv run eval/planner_eval.py --rescore      # KPI-18 again from the recorded steps; no LLM calls

The loop budgets are widened for this run (a remote LLM cannot answer inside the blueprint's 250 ms step): ADR-017.
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
        self.steps.append(f"{action.thought} {what}".strip())
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

    planner = Recording(LLMPlanner(complete, models, entailment))
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


def alignment(rows: list[dict]) -> dict[str, Any]:
    """KPI-18 over every (goal, step) pair: a step's own goal against all other tasks' steps for that goal."""
    import numpy as np

    from mcp_servers.rag_server import embed

    rows = [r for r in rows if r["steps"]]
    goals, own, unrelated, first, last = embed([r["goal"] for r in rows]), [], [], [], []
    for i, r in enumerate(rows):
        sims = embed(r["steps"]) @ goals.T  # unit vectors: steps x goals
        r["drift"] = sims[:, i].tolist()
        own += r["drift"]
        unrelated += np.delete(sims, i, axis=1).ravel().tolist()
        first.append(r["drift"][0])
        last.append(r["drift"][-1])
    a, b = np.array(own), np.array(unrelated)
    mean = {"kpi18_mean_cosine": a, "kpi18_unrelated_baseline": b, "kpi18_first_step": first, "kpi18_last_step": last}
    auc = float((a[:, None] > b).mean() + (a[:, None] == b).mean() / 2) if len(a) and len(b) else None
    return {"kpi18_auc": auc, "steps_scored": len(own), **{k: float(np.mean(v)) if len(v) else None for k, v in mean.items()}}


async def main(limit: int) -> None:
    tasks = rr.read(TASKS)[:limit or None]
    client, sem = rr.make_client(), asyncio.Semaphore(4)
    rows = await asyncio.gather(*(run_task(t, client, sem) for t in tasks))
    done = [r for r in rows if r["completed"]]
    calls, ok = sum(r["calls"] for r in rows), sum(r["first_pass"] for r in rows)
    rep = {"generator": rr.GENERATOR, "tasks": len(rows), "completed": len(done), "expected_tools_used": sum(r["expected_ok"] for r in rows),
           "kpi14_first_pass": ok / calls if calls else None, "tool_calls": calls,
           "kpi17_mean_steps": sum(r["iterations"] for r in done) / len(done) if done else None,
           **alignment(rows),
           "rows": rows}
    OUT.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "rows"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    if args.rescore:
        rep = json.loads(OUT.read_text())
        rep.update(alignment(rep["rows"]))
        OUT.write_text(json.dumps(rep, indent=1))
        print({k: v for k, v in rep.items() if k.startswith("kpi18")})
    else:
        asyncio.run(main(args.limit))
