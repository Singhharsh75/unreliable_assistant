"""Bounded ReAct loop as an explicit state machine (design.md ADR-004, Gate 4 loop engineering).

  INIT -> TRIAGE -> REASONING -> TOOL_CALL -> OBSERVATION -> VERIFY -> DONE
                       ^               |                       |
                       +---------------+  (loop)               +-> HALTED / HALTED_BREAKER

Limits (defaults from the blueprint): max_iterations=5, per_step_timeout=120 ms,
total_loop_budget=450 ms, llm_step_timeout=250 ms, breaker trips after 2 identical failures.
Every tool call goes through ToolExecutionHarness. The loop never calls itself: each
REASONING pass is an iteration of this while-loop, and a repeated observation halts it.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel

from harness.errors import (
    AUTO_REPAIR_FAILED,
    INJECTION_BLOCKED,
    HarnessError,
    HITL_REQUIRED,
    PII_BLOCKED,
    RBAC_DENIED,
    TENANT_SCOPE_VIOLATION,
)
from harness.tool_harness import Tier, ToolExecutionHarness, canonical_hash


class State(str, Enum):
    INIT = "INIT"
    TRIAGE = "TRIAGE"
    REASONING = "REASONING"
    TOOL_CALL = "TOOL_CALL"
    OBSERVATION = "OBSERVATION"
    VERIFY = "VERIFY"
    DONE = "DONE"
    HALTED = "HALTED"
    HALTED_BREAKER = "HALTED_BREAKER"


# Errors that must not be retried by the loop. Each one halts the run immediately.
NON_RETRYABLE = frozenset(
    {RBAC_DENIED, HITL_REQUIRED, TENANT_SCOPE_VIOLATION, PII_BLOCKED, INJECTION_BLOCKED, AUTO_REPAIR_FAILED}
)


@dataclass(frozen=True)
class ToolAction:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    text: str


class Planner(Protocol):
    async def reason(self, goal: str, observations: list[dict[str, Any]]) -> ToolAction | FinalAnswer: ...

    async def verify(self, answer: str, observations: list[dict[str, Any]]) -> bool: ...


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int = 5
    per_step_timeout_s: float = 0.120
    total_loop_budget_s: float = 0.450
    llm_step_timeout_s: float = 0.250
    breaker_threshold: int = 2


@dataclass
class LoopResult:
    state: State
    halt_reason: str | None
    answer: str | None
    observations: list[dict[str, Any]]
    iterations: int
    trace: list[State] = field(default_factory=list)


class BoundedLoop:
    def __init__(
        self,
        harness: ToolExecutionHarness,
        planner: Planner,
        config: LoopConfig | None = None,
        clock=time.monotonic,
    ) -> None:
        self.harness = harness
        self.planner = planner
        self.config = config or LoopConfig()
        self.clock = clock

    async def run(self, goal: str, *, base_ctx: dict[str, Any], caller_tier: Tier) -> LoopResult:
        """Run one task. `base_ctx` holds tenant_id, user_id, session_id, traceparent. A fresh UUIDv4 is minted per tool call."""
        cfg = self.config
        t0 = self.clock()
        trace: list[State] = [State.INIT]
        observations: list[dict[str, Any]] = []
        failures: Counter[tuple[str, str]] = Counter()
        last_digest: str | None = None
        iterations = 0

        def remaining() -> float:
            return cfg.total_loop_budget_s - (self.clock() - t0)

        def move(s: State) -> None:
            trace.append(s)

        def halt(s: State, reason: str, answer: str | None = None) -> LoopResult:
            return LoopResult(s, reason, answer, observations, iterations, trace)

        move(State.TRIAGE)  # routing is decided inside planner.reason until Gate 6; the state is recorded for OTel parity

        while True:
            if remaining() <= 0:
                return halt(State.HALTED, "total_loop_budget exceeded")
            if iterations >= cfg.max_iterations:
                return halt(State.HALTED, "max_iterations reached")

            move(State.REASONING)
            iterations += 1
            try:
                action = await asyncio.wait_for(
                    self.planner.reason(goal, observations),
                    timeout=min(cfg.llm_step_timeout_s, max(remaining(), 0.0)),
                )
            except TimeoutError:
                return halt(State.HALTED, "llm_step_timeout")

            if isinstance(action, FinalAnswer):
                move(State.VERIFY)
                try:
                    ok = await asyncio.wait_for(
                        self.planner.verify(action.text, observations),
                        timeout=max(remaining(), 1e-3),
                    )
                except Exception:  # timeout or verifier error: an unverified draft is never released
                    return halt(State.HALTED, "verify_failed")
                if ok:
                    move(State.DONE)
                    return halt(State.DONE, None, action.text)
                return halt(State.HALTED, "verification_veto")

            key = (action.tool, canonical_hash(action.args))
            if remaining() <= 0:
                return halt(State.HALTED, "total_loop_budget exceeded")

            move(State.TOOL_CALL)
            payload = {**action.args, "ctx": {**base_ctx, "idempotency_key": str(uuid4())}}
            try:
                res = await asyncio.wait_for(
                    self.harness.call(action.tool, payload, caller_tier=caller_tier),
                    timeout=min(cfg.per_step_timeout_s, remaining()),
                )
                value = res.value.model_dump(mode="json") if isinstance(res.value, BaseModel) else res.value
                obs = {"tool": action.tool, "ok": True, "result": value}
            except HarnessError as e:
                if e.code in NON_RETRYABLE:
                    return halt(State.HALTED, f"{e.name}: {e.message}")
                failures[key] += 1
                obs = {"tool": action.tool, "ok": False, "error": e.name, "detail": e.message}
            except TimeoutError:
                failures[key] += 1
                obs = {"tool": action.tool, "ok": False, "error": "UPSTREAM_TIMEOUT", "detail": "step budget exceeded"}

            move(State.OBSERVATION)
            observations.append(obs)

            if failures[key] >= cfg.breaker_threshold:
                move(State.HALTED_BREAKER)
                return halt(State.HALTED_BREAKER, f"breaker tripped on {action.tool} after {failures[key]} failures")

            digest = canonical_hash(obs)
            if digest == last_digest:
                # Same observation as the last step: nothing changed, so another REASONING pass cannot make progress.
                return halt(State.HALTED, "no state change since last observation")
            last_digest = digest
