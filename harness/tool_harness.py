"""ToolExecutionHarness (design.md §1.2, Gate 4).

Every MCP tool call passes through four steps, in this order:
  1. Pydantic v2 strict validation (on the JSON wire form, as JSON-RPC delivers it)
  2. Deterministic regex auto-repair of ID-shaped values (SKU99 -> SKU-099, lot 44 -> LOT-00044),
     followed by one re-validation
  3. RBAC tier check (caller tier must be >= the tool's required tier; Tier 3 only via the HITL gate)
  4. UUIDv4 idempotency: a repeated key with the same arguments returns the cached result
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ValidationError

from harness.errors import (
    AUTO_REPAIR_FAILED,
    HITL_REQUIRED,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    RBAC_DENIED,
    SCHEMA_VALIDATION_FAILED,
    HarnessError,
)

Tier = Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"]
TIER_RANK: dict[str, int] = {"TIER_1_READ": 1, "TIER_2_COMPUTE": 2, "TIER_3_MUTATE": 3}
METHOD_NOT_FOUND = -32601

# Whole-value rewrites only. Free text is never touched, so a query that merely mentions a lot is unaffected.
_REPAIR_RULES: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (re.compile(r"^\s*lot[\s-]*([0-9]{1,5})\s*$", re.IGNORECASE), lambda m: f"LOT-{int(m.group(1)):05d}"),
    (re.compile(r"^\s*sku[\s-]*([0-9]{1,3})\s*$", re.IGNORECASE), lambda m: f"SKU-{int(m.group(1)):03d}"),
]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_model: type[BaseModel]
    tier: Tier
    handler: Callable[[BaseModel], Any]
    # Tier a caller must hold to invoke the tool. Defaults to `tier`.
    min_caller_tier: Tier | None = None
    # True only for the entry point that writes PENDING_APPROVAL (submit_transaction). Other Tier 3 tools never run directly.
    hitl_gate: bool = False

    def __post_init__(self) -> None:
        if "ctx" not in self.input_model.model_fields:
            raise ValueError(f"{self.name}: input model must carry a `ctx: CallContext` field")

    @property
    def required_tier(self) -> Tier:
        return self.min_caller_tier or self.tier


@dataclass(frozen=True)
class CallResult:
    value: Any
    replayed: bool
    repairs: tuple[dict[str, str], ...] = ()


def canonical_hash(obj: Any) -> str:
    """SHA256 over canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def repair_ids(obj: Any, path: str = "$") -> tuple[Any, list[dict[str, str]]]:
    """Recursively rewrite ID-shaped string values. Returns the new object and a list of repairs made."""
    if isinstance(obj, dict):
        out, repairs = {}, []
        for k, v in obj.items():
            out[k], r = repair_ids(v, f"{path}.{k}")
            repairs += r
        return out, repairs
    if isinstance(obj, list):
        out, repairs = [], []
        for i, v in enumerate(obj):
            new, r = repair_ids(v, f"{path}[{i}]")
            out.append(new)
            repairs += r
        return out, repairs
    if isinstance(obj, str):
        for pattern, rewrite in _REPAIR_RULES:
            m = pattern.match(obj)
            if m:
                fixed = rewrite(m)
                if fixed != obj:
                    return fixed, [{"path": path, "before": obj, "after": fixed}]
                return obj, []
    return obj, []


@dataclass
class ToolExecutionHarness:
    tools: Iterable[ToolSpec]
    _specs: dict[str, ToolSpec] = field(init=False, default_factory=dict)
    # (tenant_id, idempotency_key) -> (args_digest, result). Successful calls only.
    _results: dict[tuple[str, str], tuple[str, Any]] = field(init=False, default_factory=dict)
    _locks: dict[tuple[str, str], asyncio.Lock] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        for spec in self.tools:
            if spec.name in self._specs:
                raise ValueError(f"duplicate tool {spec.name}")
            self._specs[spec.name] = spec

    async def call(self, tool: str, payload: dict[str, Any], *, caller_tier: Tier) -> CallResult:
        spec = self._specs.get(tool)
        if spec is None:
            raise HarnessError(METHOD_NOT_FOUND, f"unknown tool {tool!r}")
        if not isinstance(payload, dict):
            raise HarnessError(INVALID_PARAMS, "payload must be a JSON object")

        # Steps 1 and 2: strict validation, then regex auto-repair and one re-validation.
        model, repairs = self._validate_and_repair(spec, payload)

        # Step 3: RBAC.
        if TIER_RANK[caller_tier] < TIER_RANK[spec.required_tier]:
            raise HarnessError(
                RBAC_DENIED,
                f"{tool} requires {spec.required_tier}; caller holds {caller_tier}. Do not retry.",
            )
        if spec.tier == "TIER_3_MUTATE" and not spec.hitl_gate:
            raise HarnessError(HITL_REQUIRED, f"{tool} is a Tier 3 mutation and must go through the HITL gate")

        # Step 4: UUIDv4 idempotency.
        ctx = model.ctx  # type: ignore[attr-defined]
        key: UUID = ctx.idempotency_key
        if key.version != 4:
            raise HarnessError(INVALID_PARAMS, "idempotency_key must be a UUIDv4")
        scope = (ctx.tenant_id, str(key))
        digest = canonical_hash(model.model_dump(mode="json", exclude={"ctx"}))

        lock = self._locks.setdefault(scope, asyncio.Lock())
        async with lock:  # a concurrent retry with the same key waits, then sees the cached result
            if scope in self._results:
                cached_digest, cached_value = self._results[scope]
                if cached_digest != digest:
                    raise HarnessError(
                        INVALID_PARAMS, "idempotency_key was already used with different arguments"
                    )
                return CallResult(value=cached_value, replayed=True, repairs=tuple(repairs))

            try:
                if inspect.iscoroutinefunction(spec.handler):
                    value = await spec.handler(model)
                else:
                    value = await asyncio.to_thread(spec.handler, model)
            except HarnessError:
                raise
            except Exception as exc:  # a crashing tool is a failed step, not a crashed loop
                raise HarnessError(INTERNAL_ERROR, f"{tool} raised {type(exc).__name__}") from exc
            self._results[scope] = (digest, value)
            return CallResult(value=value, replayed=False, repairs=tuple(repairs))

    @staticmethod
    def _validate_and_repair(spec: ToolSpec, payload: dict[str, Any]) -> tuple[BaseModel, list[dict[str, str]]]:
        try:
            return spec.input_model.model_validate_json(json.dumps(payload)), []
        except ValidationError as first:
            repaired, repairs = repair_ids(payload)
            if not repairs:
                raise HarnessError(
                    SCHEMA_VALIDATION_FAILED,
                    f"{spec.name}: strict validation failed",
                    {"errors": _summarise(first)},
                ) from first
            try:
                return spec.input_model.model_validate_json(json.dumps(repaired)), repairs
            except ValidationError as second:
                raise HarnessError(
                    AUTO_REPAIR_FAILED,
                    f"{spec.name}: regex repair did not produce a valid request; ask the user to clarify",
                    {"repairs": repairs, "errors": _summarise(second)},
                ) from second


def _summarise(err: ValidationError) -> list[dict[str, str]]:
    return [{"loc": ".".join(str(p) for p in e["loc"]), "type": e["type"]} for e in err.errors()]
