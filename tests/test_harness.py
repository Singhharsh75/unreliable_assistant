"""Gate 4 — ToolExecutionHarness: strict validation, regex repair, RBAC, UUIDv4 idempotency."""
import asyncio
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field

from contracts.models import CallContext, LOT_RE, SKU_RE
from harness import HarnessError, ToolExecutionHarness, ToolSpec
from harness.errors import (
    AUTO_REPAIR_FAILED,
    HITL_REQUIRED,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    RBAC_DENIED,
    SCHEMA_VALIDATION_FAILED,
)
from harness.tool_harness import repair_ids

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


def ctx(key=None, tenant="tenant_acme"):
    return {
        "tenant_id": tenant,
        "user_id": "user_1",
        "session_id": str(uuid4()),
        "idempotency_key": str(key or uuid4()),
        "traceparent": TP,
    }


class LotIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    lot_id: str = Field(pattern=LOT_RE)
    count: int = 1


class SkuIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    sku: str = Field(pattern=SKU_RE)


class SubmitIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    action: str
    payload: dict
    justification: str


class SpyHandler:
    def __init__(self, result="ok", fail_with=None):
        self.calls = 0
        self.result = result
        self.fail_with = fail_with

    def __call__(self, model):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return {"echo": model.model_dump(mode="json", exclude={"ctx"}), "result": self.result}


def harness(lot=None, sku=None, submit=None):
    tools = [
        ToolSpec("lookup_lot", LotIn, "TIER_1_READ", lot or SpyHandler()),
        ToolSpec("sku_info", SkuIn, "TIER_2_COMPUTE", sku or SpyHandler()),
        ToolSpec(
            "submit_transaction",
            SubmitIn,
            "TIER_3_MUTATE",
            submit or SpyHandler("PENDING_APPROVAL"),
            min_caller_tier="TIER_2_COMPUTE",
            hitl_gate=True,
        ),
        ToolSpec("raw_mutate", LotIn, "TIER_3_MUTATE", SpyHandler()),  # Tier 3 without a HITL gate
    ]
    return ToolExecutionHarness(tools)


def run(coro):
    return asyncio.run(coro)


# --- step 1: strict validation -------------------------------------------------

def test_valid_call_runs_handler():
    h = harness()
    res = run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "LOT-88392"}, caller_tier="TIER_1_READ"))
    assert res.value["result"] == "ok" and not res.replayed and res.repairs == ()


def test_strict_mode_rejects_wrong_type_without_repair():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "LOT-88392", "count": "3"}, caller_tier="TIER_1_READ"))
    assert e.value.code == SCHEMA_VALIDATION_FAILED


def test_extra_fields_rejected():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "LOT-88392", "rogue": 1}, caller_tier="TIER_1_READ"))
    assert e.value.code == SCHEMA_VALIDATION_FAILED


# --- step 2: deterministic regex repair ----------------------------------------

def test_lot_repair_lot_44_to_canonical():
    h = harness()
    res = run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "lot 44"}, caller_tier="TIER_1_READ"))
    assert res.value["echo"]["lot_id"] == "LOT-00044"
    assert res.repairs == ({"path": "$.lot_id", "before": "lot 44", "after": "LOT-00044"},)


def test_sku_repair_sku99_to_canonical():
    h = harness()
    res = run(h.call("sku_info", {"ctx": ctx(), "sku": "SKU99"}, caller_tier="TIER_2_COMPUTE"))
    assert res.value["echo"]["sku"] == "SKU-099"


def test_repair_leaves_free_text_alone():
    out, repairs = repair_ids({"query": "is lot 44 released?"})
    assert out == {"query": "is lot 44 released?"} and repairs == []


def test_repair_that_still_fails_is_auto_repair_failed():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "lot 44", "count": "many"}, caller_tier="TIER_1_READ"))
    assert e.value.code == AUTO_REPAIR_FAILED


def test_unrepairable_id_is_schema_failure():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "LOT-1234567"}, caller_tier="TIER_1_READ"))
    assert e.value.code == SCHEMA_VALIDATION_FAILED


# --- step 3: RBAC --------------------------------------------------------------

def test_tier1_caller_denied_tier2_tool():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("sku_info", {"ctx": ctx(), "sku": "SKU-001"}, caller_tier="TIER_1_READ"))
    assert e.value.code == RBAC_DENIED


def test_operator_can_request_tier3_via_gate_only():
    h = harness()
    res = run(h.call("submit_transaction", {"ctx": ctx(), "action": "release", "payload": {}, "justification": "x"},
                     caller_tier="TIER_2_COMPUTE"))
    assert res.value["result"] == "PENDING_APPROVAL"


def test_tier1_cannot_request_tier3():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("submit_transaction", {"ctx": ctx(), "action": "release", "payload": {}, "justification": "x"},
                   caller_tier="TIER_1_READ"))
    assert e.value.code == RBAC_DENIED


def test_tier3_tool_without_gate_never_runs_even_for_reviewer_tier():
    spy = SpyHandler()
    h = ToolExecutionHarness([ToolSpec("raw_mutate", LotIn, "TIER_3_MUTATE", spy)])
    with pytest.raises(HarnessError) as e:
        run(h.call("raw_mutate", {"ctx": ctx(), "lot_id": "LOT-00001"}, caller_tier="TIER_3_MUTATE"))
    assert e.value.code == HITL_REQUIRED and spy.calls == 0


# --- step 4: UUIDv4 idempotency ------------------------------------------------

def test_replayed_key_returns_cached_result_without_rerunning():
    spy = SpyHandler()
    h = ToolExecutionHarness([ToolSpec("lookup_lot", LotIn, "TIER_1_READ", spy)])
    c = ctx()
    first = run(h.call("lookup_lot", {"ctx": c, "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    second = run(h.call("lookup_lot", {"ctx": c, "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    assert spy.calls == 1
    assert not first.replayed and second.replayed
    assert second.value == first.value


def test_same_key_different_args_is_rejected():
    h = harness()
    c = ctx()
    run(h.call("lookup_lot", {"ctx": c, "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": c, "lot_id": "LOT-00002"}, caller_tier="TIER_1_READ"))
    assert e.value.code == INVALID_PARAMS


def test_key_is_scoped_per_tenant():
    spy = SpyHandler()
    h = ToolExecutionHarness([ToolSpec("lookup_lot", LotIn, "TIER_1_READ", spy)])
    key = uuid4()
    run(h.call("lookup_lot", {"ctx": ctx(key, "tenant_acme"), "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    run(h.call("lookup_lot", {"ctx": ctx(key, "tenant_globex"), "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    assert spy.calls == 2


def test_non_v4_idempotency_key_rejected():
    h = harness()
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": ctx(key="12345678-1234-1234-1234-123456789abc"), "lot_id": "LOT-00001"},
                   caller_tier="TIER_1_READ"))
    assert e.value.code == INVALID_PARAMS


def test_failed_call_is_not_cached_so_retry_executes():
    spy = SpyHandler(fail_with=RuntimeError("boom"))
    h = ToolExecutionHarness([ToolSpec("lookup_lot", LotIn, "TIER_1_READ", spy)])
    c = ctx()
    with pytest.raises(HarnessError) as e:
        run(h.call("lookup_lot", {"ctx": c, "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    assert e.value.code == INTERNAL_ERROR
    spy.fail_with = None
    res = run(h.call("lookup_lot", {"ctx": c, "lot_id": "LOT-00001"}, caller_tier="TIER_1_READ"))
    assert not res.replayed and spy.calls == 2


def test_unknown_tool_rejected():
    with pytest.raises(HarnessError):
        run(harness().call("drop_table", {"ctx": ctx()}, caller_tier="TIER_3_MUTATE"))


def test_spec_requires_ctx_field():
    class NoCtx(BaseModel):
        lot_id: str

    with pytest.raises(ValueError):
        ToolSpec("x", NoCtx, "TIER_1_READ", SpyHandler())
