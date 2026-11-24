"""Frozen MCP contracts from design.md §1.2 (v1.0)."""
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

TENANT_RE = r"^tenant_[a-z0-9]{1,32}$"
USER_RE = r"^user_[a-z0-9]{1,64}$"
SKU_RE = r"^SKU-\d{3}$"
LOT_RE = r"^LOT-\d{5}$"
CHUNK_RE = r"^[a-f0-9]{64}$"
# Deterministic abstention (design.md §1.3 step 5): shared by the runtime and the eval so both score the same string.
ABSTAIN_TEXT = "The knowledge base does not cover this question."

Tier = Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"]


class CallContext(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    tenant_id: str = Field(pattern=TENANT_RE)
    user_id: str = Field(pattern=USER_RE)
    session_id: UUID
    idempotency_key: UUID
    traceparent: str = Field(pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$")


class HybridRetrieveIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    query: str = Field(min_length=1, max_length=2000)
    domain: str | None = Field(default=None, pattern=r"^[a-z_]{1,32}$")
    bm25_top_k: int = Field(default=50, ge=1, le=100)
    dense_top_k: int = Field(default=50, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rerank_k: int = Field(default=4, ge=1, le=10)
    min_rerank_score: float = Field(default=0.42, ge=0.0, le=1.0)


class Chunk(BaseModel):
    chunk_hash: str = Field(pattern=CHUNK_RE)
    uri: str = Field(pattern=r"^rag://chunks/[a-f0-9]{64}$")
    text: str = Field(max_length=4096)
    note_id: str
    bm25_score: float | None
    dense_cosine: float | None
    rrf_rank: int
    rerank_score: float


class HybridRetrieveOut(BaseModel):
    chunks: list[Chunk] = Field(max_length=10)
    top_score: float
    abstain: bool
    stage_latency_ms: dict[Literal["bm25", "dense", "rrf", "rerank"], float]


class GraphSubgraphIn(BaseModel):
    """design.md §1.2.3: the connected vault subgraph around the anchor notes, within a token budget."""
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    anchor_note_ids: list[str] = Field(min_length=1, max_length=4)
    depth: int = Field(default=2, ge=1, le=2)  # hard cap at 2 hops
    max_tokens: int = Field(default=320, ge=64, le=800)


class GraphSubgraphOut(BaseModel):
    nodes: list[dict]  # {id, domain, risk_tier, version, summary}
    edges: list[tuple[str, str, str]]  # (src, rel, dst)
    token_count: int
    raw_token_estimate: int  # full text of every reachable note: the dump the subgraph replaces (KPI-26)


class SessionFact(BaseModel):
    """design.md §1.2.4. Only verified facts are persisted."""
    model_config = ConfigDict(strict=True, extra="forbid")
    key: str = Field(pattern=r"^[a-z_]{1,64}$")
    value: str = Field(max_length=512)
    source_chunk_hash: str | None = Field(default=None, pattern=CHUNK_RE)


# --- mcp-state-store inputs (design.md §1.2.4) ------------------------------------

class _In(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext


TransactionAction = Literal["release", "quarantine", "hold", "reject", "open_deviation", "open_capa", "update_status"]
ApprovalStatus = Literal["PENDING_APPROVAL", "APPROVED", "REJECTED"]


class GetSessionFactsIn(_In):
    pass


class PutSessionFactIn(_In):
    fact: SessionFact


class LookupLotIn(_In):
    lot_id: str = Field(pattern=LOT_RE)


class LotPayload(BaseModel):
    """release, quarantine, hold, reject"""
    model_config = ConfigDict(strict=True, extra="forbid")
    lot_id: str = Field(pattern=LOT_RE)


class RecordPayload(LotPayload):
    """open_deviation, open_capa: the reviewer needs to know what happened"""
    # no control or bidirectional-override characters: they can make a reviewer see something else
    description: str = Field(min_length=1, max_length=500, pattern="^[^\\x00-\\x1f\\x7f\u202a-\u202e\u2066-\u2069]+$")


class StatusPayload(LotPayload):
    """update_status"""
    status: Literal["QUARANTINE", "HOLD", "RELEASED", "REJECTED"]


# The three shapes are disjoint (extra fields are forbidden), so the union never picks the wrong one.
PAYLOADS: dict[str, type[LotPayload]] = {
    "release": LotPayload, "quarantine": LotPayload, "hold": LotPayload, "reject": LotPayload,
    "open_deviation": RecordPayload, "open_capa": RecordPayload, "update_status": StatusPayload,
}


class SubmitTransactionIn(_In):
    action: TransactionAction
    payload: LotPayload | RecordPayload | StatusPayload = Field(
        description="release, quarantine, hold, reject: LotPayload. open_deviation, open_capa: RecordPayload. "
                    "update_status: StatusPayload.")
    justification: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _payload_fits_action(self) -> "SubmitTransactionIn":
        if type(self.payload) is not PAYLOADS[self.action]:
            raise ValueError(f"{self.action} needs a {PAYLOADS[self.action].__name__}")
        return self


class GetApprovalIn(_In):
    approval_id: str = Field(pattern=r"^apr_[a-f0-9]{32}$")


class AppendAuditIn(_In):
    event: dict
