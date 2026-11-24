"""Frozen MCP contracts from design.md §1.2 (v1.0)."""
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

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


class SubmitTransactionIn(_In):
    action: TransactionAction
    payload: dict[str, str | int | float | bool | None]
    justification: str = Field(min_length=1, max_length=1000)


class GetApprovalIn(_In):
    approval_id: str = Field(pattern=r"^apr_[a-f0-9]{32}$")


class AppendAuditIn(_In):
    event: dict
