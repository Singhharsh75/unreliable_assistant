"""Gate 9: mcp-state-store (design.md §1.2.4) with the asynchronous HITL gate.

SQLite locally; Postgres in dev and prod (same schema). Every row is keyed by tenant_id and every read filters on it.
- submit_transaction never executes the action. It writes PENDING_APPROVAL and returns; the agent session pauses.
- A Reviewer decides through `decide` (the Reviewer UI/API, not an agent tool). The requester cannot approve their
  own request, and a decision is final. A request older than the TTL expires to REJECTED (FMEA #14), never approved.
- The audit log is a per-tenant hash chain: hash = SHA256(prev_hash + canonical(event)). `verify_chain` re-derives it.

shortcut: an APPROVED request is only recorded; the worker that applies the action (design.md §1.2.4) is not built.
Add it before any Tier 3 action has to take effect.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from contracts.models import (
    AppendAuditIn,
    GetApprovalIn,
    GetSessionFactsIn,
    LookupLotIn,
    PutSessionFactIn,
    SessionFact,
    SubmitTransactionIn,
)
from harness.errors import INVALID_PARAMS, RBAC_DENIED, TENANT_SCOPE_VIOLATION, HarnessError
from harness.tool_harness import ToolSpec

GENESIS = "0" * 64
APPROVAL_TTL = timedelta(hours=24)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, session_id TEXT NOT NULL,
    key TEXT NOT NULL, value TEXT NOT NULL, source_chunk_hash TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_facts_scope ON session_facts (tenant_id, session_id);
CREATE TABLE IF NOT EXISTS lots (
    tenant_id TEXT NOT NULL, lot_id TEXT NOT NULL, status TEXT NOT NULL, protocol_ref TEXT, updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, lot_id));
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
    action TEXT NOT NULL, payload TEXT NOT NULL, justification TEXT NOT NULL, status TEXT NOT NULL,
    reviewer_id TEXT, decided_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, event TEXT NOT NULL,
    prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_audit_tenant ON audit (tenant_id, seq);
"""


def _now() -> datetime:
    return datetime.now(UTC)


class StateStore:
    def __init__(self, path: str = ":memory:", clock=_now, approval_ttl: timedelta = APPROVAL_TTL) -> None:
        # One connection, serialized by a lock, so sync callers on worker threads are safe.
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self.clock, self.approval_ttl = clock, approval_ttl

    # --- internals ------------------------------------------------------------

    def _q(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _audit(self, tenant_id: str, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")  # prev_hash read and insert must not interleave
            try:
                row = self._db.execute(
                    "SELECT hash FROM audit WHERE tenant_id = ? ORDER BY seq DESC LIMIT 1", (tenant_id,)).fetchone()
                prev = row[0] if row else GENESIS
                body = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
                digest = canonical_hash_chain(prev, body)
                cur = self._db.execute("INSERT INTO audit (tenant_id, event, prev_hash, hash) VALUES (?, ?, ?, ?)",
                                       (tenant_id, body, prev, digest))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return {"seq": cur.lastrowid, "hash": digest}

    def _violation(self, ctx, resource: str) -> HarnessError:
        self._audit(ctx.tenant_id, {"type": "security.tenant_scope_violation", "user_id": ctx.user_id,
                                    "resource": resource})
        return HarnessError(TENANT_SCOPE_VIOLATION, f"{resource} is not owned by this tenant")

    def _expire(self, approval_id: str, tenant_id: str) -> None:
        cutoff = (self.clock() - self.approval_ttl).isoformat()
        now = self.clock().isoformat()
        with self._lock:
            n = self._db.execute(
                "UPDATE approvals SET status = 'REJECTED', reviewer_id = 'system:expired', decided_at = ?"
                " WHERE approval_id = ? AND status = 'PENDING_APPROVAL' AND created_at < ?",
                (now, approval_id, cutoff)).rowcount
        if n:
            self._audit(tenant_id, {"type": "hitl.expired", "approval_id": approval_id})

    # --- tools (agent-facing, through the harness). Sync on purpose: blocking DB work runs on the db pool's
    # threads, never on the event loop. ------------------------------------------------------------------

    def get_session_facts(self, m: GetSessionFactsIn) -> dict[str, Any]:
        rows = self._q("SELECT key, value, source_chunk_hash FROM session_facts WHERE tenant_id = ? AND session_id = ?"
                       " ORDER BY id", (m.ctx.tenant_id, str(m.ctx.session_id)))
        latest = {k: SessionFact(key=k, value=v, source_chunk_hash=h) for k, v, h in rows}
        return {"facts": [f.model_dump() for f in latest.values()]}

    def put_session_fact(self, m: PutSessionFactIn) -> dict[str, Any]:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO session_facts (tenant_id, session_id, key, value, source_chunk_hash, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (m.ctx.tenant_id, str(m.ctx.session_id), m.fact.key, m.fact.value, m.fact.source_chunk_hash,
                 self.clock().isoformat()))
        return {"fact_id": cur.lastrowid}

    def lookup_lot(self, m: LookupLotIn) -> dict[str, Any]:
        rows = self._q("SELECT tenant_id, status, protocol_ref, updated_at FROM lots WHERE lot_id = ?", (m.lot_id,))
        own = [r for r in rows if r[0] == m.ctx.tenant_id]
        if own:
            _, status, ref, updated = own[0]
            return {"lot_id": m.lot_id, "status": status, "protocol_ref": ref, "updated_at": updated}
        if rows:
            raise self._violation(m.ctx, m.lot_id)
        raise HarnessError(INVALID_PARAMS, f"unknown lot {m.lot_id}")

    def submit_transaction(self, m: SubmitTransactionIn) -> dict[str, Any]:
        """The only Tier 3 entry point. Writes PENDING_APPROVAL; nothing is executed here."""
        approval_id = f"apr_{uuid4().hex}"
        with self._lock:
            self._db.execute(
                "INSERT INTO approvals (approval_id, tenant_id, user_id, session_id, action, payload, justification,"
                " status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING_APPROVAL', ?)",
                (approval_id, m.ctx.tenant_id, m.ctx.user_id, str(m.ctx.session_id), m.action,
                 json.dumps(m.payload, sort_keys=True), m.justification, self.clock().isoformat()))
        self._audit(m.ctx.tenant_id, {"type": "hitl.requested", "approval_id": approval_id, "user_id": m.ctx.user_id,
                                      "action": m.action, "payload": m.payload})
        return {"approval_id": approval_id, "status": "PENDING_APPROVAL"}

    def get_approval(self, m: GetApprovalIn) -> dict[str, Any]:
        return self.approval(m.ctx.tenant_id, m.approval_id, user_id=m.ctx.user_id)

    def append_audit(self, m: AppendAuditIn) -> dict[str, Any]:
        return self._audit(m.ctx.tenant_id, {**m.event, "user_id": m.ctx.user_id, "traceparent": m.ctx.traceparent})

    # --- Reviewer API and admin (not exposed to agents) -------------------------------

    def approval(self, tenant_id: str, approval_id: str, *, user_id: str = "system") -> dict[str, Any]:
        rows = self._q("SELECT tenant_id FROM approvals WHERE approval_id = ?", (approval_id,))
        if not rows:
            raise HarnessError(INVALID_PARAMS, f"unknown approval {approval_id}")
        if rows[0][0] != tenant_id:
            self._audit(tenant_id, {"type": "security.tenant_scope_violation", "user_id": user_id,
                                    "resource": approval_id})
            raise HarnessError(TENANT_SCOPE_VIOLATION, f"{approval_id} is not owned by this tenant")
        self._expire(approval_id, tenant_id)
        (status, reviewer, decided, action, payload), = self._q(
            "SELECT status, reviewer_id, decided_at, action, payload FROM approvals WHERE approval_id = ?",
            (approval_id,))
        return {"approval_id": approval_id, "status": status, "reviewer_id": reviewer, "decided_at": decided,
                "action": action, "payload": json.loads(payload)}

    def pending(self, tenant_id: str) -> list[dict[str, Any]]:
        ids = self._q("SELECT approval_id FROM approvals WHERE tenant_id = ? AND status = 'PENDING_APPROVAL'"
                      " ORDER BY created_at", (tenant_id,))
        out = [self.approval(tenant_id, i) for (i,) in ids]
        return [a for a in out if a["status"] == "PENDING_APPROVAL"]

    def decide(self, tenant_id: str, approval_id: str, *, reviewer_id: str, reviewer_role: str,
               decision: Literal["APPROVED", "REJECTED"], note: str = "") -> dict[str, Any]:
        if reviewer_role != "reviewer":
            raise HarnessError(RBAC_DENIED, "only a Reviewer can decide a pending Tier 3 request")
        if decision not in ("APPROVED", "REJECTED"):
            raise HarnessError(INVALID_PARAMS, "decision must be APPROVED or REJECTED")
        current = self.approval(tenant_id, approval_id, user_id=reviewer_id)  # tenant check and TTL expiry
        (requester,), = self._q("SELECT user_id FROM approvals WHERE approval_id = ?", (approval_id,))
        if requester == reviewer_id:
            raise HarnessError(RBAC_DENIED, "a requester cannot approve their own request (four-eyes rule)")
        if current["status"] != "PENDING_APPROVAL":
            raise HarnessError(INVALID_PARAMS, f"{approval_id} is already {current['status']}")
        with self._lock:
            n = self._db.execute(
                "UPDATE approvals SET status = ?, reviewer_id = ?, decided_at = ?"
                " WHERE approval_id = ? AND status = 'PENDING_APPROVAL'",
                (decision, reviewer_id, self.clock().isoformat(), approval_id)).rowcount
        if n == 0:  # another reviewer decided between the read and the update
            raise HarnessError(INVALID_PARAMS, f"{approval_id} was decided concurrently")
        self._audit(tenant_id, {"type": "hitl.decided", "approval_id": approval_id, "decision": decision,
                                "reviewer_id": reviewer_id, "note": note})
        return self.approval(tenant_id, approval_id, user_id=reviewer_id)

    def upsert_lot(self, tenant_id: str, lot_id: str, status: str, protocol_ref: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO lots VALUES (?, ?, ?, ?, ?) ON CONFLICT (tenant_id, lot_id) DO UPDATE SET"
                " status = excluded.status, protocol_ref = excluded.protocol_ref, updated_at = excluded.updated_at",
                (tenant_id, lot_id, status, protocol_ref, self.clock().isoformat()))

    def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
        rows = self._q("SELECT event FROM audit WHERE tenant_id = ? ORDER BY seq", (tenant_id,))
        return [json.loads(e) for (e,) in rows]

    def verify_chain(self, tenant_id: str) -> bool:
        """KPI-19 audit-log integrity: every link re-derives from the previous hash and the stored event."""
        prev = GENESIS
        for event, stored_prev, digest in self._q(
                "SELECT event, prev_hash, hash FROM audit WHERE tenant_id = ? ORDER BY seq", (tenant_id,)):
            if stored_prev != prev or canonical_hash_chain(prev, event) != digest:
                return False
            prev = digest
        return True

    def specs(self, pool: str | None = "db") -> list[ToolSpec]:
        return [
            ToolSpec("get_session_facts", GetSessionFactsIn, "TIER_1_READ", self.get_session_facts, pool=pool),
            ToolSpec("put_session_fact", PutSessionFactIn, "TIER_2_COMPUTE", self.put_session_fact, pool=pool),
            ToolSpec("lookup_lot", LookupLotIn, "TIER_1_READ", self.lookup_lot, pool=pool),
            ToolSpec("submit_transaction", SubmitTransactionIn, "TIER_3_MUTATE", self.submit_transaction,
                     min_caller_tier="TIER_2_COMPUTE", hitl_gate=True, pool=pool),
            ToolSpec("get_approval", GetApprovalIn, "TIER_1_READ", self.get_approval, pool=pool),
            ToolSpec("append_audit", AppendAuditIn, "TIER_1_READ", self.append_audit, pool=pool),
        ]


def canonical_hash_chain(prev_hash: str, canonical_event: str) -> str:
    return hashlib.sha256((prev_hash + canonical_event).encode()).hexdigest()
