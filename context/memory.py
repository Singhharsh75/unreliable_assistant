"""Sliding window (hot) plus persisted verified facts (durable). design.md Gate 5 §3.

Hot tier: last N turns per (tenant_id, session_id), held in RAM. Stand-in for the Redis hot window.
Durable tier: `session_facts` in SQLite locally (Postgres in dev and prod), keyed by tenant and session.
"""
from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime

from contracts.models import SessionFact
from context.anchor import Turn

WINDOW_SIZE = 3


class SqliteFactStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS session_facts (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   tenant_id TEXT NOT NULL,
                   session_id TEXT NOT NULL,
                   key TEXT NOT NULL,
                   value TEXT NOT NULL,
                   source_chunk_hash TEXT,
                   created_at TEXT NOT NULL
               )"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_facts_scope ON session_facts (tenant_id, session_id)")
        self._db.commit()

    def put(self, tenant_id: str, session_id: str, fact: SessionFact) -> int:
        cur = self._db.execute(
            "INSERT INTO session_facts (tenant_id, session_id, key, value, source_chunk_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (tenant_id, session_id, fact.key, fact.value, fact.source_chunk_hash, datetime.now(UTC).isoformat()),
        )
        self._db.commit()
        return cur.lastrowid

    def latest(self, tenant_id: str, session_id: str) -> dict[str, SessionFact]:
        """Latest value per key within one tenant and session. Other tenants' rows are never read."""
        rows = self._db.execute(
            """SELECT key, value, source_chunk_hash FROM session_facts
               WHERE tenant_id = ? AND session_id = ? ORDER BY id""",
            (tenant_id, session_id),
        ).fetchall()
        out: dict[str, SessionFact] = {}
        for key, value, chunk in rows:
            out[key] = SessionFact(key=key, value=value, source_chunk_hash=chunk)
        return out


class SessionMemory:
    def __init__(self, store: SqliteFactStore | None = None, window_size: int = WINDOW_SIZE) -> None:
        self.store = store or SqliteFactStore()
        self.window_size = window_size
        self._hot: dict[tuple[str, str], deque[Turn]] = {}

    def append_turn(self, tenant_id: str, session_id: str, turn: Turn) -> None:
        win = self._hot.setdefault((tenant_id, session_id), deque(maxlen=self.window_size))
        win.append(turn)  # maxlen drops the oldest turn, so the window never exceeds window_size

    def window(self, tenant_id: str, session_id: str) -> tuple[Turn, ...]:
        return tuple(self._hot.get((tenant_id, session_id), ()))

    def persist_verified(self, tenant_id: str, session_id: str, facts: Iterable[SessionFact]) -> None:
        """Only call this for facts that passed verification."""
        for fact in facts:
            self.store.put(tenant_id, session_id, fact)

    def facts(self, tenant_id: str, session_id: str) -> dict[str, SessionFact]:
        return self.store.latest(tenant_id, session_id)
