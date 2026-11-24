"""Index-time injection review queue for knowledge-vault chunks (ADR-010).

`build` scans every chunk (regex shield + classifier). Flagged chunks are QUARANTINED and never served until a
human APPROVES them; clean ones are CLEAN. Verdicts are keyed by chunk_hash, so editing a chunk's text yields a new
hash that is scanned again, and a rebuild keeps earlier human decisions. A hash the queue has never seen (tool output,
or an index built without a queue) is scanned live and fails closed.

  uv run guardrails/review.py scan                         # (re)scan the built index; `rag_server.py build` runs it too
  uv run guardrails/review.py list                         # quarantined chunks waiting for review
  uv run guardrails/review.py approve <hash> <reviewer> "<reason>"
  uv run guardrails/review.py reject  <hash> <reviewer> "<reason>"
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardrails import injection  # noqa: E402
from guardrails.prehook import is_poisoned  # noqa: E402

# tracked in git next to the vault: the decisions are an audit trail, the index directory is a disposable build output
FILE = Path(os.getenv("RAG_REVIEW", ROOT / "knowledge_vault" / "chunk_review.json"))
CHUNKS = Path(os.getenv("RAG_INDEX", ROOT / "indexes" / "rag")) / "chunks.jsonl"
SERVE = {"CLEAN", "APPROVED"}
HUMAN = {"APPROVED", "REJECTED"}


class ReviewQueue:
    def __init__(self, path: Path = FILE) -> None:
        self.path, self._mtime, self._rows = path, None, {}

    def rows(self) -> dict[str, dict]:
        """Reloaded when the file changes, so a decision takes effect without restarting the server."""
        try:
            m = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._mtime, self._rows = None, {}
            return self._rows
        if m != self._mtime:
            self._rows, self._mtime = json.loads(self.path.read_text()), m
        return self._rows

    def allows(self, chunk_hash: str, text: str) -> bool:
        row = self.rows().get(chunk_hash)
        return row["status"] in SERVE if row else not is_poisoned(text)

    def scan(self, chunks: list[dict]) -> dict[str, int]:
        """Index-time pass over the new index's chunks. Keeps human decisions; drops rows for chunks no longer indexed."""
        from guardrails import classifier

        old, rows = self.rows(), {}
        for c in chunks:
            h = c["chunk_hash"]
            if h in old and old[h]["status"] in HUMAN:
                rows[h] = old[h]
                continue
            flagged = is_poisoned(c["text"])
            rows[h] = {"status": "QUARANTINED" if flagged else "CLEAN", "note_id": c["note_id"],
                       "rules": list(injection.scan(c["text"]).hits) if flagged else [],
                       "score": round(classifier.injection_prob(injection._fold(c["text"])), 4) if flagged else None}
        self._write(rows)
        return {s: sum(r["status"] == s for r in rows.values()) for s in ("CLEAN", "QUARANTINED", *sorted(HUMAN))}

    def decide(self, chunk_hash: str, status: str, reviewer: str, reason: str) -> dict:
        if status not in HUMAN:
            raise ValueError(f"status must be one of {sorted(HUMAN)}")
        if not reviewer.strip() or not reason.strip():
            raise ValueError("reviewer and reason are required")
        rows = dict(self.rows())
        if chunk_hash not in rows:
            raise KeyError(f"unknown chunk {chunk_hash}")
        rows[chunk_hash] = {**rows[chunk_hash], "status": status, "reviewer": reviewer, "reason": reason,
                            "decided_at": datetime.now(timezone.utc).isoformat()}
        self._write(rows)
        return rows[chunk_hash]

    def pending(self) -> dict[str, dict]:
        return {h: r for h, r in self.rows().items() if r["status"] == "QUARANTINED"}

    def _write(self, rows: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.path)  # atomic: a crash mid-write never leaves a half-written queue


def main(argv: list[str]) -> int:
    q = ReviewQueue()
    if argv[:1] == ["scan"]:
        print(q.scan([json.loads(line) for line in CHUNKS.read_text().splitlines()]))
        return 0
    if argv[:1] == ["list"]:
        texts = {}
        if CHUNKS.exists():
            texts = {c["chunk_hash"]: c["text"] for c in map(json.loads, CHUNKS.read_text().splitlines())}
        for h, r in q.pending().items():
            print(f"{h}  {r['note_id']}  score={r['score']}  rules={r['rules']}\n    {texts.get(h, '')[:300]!r}\n")
        print(f"{len(q.pending())} chunk(s) quarantined")
        return 0
    if len(argv) == 4 and argv[0] in ("approve", "reject"):
        print(q.decide(argv[1], "APPROVED" if argv[0] == "approve" else "REJECTED", argv[2], argv[3]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
