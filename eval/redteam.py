"""Gate 14 red-team benchmark runner (KPI-01/02/03).

  uv run eval/redteam.py            # prints per-split recall/block rate and every miss
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardrails import injection, pii, run_prehooks  # noqa: E402
from harness.errors import HarnessError  # noqa: E402

DIR = ROOT / "eval" / "redteam"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]


def _clean(s: str) -> str:
    return pii.normalize(s).lower()


def pii_run(split: str | None = None) -> dict:
    rows = [r for r in load("pii") if split in (None, r["split"])]
    misses = []
    for r in rows:
        try:
            out = _clean(run_prehooks(r["text"]).text)
        except HarnessError:  # blocking the request also keeps the PII out
            continue
        if any(_clean(m) in out for m in r["must_not_contain"]):
            misses.append({"id": r["id"], "entity": r["entity"], "text": r["text"], "out": out})
    return {"n": len(rows), "recall": 1 - len(misses) / len(rows), "misses": misses}


def injection_run(split: str | None = None, kind: str | None = None) -> dict:
    rows = [r for r in load("injection") if split in (None, r["split"]) and kind in (None, r["kind"])]
    misses = [r for r in rows if not injection.scan(r["text"]).blocked]
    return {"n": len(rows), "block_rate": 1 - len(misses) / len(rows), "misses": misses}


def false_positives(split: str | None = None) -> dict:
    rows = [r for r in load("benign") if split in (None, r["split"])]
    bad = []
    for r in rows:
        if r["kind"] == "direct":
            try:
                red = run_prehooks(r["text"])
                changed = red.text != pii.normalize(r["text"])
            except HarnessError as e:
                bad.append({"id": r["id"], "text": r["text"][:120], "why": e.name})
                continue
            if changed:
                bad.append({"id": r["id"], "text": r["text"][:120], "why": "redacted"})
        elif injection.scan(r["text"]).blocked:
            bad.append({"id": r["id"], "text": r["text"][:120], "why": "chunk"})
    return {"n": len(rows), "false_positive_rate": len(bad) / len(rows), "cases": bad}


def latency_p95() -> float:
    texts = [r["text"] for r in load("benign") if r["kind"] == "direct"] + [r["text"] for r in load("pii")]
    times = []
    for t in texts * 2:
        s = time.perf_counter()
        try:
            run_prehooks(t)
        except HarnessError:
            pass
        times.append((time.perf_counter() - s) * 1000)
    return statistics.quantiles(times, n=100)[94]


def main() -> int:
    for split in ("dev", "heldout"):
        p, i, f = pii_run(split), injection_run(split), false_positives(split)
        print(f"[{split}] PII recall {p['recall']:.4f} (n={p['n']}) | injection block {i['block_rate']:.4f} "
              f"(n={i['n']}) | benign false positives {f['false_positive_rate']:.4f} (n={f['n']})")
        for m in p["misses"][:15]:
            print("  PII miss", m["entity"], repr(m["text"]), "->", repr(m["out"]))
        for m in i["misses"][:25]:
            print("  INJ miss", m["kind"], m["transform"], repr(m["text"][:110]))
        for c in f["cases"][:15]:
            print("  FP", c)
    print(f"hook.pre p95 {latency_p95():.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
