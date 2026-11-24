"""Gate 16 soak (KPI-36): k6 at constant load for --minutes while sampling the API server's RSS.

  uv run load/soak.py --pid $(lsof -ti tcp:8000 -sTCP:LISTEN) --minutes 60 --vus 4

RSS growth compares the median of the last 5 samples with the median of the first 5 after --warmup minutes (model
sessions and caches fill up first). Medians, because macOS compresses idle pages and single RSS samples swing. Writes eval/runs/soak.json and a /metrics snapshot to
eval/runs/api_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"


def rss_kb(pid: int) -> int:
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True).stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="uvicorn server process")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--vus", type=int, default=4)
    ap.add_argument("--every", type=float, default=30, help="seconds between RSS samples")
    ap.add_argument("--base", default="http://localhost:8000")
    a = ap.parse_args(argv)

    env = {**os.environ, "VUS": str(a.vus), "RAMP": "5s", "DURATION": f"{int(a.minutes * 60)}s", "BASE": a.base}
    # macOS idle sleep pauses everything and k6 then reports the sleep as one 10-minute request; caffeinate -i
    # holds the machine awake for as long as k6 runs
    cmd = (["caffeinate", "-i"] if sys.platform == "darwin" else []) + ["k6", "run", "--quiet", "load/k6_query.js"]
    k6 = subprocess.Popen(cmd, cwd=ROOT, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, samples, alive = time.monotonic(), [], True
    try:
        while k6.poll() is None:
            samples.append({"t_s": round(time.monotonic() - t0, 1), "rss_kb": rss_kb(a.pid)})
            print(json.dumps(samples[-1]), flush=True)
            time.sleep(a.every)
    except subprocess.CalledProcessError:  # the server died: that is a soak result too
        k6.terminate()
        alive = False
        print("server process is gone", flush=True)
    after = [s for s in samples if s["t_s"] >= a.warmup * 60]
    if len(after) >= 10:
        first, last = (statistics.median(s["rss_kb"] for s in w) for w in (after[:5], after[-5:]))
        growth = (last - first) / first
    else:
        growth = None
    elapsed = time.monotonic() - t0
    out = {"minutes": round(elapsed / 60, 1), "vus": a.vus, "warmup_min": a.warmup, "rss_growth": growth,
           "server_alive": alive,
           "k6_exit": k6.returncode, "samples": samples}
    RUNS.mkdir(exist_ok=True)
    (RUNS / "soak.json").write_text(json.dumps(out, indent=1) + "\n")
    try:
        (RUNS / "api_metrics.json").write_text(urllib.request.urlopen(f"{a.base}/metrics", timeout=10).read().decode())
    except OSError as e:
        print("metrics snapshot failed:", e, flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
