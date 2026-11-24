// Gate 16 load test (KPI-31..35): POST /api/v1/agent/query with the 250 golden questions, each under its tenant.
//
//   k6 run load/k6_query.js                                   # 10 VUs for 60 s against localhost:8000
//   VUS=200 DURATION=10m BASE=http://host:8000 k6 run load/k6_query.js
//
// The summary is written to eval/runs/k6_summary.json, which eval/kpis.py reads. KPI-31..35 are defined at 50K VUs:
// kpis.py judges them only when the run reached that, and reports smaller runs as NOT_MEASURED with their numbers.
// Latency here is the final byte of the JSON endpoint. Streamed first-token latency is server-side: GET /metrics.
import http from "k6/http";
import { check } from "k6";
import { Rate } from "k6/metrics";
import { SharedArray } from "k6/data";
import { uuidv4 } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

const BASE = __ENV.BASE || "http://localhost:8000";
const queries = new SharedArray("queries", () => JSON.parse(open("./queries.json")));
const server5xx = new Rate("server_5xx");

export const options = {
  scenarios: {
    load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: __ENV.RAMP || "15s", target: Number(__ENV.VUS || 10) },
        { duration: __ENV.DURATION || "60s", target: Number(__ENV.VUS || 10) },
        { duration: "10s", target: 0 },
      ],
      gracefulRampDown: "30s",
    },
  },
  // blueprint targets (design.md §1.5); k6 exits non-zero when one is missed
  thresholds: {
    "http_req_duration{expected_response:true}": ["p(50)<240", "p(95)<450", "p(99)<590"],
    server_5xx: ["rate<0.0005"],
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

export default function () {
  const q = queries[Math.floor(Math.random() * queries.length)];
  const res = http.post(`${BASE}/api/v1/agent/query`,
    JSON.stringify({ tenant_id: q.t, user_id: "user_k6", session_id: uuidv4(), query: q.q }),
    { headers: { "Content-Type": "application/json", "X-Tenant-ID": q.t }, timeout: "30s" });
  server5xx.add(res.status >= 500 || res.status === 0);  // 0: connection refused / timed out
  check(res, { "200": (r) => r.status === 200, "has answer": (r) => r.status === 200 && !!r.json("answer") });
}

export function handleSummary(data) {
  const out = {
    vus_max: data.metrics.vus_max.values.max,
    duration_s: data.state.testRunDurationMs / 1000,
    rps: data.metrics.http_reqs.values.rate,
    requests: data.metrics.http_reqs.values.count,
    latency_ms: data.metrics["http_req_duration{expected_response:true}"]?.values ?? data.metrics.http_req_duration.values,
    server_5xx_rate: data.metrics.server_5xx.values.rate,
    checks_pass_rate: data.metrics.checks.values.rate,
  };
  return {
    "eval/runs/k6_summary.json": JSON.stringify(out, null, 2) + "\n",
    stdout: JSON.stringify(out, null, 2) + "\n",
  };
}
