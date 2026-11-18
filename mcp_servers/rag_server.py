"""Gate 3: mcp-hybrid-rag — BM25 ∥ bi-encoder+FAISS → RRF → cross-encoder (design.md §1.2.2, ADR-001).

Usage:
  uv run mcp_servers/rag_server.py build            # chunk + embed knowledge_vault/ into indexes/rag/
  uv run mcp_servers/rag_server.py query "..."      # one retrieval, printed as JSON
  uv run mcp_servers/rag_server.py bench [N]        # per-stage p95 latency (KPI-07/08/09)
  uv run mcp_servers/rag_server.py                  # serve MCP over stdio
"""
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
import yaml
from huggingface_hub import hf_hub_download
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

from contracts.models import Chunk, HybridRetrieveIn, HybridRetrieveOut

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path(os.getenv("RAG_VAULT", ROOT / "knowledge_vault"))
INDEX = Path(os.getenv("RAG_INDEX", ROOT / "indexes" / "rag"))
BI_REPO, BI_FILE = "BAAI/bge-large-en-v1.5", "onnx/model.onnx"
CE_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
CE_FILE = os.getenv("RAG_RERANK_FILE", "onnx/model_fp16.onnx")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-en-v1.5 query instruction
MAX_LEN, OVERLAP = 512, 64
BM25_K1, BM25_B, EF_SEARCH, HNSW_M, RRF_POOL, BATCH = 1.25, 0.75, 64, 32, 40, 32

_stem = PorterStemmer().stem
POOL = ThreadPoolExecutor(max_workers=2)  # BM25 and dense run in parallel


def bm25_tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())]


def rrf(rankings: list[list[int]], k: int, pool: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over lists of chunk indices (best first); index = deduped chunk_hash."""
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, i in enumerate(ranked, 1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:pool]


def windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    step = size - overlap
    return [(s, min(s + size, n_tokens)) for s in range(0, max(n_tokens - overlap, 1), step)]


def _tokenizer(repo: str, max_len: int | None) -> Tokenizer:
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    if max_len:
        pad = next(p for p in ("[PAD]", "<pad>") if tok.token_to_id(p) is not None)
        tok.enable_truncation(max_len)
        tok.enable_padding(pad_id=tok.token_to_id(pad), pad_token=pad)
    return tok


def _session(repo: str, file: str) -> ort.InferenceSession:
    return ort.InferenceSession(hf_hub_download(repo, file), providers=["CPUExecutionProvider"])


def _run(sess: ort.InferenceSession, tok: Tokenizer, texts) -> np.ndarray:
    encs = tok.encode_batch(texts)
    feeds = {
        "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
    }
    names = {i.name for i in sess.get_inputs()}
    return sess.run(None, {k: v for k, v in feeds.items() if k in names})[0].astype(np.float32)


@cache
def bi_encoder():
    return _session(BI_REPO, BI_FILE), _tokenizer(BI_REPO, MAX_LEN)


@cache
def cross_encoder():
    return _session(CE_REPO, CE_FILE), _tokenizer(CE_REPO, MAX_LEN)


def embed(texts: list[str]) -> np.ndarray:
    sess, tok = bi_encoder()
    out = np.concatenate([_run(sess, tok, texts[i:i + BATCH])[:, 0] for i in range(0, len(texts), BATCH)])  # CLS pooling
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def rerank(query: str, texts: list[str]) -> np.ndarray:
    sess, tok = cross_encoder()
    logits = np.concatenate([_run(sess, tok, [(query, t) for t in texts[i:i + BATCH]]).reshape(-1)
                             for i in range(0, len(texts), BATCH)])
    return 1.0 / (1.0 + np.exp(-logits))


def load_notes(vault: Path) -> list[tuple[dict, str]]:
    notes = []
    for path in sorted(vault.glob("**/*.md")):
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(), re.S)
        if m and (meta := yaml.safe_load(m.group(1)) or {}).get("id"):  # notes without frontmatter id are skipped
            notes.append((meta, m.group(2).strip()))
    return notes


def chunk_notes(notes: list[tuple[dict, str]]) -> list[dict]:
    tok = _tokenizer(BI_REPO, None)
    chunks, seen = [], set()
    for meta, body in notes:
        enc = tok.encode(body, add_special_tokens=False)
        if not enc.ids:
            continue
        for s, e in windows(len(enc.ids), MAX_LEN - 2, OVERLAP):  # -2 leaves room for [CLS]/[SEP]
            text = body[enc.offsets[s][0]:enc.offsets[e - 1][1]]
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            chunks.append({"chunk_hash": h, "text": text, "note_id": str(meta["id"]),
                           "domain": meta.get("domain"), "tenants": meta.get("tenants"),
                           "tokens": bm25_tokens(text)})
    return chunks


def build(vault: Path = VAULT, out: Path = INDEX) -> int:
    chunks = chunk_notes(load_notes(vault))
    if not chunks:
        sys.exit(f"no notes with frontmatter id under {vault}")
    vecs = embed([c["text"] for c in chunks])
    index = faiss.IndexHNSWFlat(vecs.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "dense.faiss"))
    (out / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))
    return len(chunks)


class Retriever:
    def __init__(self, index_dir: Path = INDEX):
        self.chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text().splitlines()]
        self.bm25 = BM25Okapi([c["tokens"] for c in self.chunks], k1=BM25_K1, b=BM25_B)
        self.dense = faiss.read_index(str(index_dir / "dense.faiss"))
        self.dense.hnsw.efSearch = EF_SEARCH
        bi_encoder(), cross_encoder()  # load models before the first request

    def _allowed(self, i: int, tenant: str, domain: str | None) -> bool:
        c = self.chunks[i]
        return (c["tenants"] is None or tenant in c["tenants"]) and (domain is None or c["domain"] == domain)

    # shortcut: tenant/domain filter runs after top-k, so a tenant's private chunks can be crowded out
    # of the top 50 by shared ones; move to per-tenant indices when tenants have large private corpora.
    def _bm25(self, query: str, k: int, ok) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0 and ok(i)]

    def _dense(self, query: str, k: int, ok) -> list[tuple[int, float]]:
        sims, ids = self.dense.search(embed([QUERY_PREFIX + query]), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0 and ok(i)]

    def retrieve(self, req: HybridRetrieveIn) -> HybridRetrieveOut:
        ok = lambda i: self._allowed(i, req.ctx.tenant_id, req.domain)  # noqa: E731
        lat = {}

        def timed(name, fn, *args):
            t0 = time.perf_counter()
            res = fn(*args)
            lat[name] = (time.perf_counter() - t0) * 1000
            return res

        f_sparse = POOL.submit(timed, "bm25", self._bm25, req.query, req.bm25_top_k, ok)
        f_dense = POOL.submit(timed, "dense", self._dense, req.query, req.dense_top_k, ok)
        sparse, dense = dict(f_sparse.result()), dict(f_dense.result())

        fused = timed("rrf", rrf, [list(sparse), list(dense)], req.rrf_k, RRF_POOL)
        ids = [i for i, _ in fused]
        scores = timed("rerank", rerank, req.query, [self.chunks[i]["text"] for i in ids]) if ids else np.array([])
        lat.setdefault("rerank", 0.0)

        ranked = sorted(zip(ids, scores, range(1, len(ids) + 1)), key=lambda t: -t[1])
        top_score = float(ranked[0][1]) if ranked else 0.0
        kept = [t for t in ranked[:req.rerank_k] if t[1] >= req.min_rerank_score]
        return HybridRetrieveOut(
            chunks=[Chunk(chunk_hash=self.chunks[i]["chunk_hash"], uri=f"rag://chunks/{self.chunks[i]['chunk_hash']}",
                          text=self.chunks[i]["text"], note_id=self.chunks[i]["note_id"],
                          bm25_score=sparse.get(i), dense_cosine=dense.get(i), rrf_rank=r, rerank_score=float(s))
                    for i, s, r in kept],
            top_score=top_score,
            abstain=not kept,
            stage_latency_ms={k: round(v, 2) for k, v in lat.items()},
        )


def serve():
    from mcp.server.mcpserver import MCPServer

    retriever = Retriever()
    by_hash = {c["chunk_hash"]: c for c in retriever.chunks}
    mcp = MCPServer("mcp-hybrid-rag")

    @mcp.tool()
    def hybrid_retrieve(req: dict) -> dict:
        """3-stage hybrid retrieval. `req` must match contracts.models.HybridRetrieveIn."""
        # validate as JSON so strict mode still accepts UUID strings from the wire
        return retriever.retrieve(HybridRetrieveIn.model_validate_json(json.dumps(req))).model_dump()

    @mcp.resource("rag://chunks/{chunk_hash}")
    def chunk(chunk_hash: str) -> str:
        c = by_hash.get(chunk_hash)
        if c is None or c["tenants"] is not None:  # no tenant context on resource reads: shared chunks only
            raise ValueError("TENANT_SCOPE_VIOLATION or unknown chunk")
        return json.dumps({k: c[k] for k in ("chunk_hash", "text", "note_id", "domain")})

    mcp.run("stdio")


def _demo_req(query: str, tenant: str = "tenant_demo") -> HybridRetrieveIn:
    import uuid
    return HybridRetrieveIn.model_validate_json(json.dumps({
        "ctx": {"tenant_id": tenant, "user_id": "user_cli", "session_id": str(uuid.uuid4()),
                "idempotency_key": str(uuid.uuid4()), "traceparent": f"00-{'0' * 31}1-{'0' * 15}1-01"},
        "query": query}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "build":
        print(f"indexed {build()} chunks → {INDEX}")
    elif cmd == "query":
        print(Retriever().retrieve(_demo_req(sys.argv[2])).model_dump_json(indent=2))
    elif cmd == "bench":
        r, n = Retriever(), int(sys.argv[2]) if len(sys.argv) > 2 else 50
        qs = ["Verify batch LOT-88392 safety protocol", "what temperature for SKU-099",
              "how are critical deviations handled", "who approves a lot release", "capital of France"]
        r.retrieve(_demo_req(qs[0]))  # warm-up
        runs = [r.retrieve(_demo_req(qs[i % len(qs)])).stage_latency_ms for i in range(n)]
        budget = {"bm25": 12, "dense": 25, "rrf": 3, "rerank": 45}
        for stage, b in budget.items():
            p95 = float(np.percentile([x[stage] for x in runs], 95))
            print(f"{stage:7s} p95={p95:8.2f} ms  budget<{b} ms  {'PASS' if p95 < b else 'FAIL'}")
    else:
        serve()
