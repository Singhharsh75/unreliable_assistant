"""ONNX NLI cross-encoder for the Verification Subagent (design.md §1.4 row 12: verify.nli <= 40 ms, KPI-24)."""
from __future__ import annotations

import os
from functools import cache

import numpy as np

from mcp_servers.rag_server import BATCH, MAX_LEN, _run, _session, _tokenizer

NLI_REPO = os.getenv("NLI_REPO", "cross-encoder/nli-deberta-v3-small")
NLI_FILE = os.getenv("NLI_FILE", "onnx/model_quint8_avx2.onnx")
CONTRA_INDEX, ENTAIL_INDEX = 0, 1  # cross-encoder/nli-*: 0 contradiction, 1 entailment, 2 neutral


@cache
def _model():
    return _session(NLI_REPO, NLI_FILE), _tokenizer(NLI_REPO, MAX_LEN)


def entailment(pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """(P(entailment), P(contradiction)) for each (premise, hypothesis) pair."""
    if not pairs:
        return []
    sess, tok = _model()
    logits = np.concatenate([_run(sess, tok, pairs[i:i + BATCH]) for i in range(0, len(pairs), BATCH)])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return list(zip(p[:, ENTAIL_INDEX].tolist(), p[:, CONTRA_INDEX].tolist()))
