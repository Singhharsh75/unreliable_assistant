"""Prompt-injection classifier for retrieved chunks and tool output (KPI-03 indirect path).

ONNX DeBERTa-v3 fine-tuned for injection detection. Scored over 3-sentence windows: a whole chunk dilutes one
injected line, and single sentences are miscalibrated ("id: sop-returns" scores 0.997). YAML frontmatter is dropped.
Kept off the user-query pre-hook: ~20 ms per window would break KPI-01 (<= 15 ms).

shortcut: verdicts are memoized per process by text; move the scan to index time (per chunk_hash) when ingestion
lands, so query time is a lookup.
"""
from __future__ import annotations

import os
import re
from functools import cache, lru_cache

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from mcp_servers.rag_server import BATCH, _run, _session

REPO = os.getenv("INJECTION_CLF_REPO", "protectai/deberta-v3-base-prompt-injection-v2")
THRESHOLD = float(os.getenv("INJECTION_CLF_THRESHOLD", "0.9"))
INJECTION_INDEX = 1  # id2label {0: SAFE, 1: INJECTION}
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
WINDOW = 3


@cache
def _model():
    tok = Tokenizer.from_file(hf_hub_download(REPO, "onnx/tokenizer.json"))
    tok.enable_truncation(512)
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    return _session(REPO, "onnx/model.onnx"), tok


@lru_cache(maxsize=4096)
def injection_prob(text: str) -> float:
    """Max P(injection) over 3-sentence windows of `text`."""
    s = [x for x in _SENT.split(_FRONTMATTER.sub("", text)) if x.strip()]
    if not s:
        return 0.0
    sents = [" ".join(s[i:i + WINDOW]) for i in range(max(1, len(s) - WINDOW + 1))]
    sess, tok = _model()
    logits = np.concatenate([_run(sess, tok, sents[i:i + BATCH]) for i in range(0, len(sents), BATCH)])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return float((e[:, INJECTION_INDEX] / e.sum(axis=1)).max())
