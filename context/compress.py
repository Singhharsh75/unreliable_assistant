"""Extractive observation compression (design.md Gate 5 §2).

Only values already present in the tool result are kept. Boilerplate keys are dropped, empty values are
pruned, and the result is truncated to the token budget (<= 180 tokens by default).
"""
from __future__ import annotations

import json
import math
from typing import Any

MAX_OBSERVATION_TOKENS = 180
# Conservative: JSON is token-dense, so 3 characters per token overestimates. The real tokenizer may differ.
CHARS_PER_TOKEN = 3
TRUNCATION_MARK = " …[truncated]"

BOILERPLATE_KEYS = frozenset(
    {
        "traceparent",
        "tracestate",
        "trace_id",
        "span_id",
        "etag",
        "_links",
        "_meta",
        "meta",
        "__typename",
        "request_id",
        "stage_latency_ms",
        "raw_token_estimate",
    }
)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _prune(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in BOILERPLATE_KEYS:
                continue
            pv = _prune(v)
            if pv is None or pv == "" or pv == [] or pv == {}:
                continue
            out[k] = pv
        return out
    if isinstance(obj, list):
        items = [_prune(v) for v in obj]
        return [v for v in items if v is not None and v != "" and v != [] and v != {}]
    return obj


def compress_observation(result: Any, max_tokens: int = MAX_OBSERVATION_TOKENS) -> str:
    text = json.dumps(_prune(result), separators=(",", ":"), ensure_ascii=False, default=str)
    if estimate_tokens(text) <= max_tokens:
        return text
    # Cut so that the text plus the marker stays within the budget.
    keep = max_tokens * CHARS_PER_TOKEN - len(TRUNCATION_MARK)
    out = text[:keep] + TRUNCATION_MARK
    assert estimate_tokens(out) <= max_tokens
    return out
