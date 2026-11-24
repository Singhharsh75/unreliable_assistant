"""Lifecycle plugin registry (blueprint Gate 0): mcp_config.json says which plugins run, in what order and with which
options; code registers the hooks. To add a PRE_EXECUTION plugin, decorate a function with `hook` and add a config line.

The config can extend the guardrails but never weaken them (ADR-012): `load` refuses a file that drops, disables or
opens a mandatory plugin, lets anything run before the PII redactor, or loosens the circuit breaker.

shortcut: only PRE_EXECUTION runs through hooks; the other stages read their options at startup (api/main.py). Add hooks
to a stage when a second plugin needs it.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG = Path(__file__).resolve().parent.parent / "mcp_config.json"
Stage = Literal["PRE_EXECUTION", "TOOL_DISPATCH", "POST_EXECUTION", "ALL_SPANS"]
FAIL_CLOSED = ("pii_redactor", "injection_shield")
# name -> (hook, HarnessError code raised when the hook itself crashes)
HOOKS: dict[str, tuple[Callable[[Any], None], int]] = {}


class Plugin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    stage: Stage
    priority: int = 100
    enabled: bool = True
    fail_closed: bool = False
    max_identical_calls: int = Field(default=2, ge=1, le=2)   # circuit_breaker
    min_faithfulness: float = Field(default=0.90, ge=0, le=1)  # ragas_auditor
    endpoint: str | None = None                                # otel_exporter


class PluginConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mcpServers: dict[str, dict[str, Any]]
    plugins: list[Plugin]

    @model_validator(mode="after")
    def _guardrails_intact(self) -> "PluginConfig":
        names = [p.name for p in self.plugins]
        if len(set(names)) != len(names):
            raise ValueError("duplicate plugin name")
        for name, stage in [(n, "PRE_EXECUTION") for n in FAIL_CLOSED] + [("circuit_breaker", "TOOL_DISPATCH")]:
            p = self.get(name)
            if p is None or not p.enabled or p.stage != stage or (name in FAIL_CLOSED and not p.fail_closed):
                raise ValueError(f"{name} is mandatory, enabled, in {stage}" + (" and fail_closed" * (name in FAIL_CLOSED)))
        if self.stage("PRE_EXECUTION")[0].name != "pii_redactor":
            raise ValueError("pii_redactor must run first: nothing else may see unredacted text")
        return self

    def get(self, name: str) -> Plugin | None:
        return next((p for p in self.plugins if p.name == name), None)

    def stage(self, stage: Stage) -> list[Plugin]:
        # the name is the tie-break, so order in the file never matters
        return sorted((p for p in self.plugins if p.stage == stage and p.enabled), key=lambda p: (p.priority, p.name))


@lru_cache
def load(path: Path = CONFIG) -> PluginConfig:
    return PluginConfig.model_validate(json.loads(path.read_text()))


def hook(name: str, error_code: int) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Registers the PRE_EXECUTION hook for plugin `name`. It edits the request state in place or raises HarnessError."""
    def register(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        HOOKS[name] = (fn, error_code)
        return fn
    return register
