"""JSON-RPC error codes from design.md §1.2.5, plus the loop halt states from ADR-004."""
from typing import Any

SCHEMA_VALIDATION_FAILED = -32001
AUTO_REPAIR_FAILED = -32002
RBAC_DENIED = -32003
IDEMPOTENCY_REPLAY = -32004
TENANT_SCOPE_VIOLATION = -32005
HITL_REQUIRED = -32006
CIRCUIT_OPEN = -32007
UPSTREAM_TIMEOUT = -32008
LOW_CONFIDENCE_ABSTAIN = -32009
PII_BLOCKED = -32010
INJECTION_BLOCKED = -32011

# Standard JSON-RPC codes used for malformed envelopes, bad params and unexpected tool crashes.
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

CODE_NAMES = {
    SCHEMA_VALIDATION_FAILED: "SCHEMA_VALIDATION_FAILED",
    AUTO_REPAIR_FAILED: "AUTO_REPAIR_FAILED",
    RBAC_DENIED: "RBAC_DENIED",
    IDEMPOTENCY_REPLAY: "IDEMPOTENCY_REPLAY",
    TENANT_SCOPE_VIOLATION: "TENANT_SCOPE_VIOLATION",
    HITL_REQUIRED: "HITL_REQUIRED",
    CIRCUIT_OPEN: "CIRCUIT_OPEN",
    UPSTREAM_TIMEOUT: "UPSTREAM_TIMEOUT",
    LOW_CONFIDENCE_ABSTAIN: "LOW_CONFIDENCE_ABSTAIN",
    PII_BLOCKED: "PII_BLOCKED",
    INJECTION_BLOCKED: "INJECTION_BLOCKED",
    INVALID_PARAMS: "INVALID_PARAMS",
    INTERNAL_ERROR: "INTERNAL_ERROR",
}


class HarnessError(Exception):
    """A harness rejection. `code` is a JSON-RPC error code from the table above."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None):
        super().__init__(f"{CODE_NAMES.get(code, code)}: {message}")
        self.code = code
        self.name = CODE_NAMES.get(code, str(code))
        self.message = message
        self.data = data or {}

    def to_rpc(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.name, "data": {"detail": self.message, **self.data}}
