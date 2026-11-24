"""Signed MCP server manifests (blueprint Step 13): each server's tools, tiers and input-schema hashes, signed with
Ed25519 by the release owner. The API refuses to start when a registered tool differs from its signed manifest, so
a changed contract or a swapped tool needs a deliberate re-sign.

  uv run harness/manifest.py keygen     # var/mcp_signing_key.pem (secret, git-ignored) + harness/manifests/signing_key.pub
  uv run harness/manifest.py sign       # writes harness/manifests/<server>.json for the current tool contracts
  uv run harness/manifest.py verify
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryptography.exceptions import InvalidSignature  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey  # noqa: E402

from harness.tool_harness import ToolSpec  # noqa: E402

DIR = ROOT / "harness" / "manifests"
PUBLIC_KEY = DIR / "signing_key.pub"
PRIVATE_KEY = Path(os.getenv("MCP_SIGNING_KEY", ROOT / "var" / "mcp_signing_key.pem"))


class ManifestError(RuntimeError):
    pass


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def manifest(server: str, specs: list[ToolSpec]) -> dict:
    return {"server": server, "tools": sorted(({
        "name": s.name, "tier": s.tier, "min_caller_tier": s.min_caller_tier or s.tier, "hitl_gate": s.hitl_gate,
        "input_schema_sha256": hashlib.sha256(_canonical(s.input_model.model_json_schema())).hexdigest(),
    } for s in specs), key=lambda t: t["name"])}


def sign(m: dict, key: Ed25519PrivateKey) -> dict:
    return {"manifest": m, "signature": base64.b64encode(key.sign(_canonical(m))).decode()}


def verify(server: str, specs: list[ToolSpec], directory: Path = DIR) -> None:
    """Raises ManifestError unless `specs` match the server's manifest and the signature checks out."""
    try:
        signed = json.loads((directory / f"{server}.json").read_text())
        pub = serialization.load_pem_public_key((directory / PUBLIC_KEY.name).read_bytes())
    except (OSError, ValueError) as e:
        raise ManifestError(f"{server}: no readable signed manifest ({e})") from e
    if not isinstance(pub, Ed25519PublicKey):
        raise ManifestError("signing key is not Ed25519")
    try:
        pub.verify(base64.b64decode(signed["signature"]), _canonical(signed["manifest"]))
    except (InvalidSignature, KeyError, ValueError) as e:
        raise ManifestError(f"{server}: manifest signature is invalid") from e
    if signed["manifest"] != manifest(server, specs):
        raise ManifestError(f"{server}: registered tools differ from the signed manifest; re-sign after review")


def server_specs() -> dict[str, list[ToolSpec]]:
    """The tool contracts each server exposes, with placeholder handlers (a manifest covers contracts, not code)."""
    from contracts.models import GraphSubgraphIn, HybridRetrieveIn
    from mcp_servers.state_store import StateStore

    def stub(m):
        return None
    return {
        "mcp-hybrid-rag": [ToolSpec("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ", stub, pool="rag")],
        "mcp-graphify-vault": [ToolSpec("graph_subgraph", GraphSubgraphIn, "TIER_1_READ", stub)],
        "mcp-state-store": StateStore().specs(),
    }


def main(argv: list[str]) -> int:
    if argv == ["keygen"]:
        if PRIVATE_KEY.exists():
            sys.exit(f"{PRIVATE_KEY} exists; refusing to overwrite a signing key")
        key = Ed25519PrivateKey.generate()
        PRIVATE_KEY.parent.mkdir(parents=True, exist_ok=True)
        PRIVATE_KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
        PRIVATE_KEY.chmod(0o600)
        DIR.mkdir(parents=True, exist_ok=True)
        PUBLIC_KEY.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                             serialization.PublicFormat.SubjectPublicKeyInfo))
        print(f"private key {PRIVATE_KEY} (keep secret), public key {PUBLIC_KEY}")
        return 0
    if argv == ["sign"]:
        key = serialization.load_pem_private_key(PRIVATE_KEY.read_bytes(), password=None)
        for server, specs in server_specs().items():
            (DIR / f"{server}.json").write_text(json.dumps(sign(manifest(server, specs), key), indent=1) + "\n")
            print(f"signed {server}: {len(specs)} tools")
        return 0
    if argv == ["verify"]:
        for server, specs in server_specs().items():
            verify(server, specs)
            print(f"{server}: ok")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
