"""Gate 5: context retention and drift limits (GoalAnchorBlock, observation compression, sliding window + dual memory)."""
from context.anchor import GoalAnchorBlock, ReActPrompt, Turn, build_anchor, normalize_entities
from context.compress import compress_observation, estimate_tokens
from context.memory import SessionMemory, SqliteFactStore

__all__ = [
    "GoalAnchorBlock",
    "ReActPrompt",
    "SessionMemory",
    "SqliteFactStore",
    "Turn",
    "build_anchor",
    "compress_observation",
    "estimate_tokens",
    "normalize_entities",
]
