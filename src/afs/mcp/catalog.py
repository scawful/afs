"""Frozen MCP tool catalog for agent-facing defaults.

The slim catalog is the product surface for modern agents. Growing it requires
an explicit CHANGELOG note and an update to ``FROZEN_SLIM_MCP_TOOLS``.
"""

from __future__ import annotations

# Context tools always listed even when catalog markers are missing on aliases.
DEFAULT_MCP_TOOL_CATALOG = frozenset(
    {
        "context.status",
        "context.query",
        "context.read",
        "context.list",
        "context.write",
    }
)

# Exact Oracle-hot / coordinator slim surface. CI asserts tools/list matches this.
FROZEN_SLIM_MCP_TOOLS = frozenset(
    {
        *DEFAULT_MCP_TOOL_CATALOG,
        "messages.send",
        "messages.read",
        "messages.ack",
        "note.create",
        "note.read",
        "note.list",
        "skill.match",
        "skill.read",
        "handoff.create",
        "handoff.read",
        "handoff.list",
    }
)

FS_DEPRECATION_SUFFIX = (
    " Deprecated alias of the matching `context.*` tool; scheduled for removal in 0.3."
)

LEGACY_SEARCH_DEPRECATION = (
    " Legacy unscoped retrieval; prefer `context.query` (or scoped `context.search`). "
    "Unavailable for context v2 namespaces."
)
