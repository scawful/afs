"""Builtin MCP tool definitions (schemas + handler wiring).

Handlers live in ``afs.mcp_server``; this module only assembles the catalog so
``mcp_server.py`` stays focused on serve/dispatch and tool implementations.
"""

from __future__ import annotations

from dataclasses import replace

from afs.context_index import DEFAULT_MAX_CONTENT_CHARS, DEFAULT_MAX_FILE_SIZE_BYTES
from afs.models import MountType
from afs.operator_digests import KIND_CHOICES

from .registry import MCPToolDefinition


def _tool_alias_definition(
    tool: MCPToolDefinition,
    *,
    name: str,
    description: str | None = None,
) -> MCPToolDefinition:
    return replace(
        tool,
        name=name,
        description=description or tool.description,
    )


def build_builtin_tool_definitions() -> list[MCPToolDefinition]:
    """Return core MCP tool definitions with handlers bound from mcp_server."""

    from afs import mcp_server as server

    _mcp_scope_properties = server._mcp_scope_properties

    fs_read = MCPToolDefinition(
        name="fs.read",
        description="Deprecated alias of `context.read` (removal in 0.3). Read UTF-8 text from a context-scoped file after bootstrap or `context.status` anchors the active context.",
        input_schema={
            "type": "object",
            "properties": {
                **_mcp_scope_properties(),
                "path": {"type": "string", "description": "Absolute or relative file path."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=server._tool_fs_read,
    )
    fs_write = MCPToolDefinition(
        name="fs.write",
        description="Deprecated alias of `context.write` (removal in 0.3). Write UTF-8 text to a context-scoped file, preferring scratchpad paths for working notes unless the user explicitly wants durable memory or knowledge updates.",
        input_schema={
            "type": "object",
            "properties": {
                **_mcp_scope_properties(),
                "path": {"type": "string"},
                "content": {"type": "string"},
                "if_match": {
                    "type": "string",
                    "description": "Expected raw-byte SHA-256, or missing to create.",
                },
                "append": {"type": "boolean", "default": False},
                "mkdirs": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=server._tool_fs_write,
    )
    fs_delete = MCPToolDefinition(
        name="fs.delete",
        description="Deprecated alias of `context.delete` (removal in 0.3). Delete a file or directory under allowed context roots.",
        input_schema={
            "type": "object",
            "properties": {
                **_mcp_scope_properties(),
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=server._tool_fs_delete,
    )
    fs_move = MCPToolDefinition(
        name="fs.move",
        description="Deprecated alias of `context.move` (removal in 0.3). Move or rename a file or directory under allowed context roots.",
        input_schema={
            "type": "object",
            "properties": {
                **_mcp_scope_properties(),
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "mkdirs": {"type": "boolean", "default": False},
            },
            "required": ["source", "destination"],
            "additionalProperties": False,
        },
        handler=server._tool_fs_move,
    )
    fs_list = MCPToolDefinition(
        name="fs.list",
        description="Deprecated alias of `context.list` (removal in 0.3). List files under a context-scoped path.",
        input_schema={
            "type": "object",
            "properties": {
                **_mcp_scope_properties(),
                "path": {"type": "string"},
                "max_depth": {"type": "integer", "default": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=server._tool_fs_list,
    )

    return [
        MCPToolDefinition(
            name="messages.send",
            description="Send a scoped inter-agent message to the current project or common scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "from": {"type": "string", "description": "Sending agent name."},
                    "type": {"type": "string", "default": "status"},
                    "payload": {"type": "object"},
                    "text": {"type": "string", "description": "Plain-text body (payload.text)."},
                    "to": {
                        "type": "string",
                        "description": "Recipient(s) separated by ',' or '/'; 'all' broadcasts. "
                        "Directed messages do not expire by default.",
                    },
                    "reply_to": {"type": "string", "description": "Message ID being answered."},
                    "topic": {"type": "string"},
                    "ttl_hours": {"type": "integer"},
                },
                "required": ["from"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_send,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="messages.read",
            description=(
                "Read messages visible to the current project plus common messages. "
                "Pass `to` for an inbox view with read/done state; agent, type and "
                "topic filters still apply."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "agent": {"type": "string", "description": "Filter by sender."},
                    "to": {
                        "type": "string",
                        "description": "Inbox for this agent: direct, family, and broadcast messages.",
                    },
                    "include_done": {"type": "boolean", "default": False},
                    "type": {"type": "string"},
                    "topic": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "include_legacy": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="messages.ack",
            description="Mark a message read, or done so it leaves the recipient's inbox.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "message_id": {"type": "string"},
                    "by": {"type": "string", "description": "Acknowledging agent name."},
                    "done": {"type": "boolean", "default": False},
                    "note": {"type": "string"},
                },
                "required": ["message_id", "by"],
                "additionalProperties": False,
            },
            handler=server._tool_messages_ack,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="messages.subscribe",
            description="Subscribe an agent to message topics.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "agent_name": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "ttl_hours": {"type": "integer"},
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_subscribe,
        ),
        MCPToolDefinition(
            name="messages.unsubscribe",
            description="Unsubscribe an agent from message topics.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "agent_name": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_unsubscribe,
        ),
        MCPToolDefinition(
            name="messages.clean",
            description="Preview or apply queue-wide expired-message cleanup; all_projects is required.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "max_age_hours": {"type": "integer"},
                    "apply": {"type": "boolean", "default": False},
                },
                "required": ["all_projects"],
                "additionalProperties": False,
            },
            handler=server._tool_messages_clean,
        ),
        MCPToolDefinition(
            name="note.create",
            description="Create an immutable readable Markdown note in the current project scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "task_id": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "author_kind": {
                        "type": "string",
                        "enum": ["agent", "human", "import", "system"],
                        "default": "agent",
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["internal", "private", "public", "restricted"],
                        "default": "internal",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
            handler=server._tool_note_create,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="note.read",
            description="Read one immutable note from the authorized scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "identifier": {"type": "string"},
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            handler=server._tool_note_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="note.list",
            description="List recent immutable notes from the authorized scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=server._tool_note_list,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="skill.match",
            description=(
                "Rank configured AFS skills against a bounded task description. "
                "Use when the task shifts mid-session; request bodies only when "
                "you need inline instructions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": server.MAX_SKILL_MATCH_PROMPT_CHARS,
                        "description": "Task or intent text to match against skill triggers.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": server.MAX_SKILL_MATCHES,
                        "default": 5,
                    },
                    "include_bodies": {"type": "boolean", "default": False},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            handler=server._tool_skill_match,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="skill.read",
            description=(
                "Read one configured AFS skill by name. Returns metadata and at "
                f"most {server.MAX_SKILL_BODY_CHARS} characters of instructions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                        "description": "Skill name reported by skill.match.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=server._tool_skill_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="briefing",
            description="Morning briefing — calendar, gmail, open tasks, and active agents. Use at the start of a session to understand current state.",
            input_schema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "default": 7,
                        "description": "Lookback window in days.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_briefing,
        ),
        fs_read,
        _tool_alias_definition(
            fs_read,
            name="context.read",
            description="Preferred agent-facing file read. Read UTF-8 text from a context-scoped file.",
        ),
        fs_write,
        _tool_alias_definition(
            fs_write,
            name="context.write",
            description="Preferred agent-facing file write. Write UTF-8 text to a context-scoped file.",
        ),
        fs_delete,
        _tool_alias_definition(
            fs_delete,
            name="context.delete",
            description="Preferred agent-facing file delete. Delete a file or directory under allowed context roots.",
        ),
        fs_move,
        _tool_alias_definition(
            fs_move,
            name="context.move",
            description="Preferred agent-facing file move. Move or rename a file or directory under allowed context roots.",
        ),
        fs_list,
        _tool_alias_definition(
            fs_list,
            name="context.list",
            description="Preferred agent-facing file listing. List files under a context-scoped path.",
        ),
        MCPToolDefinition(
            name="context.discover",
            description="Discover project .context roots.",
            input_schema={
                "type": "object",
                "properties": {
                    "search_paths": {"type": "array", "items": {"type": "string"}},
                    "max_depth": {"type": "integer", "default": 3},
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_discover,
        ),
        MCPToolDefinition(
            name="context.init",
            description="Initialize a context root for a project path.",
            input_schema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string"},
                    "context_root": {"type": "string"},
                    "context_dir": {"type": "string"},
                    "profile": {"type": "string"},
                    "link_context": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_init,
        ),
        MCPToolDefinition(
            name="context.mount",
            description="Mount a source path into a context mount type.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "source": {"type": "string"},
                    "mount_type": {"type": "string", "enum": [mount.value for mount in MountType]},
                    "alias": {"type": "string"},
                },
                "required": ["source", "mount_type"],
                "additionalProperties": False,
            },
            handler=server._tool_context_mount,
        ),
        MCPToolDefinition(
            name="context.unmount",
            description="Unmount a source alias from a context mount type.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "mount_type": {"type": "string", "enum": [mount.value for mount in MountType]},
                    "alias": {"type": "string"},
                },
                "required": ["mount_type", "alias"],
                "additionalProperties": False,
            },
            handler=server._tool_context_unmount,
        ),
        MCPToolDefinition(
            name="context.index.rebuild",
            description="Rebuild the SQLite context index. Use this when bootstrap/status reports the index is missing or stale.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "mount_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": [mount.value for mount in MountType]},
                    },
                    "include_content": {"type": "boolean", "default": True},
                    "max_file_size_bytes": {
                        "type": "integer",
                        "default": DEFAULT_MAX_FILE_SIZE_BYTES,
                    },
                    "max_content_chars": {
                        "type": "integer",
                        "default": DEFAULT_MAX_CONTENT_CHARS,
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_index_rebuild,
        ),
        MCPToolDefinition(
            name="context.query",
            description="Search the SQLite context index by path/content. Use this before asking the user for context that may already be in memory, knowledge, or scratchpad.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "query": {"type": "string"},
                    "relative_prefix": {"type": "string"},
                    "mount_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": [mount.value for mount in MountType]},
                    },
                    "limit": {"type": "integer", "default": 25},
                    "include_content": {"type": "boolean", "default": False},
                    "auto_index": {"type": "boolean", "default": True},
                    "auto_refresh": {"type": "boolean", "default": False},
                    "refresh": {"type": "boolean", "default": False},
                    "max_file_size_bytes": {
                        "type": "integer",
                        "default": DEFAULT_MAX_FILE_SIZE_BYTES,
                    },
                    "max_content_chars": {
                        "type": "integer",
                        "default": DEFAULT_MAX_CONTENT_CHARS,
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_query,
        ),
        MCPToolDefinition(
            name="context.search",
            description=(
                "Search the v2 scoped hybrid index. Defaults to local text retrieval; "
                "semantic=true explicitly permits the index's embedding provider for the query."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "query": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["text", "symbol"],
                        "default": "text",
                    },
                    "semantic": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=server._tool_context_search,
        ),
        MCPToolDefinition(
            name="context.diff",
            description="Show new, modified, and deleted files since the last index build. Call this before editing to understand drift since the previous session.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "mount_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": [mount.value for mount in MountType]},
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_diff,
        ),
        MCPToolDefinition(
            name="context.status",
            description="Summary of context health: mount counts, index stats, active profile. Call this first in a new session if `afs.session.bootstrap` is not available.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_status,
        ),
        MCPToolDefinition(
            name="session.pack",
            description="Build a token-budgeted context pack for Gemini, Claude, Codex, or generic clients. Use this after bootstrap when the agent needs a compact cited working set; repeated calls can reuse the last artifact-backed pack when the bootstrap snapshot and pack inputs have not changed.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "query": {"type": "string", "description": "Optional retrieval query."},
                    "task": {
                        "type": "string",
                        "description": "Explicit task statement to render at the end of the pack.",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["generic", "gemini", "claude", "codex"],
                        "default": "generic",
                    },
                    "workflow": {
                        "type": "string",
                        "enum": [
                            "general",
                            "scan_fast",
                            "edit_fast",
                            "review_deep",
                            "root_cause_deep",
                        ],
                        "default": "general",
                    },
                    "tool_profile": {
                        "type": "string",
                        "enum": [
                            "default",
                            "context_readonly",
                            "context_repair",
                            "edit_and_verify",
                            "handoff_only",
                        ],
                        "default": "default",
                    },
                    "pack_mode": {
                        "type": "string",
                        "enum": ["focused", "retrieval", "full_slice"],
                        "default": "focused",
                    },
                    "token_budget": {"type": "integer", "description": "Approximate token budget."},
                    "include_content": {"type": "boolean", "default": False},
                    "semantic": {
                        "type": "boolean",
                        "default": False,
                        "description": "Explicitly permit remote query embeddings.",
                    },
                    "max_query_results": {"type": "integer", "default": 6},
                    "max_embedding_results": {"type": "integer", "default": 4},
                },
                "additionalProperties": False,
            },
            handler=server._tool_session_pack,
        ),
        MCPToolDefinition(
            name="work.communication.list",
            description="List captured work communication samples before drafting docs, design docs, requirements, or replies in the user's work style.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "person_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=server._tool_work_communication_list,
        ),
        MCPToolDefinition(
            name="work.communication.add",
            description="Capture a work communication sample for future tone/style grounding. Store only deliberate work samples, not arbitrary private text.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "text": {"type": "string"},
                    "person_id": {"type": "string"},
                    "source_system": {"type": "string"},
                    "source_id": {"type": "string"},
                    "channel": {"type": "string"},
                    "purpose": {"type": "string"},
                    "style_notes": {"type": "array", "items": {"type": "string"}},
                    "provenance": {"type": "array", "items": {"type": "object"}},
                    "confidence": {"type": "number", "default": 0.5},
                    "dedupe_key": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=server._tool_work_communication_add,
        ),
        MCPToolDefinition(
            name="work.communication.guide",
            description="Summarize available work communication style evidence and mandatory approval guardrails for work-context writing.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "person_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=server._tool_work_communication_guide,
        ),
        MCPToolDefinition(
            name="work.communication.preflight",
            description=(
                "Run the mandatory work-writing preflight: style evidence, optional opt-in "
                "personal context, pending approvals, and explicit approval guardrails. "
                "This never executes an external write."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "person_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "approval_limit": {"type": "integer", "default": 10},
                    "personal_mode": {"type": "string"},
                    "personal_context_root": {"type": "string"},
                },
                "additionalProperties": False,
            },
            handler=server._tool_work_communication_preflight,
        ),
        MCPToolDefinition(
            name="approvals.list",
            description="List agent approval-gate requests (pending by default) with source, scope, payload, and expiry. Read-only; deciding needs `afs approvals approve|reject` at a terminal.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "source": {"type": "string"},
                    "all": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 50},
                },
                "additionalProperties": False,
            },
            handler=server._tool_approvals_list,
        ),
        MCPToolDefinition(
            name="work.approvals.list",
            description="List local AFS work approval requests. Review this before using connector write tools or claiming permission to post externally.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {"type": "string", "default": "pending"},
                    "all": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 50},
                },
                "additionalProperties": False,
            },
            handler=server._tool_work_approvals_list,
        ),
        MCPToolDefinition(
            name="work.approvals.show",
            description="Show one local AFS work approval request and its preview/result details.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "approval_id": {"type": "string"},
                },
                "required": ["approval_id"],
                "additionalProperties": False,
            },
            handler=server._tool_work_approvals_show,
        ),
        MCPToolDefinition(
            name="work.approvals.request",
            description="Create a local approval request for posting, sending, submitting, or editing an external work system. This asks for permission; it does not execute the write.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "target_system": {"type": "string"},
                    "target_id": {"type": "string"},
                    "action": {"type": "string"},
                    "summary": {"type": "string"},
                    "preview": {"type": "object"},
                    "affected_people": {"type": "array", "items": {}},
                    "risk_level": {"type": "string", "default": "medium"},
                    "permission_required": {"type": "string", "default": "human approval"},
                    "requested_by": {"type": "string", "default": "agent"},
                    "expires_at": {"type": "string"},
                    "dedupe_key": {"type": "string"},
                },
                "required": ["target_system", "target_id", "action", "summary"],
                "additionalProperties": False,
            },
            handler=server._tool_work_approvals_request,
        ),
        MCPToolDefinition(
            name="operator.digest",
            description="Compress noisy command, test, traceback, grep, or diffstat output into a small digest before sending it back into model context.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw command or tool output to summarize.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(KIND_CHOICES),
                        "default": "auto",
                        "description": "Digest strategy hint. Use auto unless the caller already knows the output type.",
                    },
                    "max_items": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum highlighted entries to keep in the digest.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional human label such as `pytest -q` or `git diff --stat`.",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=server._tool_operator_digest,
        ),
        MCPToolDefinition(
            name="context.repair",
            description="Repair mount provenance, broken mounts, and stale indexes for a context.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "profile_name": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "reapply_profile": {"type": "boolean", "default": True},
                    "remap_missing_sources": {"type": "boolean", "default": True},
                    "rebuild_index": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_repair,
        ),
        MCPToolDefinition(
            name="agent.spawn",
            description="Spawn a background agent process.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name."},
                    "module": {"type": "string", "description": "Python module to run."},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra CLI arguments.",
                    },
                },
                "required": ["name", "module"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_spawn,
        ),
        MCPToolDefinition(
            name="agent.ps",
            description="List running background agents.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=server._tool_agent_ps,
        ),
        MCPToolDefinition(
            name="agent.stop",
            description="Stop a running background agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name to stop."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_stop,
        ),
        MCPToolDefinition(
            name="agent.logs",
            description="Read recent progress events for a background agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "name": {"type": "string", "description": "Agent name."},
                    "limit": {
                        "type": "integer",
                        "description": "Max events to return.",
                        "default": 20,
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_logs,
        ),
        MCPToolDefinition(
            name="hivemind.send",
            description="Send a message to the hivemind inter-agent bus.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "from": {"type": "string", "description": "Sender agent name."},
                    "type": {
                        "type": "string",
                        "description": "Message type: finding, request, or status.",
                    },
                    "payload": {"type": "object", "description": "Message payload."},
                    "to": {"type": "string", "description": "Optional recipient agent name."},
                    "topic": {
                        "type": "string",
                        "description": "Optional topic for pub/sub routing (e.g. context:repair, agent:lifecycle).",
                    },
                    "ttl_hours": {
                        "type": "integer",
                        "description": "Optional per-message retention window in hours.",
                    },
                },
                "required": ["from", "type"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_send,
        ),
        MCPToolDefinition(
            name="hivemind.read",
            description="Read messages from the hivemind inter-agent bus.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent": {"type": "string", "description": "Filter by sender agent name."},
                    "type": {"type": "string", "description": "Filter by message type."},
                    "topic": {"type": "string", "description": "Filter by topic."},
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return.",
                        "default": 50,
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_read,
        ),
        MCPToolDefinition(
            name="task.create",
            description="Create a task in the items queue.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "title": {"type": "string", "description": "Task title."},
                    "created_by": {"type": "string", "description": "Creator agent name."},
                    "priority": {
                        "type": "integer",
                        "description": "Priority (1=highest, 10=lowest).",
                        "default": 5,
                    },
                    "context": {
                        "type": "object",
                        "description": "Task context (files, issue, etc).",
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            handler=server._tool_task_create,
        ),
        MCPToolDefinition(
            name="task.list",
            description="List tasks from the items queue.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {
                        "type": "string",
                        "description": "Filter by status: pending, claimed, in_progress, done, failed.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_task_list,
        ),
        MCPToolDefinition(
            name="task.claim",
            description="Claim a pending task for an agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "task_id": {"type": "string", "description": "Task ID to claim."},
                    "agent_name": {"type": "string", "description": "Agent claiming the task."},
                },
                "required": ["task_id", "agent_name"],
                "additionalProperties": False,
            },
            handler=server._tool_task_claim,
        ),
        MCPToolDefinition(
            name="task.complete",
            description="Mark a task as completed.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "task_id": {"type": "string", "description": "Task ID to complete."},
                    "result": {"type": "object", "description": "Optional result data."},
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            handler=server._tool_task_complete,
        ),
        MCPToolDefinition(
            name="agent.manifest.show",
            description="Show or export the repo-owned agent harness manifest.",
            input_schema={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Optional manifest TOML path."},
                    "harness": {
                        "type": "string",
                        "description": "Optional harness name to export.",
                    },
                    "validate": {"type": "boolean", "default": False},
                    "check_paths": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_manifest_show,
        ),
        MCPToolDefinition(
            name="agent.run.start",
            description="Start a replayable shared agent-run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "task": {"type": "string"},
                    "harness": {"type": "string"},
                    "workspace": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_run_start,
        ),
        MCPToolDefinition(
            name="agent.run.list",
            description="List replayable shared agent-run records.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {"type": "string", "description": "Optional status filter."},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_run_list,
        ),
        MCPToolDefinition(
            name="agent.run.show",
            description="Show one replayable agent run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "run_id": {"type": "string"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_run_show,
        ),
        MCPToolDefinition(
            name="agent.run.event",
            description="Append an event to a replayable agent run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "run_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["run_id", "event_type"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_run_event,
        ),
        MCPToolDefinition(
            name="agent.run.finish",
            description="Finish a replayable agent run record with summary, commands, and verification.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "run_id": {"type": "string"},
                    "status": {"type": "string", "default": "done"},
                    "summary": {"type": "string"},
                    "files_changed": {"type": "array", "items": {"type": "string"}},
                    "commands": {"type": "array", "items": {"type": "string"}},
                    "verification": {"type": "array", "items": {"type": "object"}},
                    "handoff_path": {"type": "string"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_run_finish,
        ),
        MCPToolDefinition(
            name="agent.job.create",
            description="Create a markdown background agent job in items/agent_jobs/queue.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "title": {"type": "string"},
                    "prompt": {"type": "string"},
                    "priority": {"type": "integer", "default": 5},
                    "created_by": {"type": "string"},
                    "scope": {"type": "string"},
                    "expected_output": {"type": "string"},
                    "allow_destructive": {"type": "boolean", "default": False},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_create,
        ),
        MCPToolDefinition(
            name="agent.job.status",
            description="Show background job queue, worker, recent run, and watchdog status.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "label": {"type": "string", "default": "com.afs.agent-jobs"},
                    "stale_after_seconds": {"type": "number", "default": 3600},
                    "recent_runs": {"type": "integer", "default": 5},
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_status,
        ),
        MCPToolDefinition(
            name="agent.job.inbox",
            description="Show completed, failed, stale, or blocked background jobs needing human review.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "stale_after_seconds": {"type": "number", "default": 3600},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_inbox,
        ),
        MCPToolDefinition(
            name="agent.job.seed",
            description="Idempotently queue safe report-only background maintenance jobs.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "profile": {"type": "string", "default": "repo-maintenance"},
                    "cadence": {"type": "string", "default": "daily"},
                    "created_by": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_seed,
        ),
        MCPToolDefinition(
            name="agent.job.list",
            description="List markdown background agent jobs.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {
                        "type": "string",
                        "description": "queue, running, done, failed, or archived.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_list,
        ),
        MCPToolDefinition(
            name="agent.job.show",
            description="Show one markdown background agent job.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_show,
        ),
        MCPToolDefinition(
            name="agent.job.review",
            description="Review one background job and its linked run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_review,
        ),
        MCPToolDefinition(
            name="agent.job.archive",
            description="Archive one background job without deleting its markdown record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_archive,
        ),
        MCPToolDefinition(
            name="agent.job.promote",
            description="Promote one background job review into a durable handoff.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                    "to_handoff": {"type": "boolean", "default": True},
                    "handoff_name": {"type": "string"},
                    "archive": {"type": "boolean", "default": False},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_promote,
        ),
        MCPToolDefinition(
            name="agent.job.claim",
            description="Claim a queued markdown background agent job.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                    "agent_name": {"type": "string"},
                },
                "required": ["job_id", "agent_name"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_claim,
        ),
        MCPToolDefinition(
            name="agent.job.move",
            description="Move a markdown background agent job to queue, running, done, failed, or archived.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                    "status": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["job_id", "status"],
                "additionalProperties": False,
            },
            handler=server._tool_agent_job_move,
        ),
        MCPToolDefinition(
            name="review.list",
            description="List agents awaiting review.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=server._tool_review_list,
        ),
        MCPToolDefinition(
            name="review.approve",
            description="Approve an agent's review and transition it back to stopped.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name to approve."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=server._tool_review_approve,
        ),
        MCPToolDefinition(
            name="review.reject",
            description="Reject an agent's review and mark it as failed.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name to reject."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=server._tool_review_reject,
        ),
        MCPToolDefinition(
            name="events.query",
            description="Query the AFS event log with optional type, source, session, since, and limit filters.",
            input_schema={
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (mcp_tool, hivemind, embedding, agent_lifecycle, session).",
                    },
                    "since": {"type": "string", "description": "ISO 8601 datetime cutoff."},
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Max events to return.",
                    },
                    "source": {"type": "string", "description": "Filter by event source."},
                    "session_id": {
                        "type": "string",
                        "description": "Filter by recorded AFS session ID.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_events_query,
        ),
        MCPToolDefinition(
            name="events.tail",
            description="Show the most recent AFS events.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max events to return.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_events_tail,
        ),
        MCPToolDefinition(
            name="events.analytics",
            description="Summarize recent AFS event volume, MCP tool usage, durations, and error rates.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Lookback window in hours.",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Optional single event type filter.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_events_analytics,
        ),
        MCPToolDefinition(
            name="events.replay",
            description="Replay a recorded AFS session timeline by session ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "session_id": {"type": "string", "description": "Recorded AFS session ID."},
                    "limit": {
                        "type": "integer",
                        "default": 200,
                        "description": "Max events to return (0 for all).",
                    },
                    "include_payloads": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include event payloads when available.",
                    },
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
            handler=server._tool_events_replay,
        ),
        MCPToolDefinition(
            name="hivemind.subscribe",
            description="Subscribe an agent to one or more hivemind topics.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent_name": {"type": "string", "description": "Agent name."},
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to subscribe to.",
                    },
                    "ttl_hours": {
                        "type": "integer",
                        "description": "Optional subscription TTL window in hours.",
                    },
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_subscribe,
        ),
        MCPToolDefinition(
            name="hivemind.unsubscribe",
            description="Unsubscribe an agent from one or more hivemind topics.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent_name": {"type": "string", "description": "Agent name."},
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to unsubscribe from.",
                    },
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_unsubscribe,
        ),
        MCPToolDefinition(
            name="hivemind.reap",
            description=(
                "Remove expired or stale hivemind messages and return cleanup statistics; "
                "v2 queues require explicit all_projects=true."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "max_age_hours": {
                        "type": "integer",
                        "description": "Override retention window in hours.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Report removals without deleting files.",
                    },
                },
                "required": ["all_projects"],
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_reap,
        ),
        MCPToolDefinition(
            name="embeddings.index",
            description="Build an embedding index for source paths. Supports incremental mode to skip unchanged files.",
            input_schema={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source paths to index.",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for the index.",
                    },
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to include.",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to exclude.",
                    },
                    "incremental": {
                        "type": "boolean",
                        "default": False,
                        "description": "Skip unchanged files using size+mtime comparison.",
                    },
                },
                "required": ["sources", "output_dir"],
                "additionalProperties": False,
            },
            handler=server._tool_embeddings_index,
        ),
        MCPToolDefinition(
            name="afs.search",
            description=(
                "Legacy unscoped retrieval; prefer `context.query` (or scoped `context.search`). "
                "Unavailable for context v2 namespaces. Unified search across FTS, embeddings, "
                "and AST symbol indexes when those sources exist."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "context_path": {
                        "type": "string",
                        "description": "Context path (uses default if omitted).",
                    },
                    "mount_types": {
                        "type": "string",
                        "description": "Comma-separated mount types for the FTS leg (e.g. 'scratchpad,knowledge').",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Embedding provider (ollama, openai, gemini, hf). Omit to skip embedding leg.",
                    },
                    "model": {"type": "string", "description": "Embedding model name."},
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max results to return.",
                    },
                    "include_symbols": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include AST symbol index results.",
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include embedding index results.",
                    },
                    "include_fts": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include FTS context index results.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=server._tool_afs_search,
        ),
        MCPToolDefinition(
            name="afs.codebase.symbols",
            description=(
                "Legacy unscoped AST lookup; prefer scoped `context.query` / `context.search`. "
                "Look up function, class, and import definitions from the AST codebase index. "
                "Faster and more precise than grep for finding where something is defined."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Symbol name or substring to search.",
                    },
                    "context_path": {
                        "type": "string",
                        "description": "Context path (uses default if omitted).",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["functions", "classes", "imports", "exports"],
                        "description": "Filter by symbol kind.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language (python, typescript, javascript, rust, go).",
                    },
                    "limit": {"type": "integer", "default": 20},
                    "exact": {
                        "type": "boolean",
                        "default": False,
                        "description": "Require exact (case-insensitive) name match.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=server._tool_afs_codebase_symbols,
        ),
        MCPToolDefinition(
            name="afs.codebase.index",
            description=(
                "Legacy unscoped AST index builder; prefer scoped context tools on v2. "
                "Build or update the AST symbol index for a project. "
                "Run this once after checkout or when files change. "
                "Results are used by afs.codebase.symbols and the symbol leg of afs.search."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {
                        "type": "string",
                        "description": "Context path (uses default if omitted).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Project root to index (inferred from context if omitted).",
                    },
                    "max_files": {
                        "type": "integer",
                        "default": 5000,
                        "description": "Max files to index.",
                    },
                    "incremental": {
                        "type": "boolean",
                        "default": True,
                        "description": "Skip unchanged files.",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Limit to specific languages (python, typescript, javascript, rust, go). All supported languages if omitted.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_afs_codebase_index,
        ),
        MCPToolDefinition(
            name="handoff.create",
            description="Create an immutable, readable handoff revision in the current scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "title": {"type": "string", "description": "Readable handoff title."},
                    "agent_name": {"type": "string", "description": "Agent creating the handoff."},
                    "accomplished": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What got done.",
                    },
                    "blocked": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What's stuck.",
                    },
                    "next_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommended actions.",
                    },
                    "context_snapshot": {
                        "type": "object",
                        "description": "Scratchpad state, open files, etc.",
                    },
                    "open_tasks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Open task dicts.",
                    },
                    "metadata": {"type": "object", "description": "Freeform metadata."},
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID (auto-generated if omitted).",
                    },
                    "target_agent": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "critical"],
                        "default": "normal",
                    },
                },
                "required": ["agent_name"],
                "additionalProperties": False,
            },
            handler=server._tool_handoff_create,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="handoff.revise",
            description="Create a new immutable revision that supersedes an earlier handoff revision.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "revision_id": {"type": "string"},
                    "title": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "accomplished": {"type": "array", "items": {"type": "string"}},
                    "blocked": {"type": "array", "items": {"type": "string"}},
                    "next_steps": {"type": "array", "items": {"type": "string"}},
                    "context_snapshot": {"type": "object"},
                    "open_tasks": {"type": "array", "items": {"type": "object"}},
                    "metadata": {"type": "object"},
                    "target_agent": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "critical"],
                        "default": "normal",
                    },
                },
                "required": ["revision_id", "title", "agent_name"],
                "additionalProperties": False,
            },
            handler=server._tool_handoff_revise,
        ),
        MCPToolDefinition(
            name="handoff.read",
            description="Read a handoff packet. Returns the latest if no session_id is given.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to read (latest if omitted).",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_handoff_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="handoff.list",
            description="List recent handoff packets.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max packets to return.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_handoff_list,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="handoff.threads",
            description="List readable handoff threads and their latest revision state.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=server._tool_handoff_threads,
        ),
        MCPToolDefinition(
            name="handoff.ack",
            description="Acknowledge a handoff revision without modifying its immutable content.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "revision_id": {"type": "string"},
                    "by": {"type": "string"},
                },
                "required": ["revision_id", "by"],
                "additionalProperties": False,
            },
            handler=server._tool_handoff_ack,
        ),
        MCPToolDefinition(
            name="handoff.close",
            description="Close a handoff thread via separate lifecycle state.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "identifier": {"type": "string"},
                    "by": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["identifier", "by"],
                "additionalProperties": False,
            },
            handler=server._tool_handoff_close,
        ),
        MCPToolDefinition(
            name="hivemind.cleanup",
            description="Clean up old hivemind messages. Returns per-agent removal stats.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "max_age_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Max message age in hours.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview without deleting.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_hivemind_cleanup,
        ),
        MCPToolDefinition(
            name="memory.status",
            description="Show memory pipeline health: entry count, cursor position, staleness.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                },
                "additionalProperties": False,
            },
            handler=server._tool_memory_status,
        ),
        MCPToolDefinition(
            name="memory.search",
            description="Search durable memory entries by keyword.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {"type": "integer", "default": 10, "description": "Max results."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=server._tool_memory_search,
        ),
        MCPToolDefinition(
            name="agent.capabilities",
            description="List agent capabilities: declared tools, topics, and mount types.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Filter by agent name."},
                },
                "additionalProperties": False,
            },
            handler=server._tool_agent_capabilities,
        ),
        MCPToolDefinition(
            name="context.freshness",
            description="Compute per-file freshness scores showing which context files are stale.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "mount_type": {"type": "string", "description": "Filter by mount type."},
                    "decay_hours": {
                        "type": "number",
                        "default": 168.0,
                        "description": "Decay window in hours.",
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Min score threshold.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_context_freshness,
        ),
        MCPToolDefinition(
            name="training.antigravity.status",
            description="Show Antigravity capture database status for editor integrations and export readiness checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Optional Antigravity state.vscdb override.",
                    },
                    "state_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional Antigravity state key override list.",
                    },
                },
                "additionalProperties": False,
            },
            handler=server._tool_training_antigravity_status,
        ),
        MCPToolDefinition(
            name="session.replay",
            description="Replay a session timeline showing chronological events.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "session_id": {"type": "string", "description": "Session ID (date) to filter."},
                    "since": {
                        "type": "string",
                        "description": "Filter events after this datetime.",
                    },
                    "limit": {"type": "integer", "default": 100, "description": "Max events."},
                },
                "additionalProperties": False,
            },
            handler=server._tool_session_replay,
        ),
    ]
