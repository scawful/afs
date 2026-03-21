"""Token-budgeted context packs for Gemini, Claude, Codex, and generic clients."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_index import ContextSQLiteIndex
from .context_paths import resolve_agent_output_root, resolve_mount_root
from .embeddings import search_embedding_index
from .manager import AFSManager
from .models import MountType
from .sensitivity import matches_path_rules
from .session_bootstrap import build_session_bootstrap

DEFAULT_CONTEXT_PACK_TOKENS = {
    "generic": 8000,
    "gemini": 16000,
    "claude": 12000,
    "codex": 12000,
}
DEFAULT_SEARCH_MOUNTS = (
    MountType.SCRATCHPAD,
    MountType.MEMORY,
    MountType.KNOWLEDGE,
    MountType.ITEMS,
)


@dataclass
class ContextPackSection:
    title: str
    body: str
    priority: int
    sources: list[str] = field(default_factory=list)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.body) + max(4, estimate_tokens(self.title))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "priority": self.priority,
            "sources": list(self.sources),
            "estimated_tokens": self.estimated_tokens(),
        }


def estimate_tokens(text: str) -> int:
    """Approximate token count cheaply for context budgeting."""
    if not text or not text.strip():
        return 0
    return max(1, (len(text) + 3) // 4)


def build_context_pack(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str = "",
    model: str = "generic",
    token_budget: int | None = None,
    include_content: bool = False,
    max_query_results: int = 6,
    max_embedding_results: int = 4,
) -> dict[str, Any]:
    """Build a model-aware context pack from AFS state."""
    context_path = context_path.expanduser().resolve()
    normalized_model = _normalize_model(model)
    resolved_budget = token_budget or DEFAULT_CONTEXT_PACK_TOKENS[normalized_model]
    bootstrap = build_session_bootstrap(manager, context_path)
    guidance = _model_guidance(normalized_model)
    sections = _build_sections(
        manager,
        context_path,
        bootstrap=bootstrap,
        query=query,
        model=normalized_model,
        include_content=include_content,
        max_query_results=max_query_results,
        max_embedding_results=max_embedding_results,
    )
    chosen, omitted = _select_sections(sections, token_budget=resolved_budget)
    sources = sorted({source for section in chosen for source in section.sources})
    estimated = sum(section.estimated_tokens() for section in chosen)

    return {
        "context_path": str(context_path),
        "project": bootstrap["project"],
        "profile": bootstrap["profile"],
        "model": normalized_model,
        "query": query,
        "token_budget": resolved_budget,
        "estimated_tokens": estimated,
        "guidance": guidance,
        "sections": [section.to_dict() for section in chosen],
        "sources": sources,
        "omitted_sections": [section.title for section in omitted],
    }


def render_context_pack(pack: dict[str, Any]) -> str:
    """Render a context pack for direct model consumption."""
    lines = [
        f"# AFS Context Pack: {pack['project']}",
        f"Context: {pack['context_path']}",
        f"Profile: {pack['profile']}",
        f"Target model: {pack['model']}",
        f"Token budget: {pack['estimated_tokens']}/{pack['token_budget']}",
    ]
    if pack.get("query"):
        lines.append(f"Query: {pack['query']}")
    guidance = pack.get("guidance")
    if isinstance(guidance, str) and guidance.strip():
        lines.extend(["", "## Guidance", guidance.strip()])

    for section in pack.get("sections", []):
        lines.extend(["", f"## {section['title']}", section["body"]])
        if section.get("sources"):
            lines.append("Sources:")
            for source in section["sources"]:
                lines.append(f"- {source}")

    omitted = pack.get("omitted_sections") or []
    if omitted:
        lines.extend(["", "## Omitted", ", ".join(str(item) for item in omitted)])
    return "\n".join(lines)


def write_context_pack_artifacts(
    manager: AFSManager,
    context_path: Path,
    pack: dict[str, Any],
) -> dict[str, str]:
    """Persist the latest context pack for wrappers and agent handoff."""
    output_root = resolve_agent_output_root(context_path, config=manager.config)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = pack["model"].replace("/", "_")
    json_path = output_root / f"session_pack_{suffix}.json"
    markdown_path = output_root / f"session_pack_{suffix}.md"

    payload = dict(pack)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_context_pack(payload) + "\n", encoding="utf-8")
    return payload["artifact_paths"]


def _normalize_model(model: str) -> str:
    normalized = (model or "generic").strip().lower()
    if normalized not in DEFAULT_CONTEXT_PACK_TOKENS:
        return "generic"
    return normalized


def _model_guidance(model: str) -> str:
    guidance = {
        "generic": "Prefer the highest-signal sections first. Cite file paths when acting on retrieved context.",
        "gemini": "Gemini should start with the recommendations and source-backed sections, then ask for more retrieval only if the pack leaves gaps.",
        "claude": "Claude should translate this pack into a short execution plan before editing, keeping cited paths attached to claims.",
        "codex": "Codex should prioritize actionable files, recent drift, and concrete next steps before writing code.",
    }
    return guidance[model]


def _build_sections(
    manager: AFSManager,
    context_path: Path,
    *,
    bootstrap: dict[str, Any],
    query: str,
    model: str,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> list[ContextPackSection]:
    sections: list[ContextPackSection] = []
    sections.append(
        ContextPackSection(
            title="Recommended Actions",
            body=_render_list(bootstrap.get("recommended_actions", []), fallback="No urgent actions."),
            priority=0,
        )
    )
    sections.append(
        ContextPackSection(
            title="Context Health",
            body=_render_status_block(bootstrap["status"], bootstrap["diff"]),
            priority=5,
        )
    )
    scratchpad = bootstrap.get("scratchpad", {})
    sections.append(
        ContextPackSection(
            title="Scratchpad State",
            body=_render_scratchpad_block(scratchpad),
            priority=10,
            sources=[scratchpad["path"]] if scratchpad.get("path") else [],
        )
    )

    handoff = bootstrap.get("handoff", {})
    if handoff.get("available"):
        sections.append(
            ContextPackSection(
                title="Latest Handoff",
                body=_render_handoff_block(handoff),
                priority=15,
            )
        )

    tasks = bootstrap.get("tasks", {})
    sections.append(
        ContextPackSection(
            title="Open Tasks",
            body=_render_tasks_block(tasks),
            priority=20,
        )
    )
    hivemind = bootstrap.get("hivemind", {})
    sections.append(
        ContextPackSection(
            title="Recent Hivemind",
            body=_render_hivemind_block(hivemind),
            priority=25,
        )
    )
    memory = bootstrap.get("memory", {})
    sections.append(
        ContextPackSection(
            title="Durable Memory",
            body=_render_memory_block(memory),
            priority=30,
            sources=[memory["latest_markdown_path"]] if memory.get("latest_markdown_path") else [],
        )
    )

    if query.strip():
        query_sections = _query_sections(
            manager,
            context_path,
            query=query,
            include_content=include_content,
            max_results=max_query_results,
        )
        sections.extend(query_sections)
        embedding_section = _embedding_section(
            manager,
            context_path,
            query=query,
            max_results=max_embedding_results,
        )
        if embedding_section is not None:
            sections.append(embedding_section)

    sections.append(
        ContextPackSection(
            title="Model Usage Notes",
            body=_model_guidance(model),
            priority=90,
        )
    )
    return [section for section in sections if section.body.strip()]


def _query_sections(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str,
    include_content: bool,
    max_results: int,
) -> list[ContextPackSection]:
    settings = manager.config.context_index
    mount_types = list(DEFAULT_SEARCH_MOUNTS)
    index = ContextSQLiteIndex(manager, context_path)
    if settings.enabled:
        if not index.has_entries(mount_types=mount_types) and settings.auto_index:
            index.rebuild(
                mount_types=mount_types,
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            )
        elif settings.auto_refresh and index.has_entries(mount_types=mount_types) and index.needs_refresh(
            mount_types=mount_types
        ):
            index.rebuild(
                mount_types=mount_types,
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            )

    if not settings.enabled or not index.has_entries(mount_types=mount_types):
        return []

    entries = index.query(
        query=query,
        mount_types=mount_types,
        limit=max(1, max_results),
        include_content=include_content,
    )
    sections: list[ContextPackSection] = []
    for offset, entry in enumerate(entries, start=1):
        if _entry_blocked(entry, manager.config.sensitivity.never_export):
            continue
        source = str(entry.get("absolute_path", "")).strip()
        excerpt = ""
        if include_content and isinstance(entry.get("content"), str):
            excerpt = _trim_text(entry["content"], limit=1200)
        elif isinstance(entry.get("content_excerpt"), str):
            excerpt = _trim_text(entry["content_excerpt"], limit=500)
        body_lines = [
            f"Path: {entry['mount_type']}/{entry['relative_path']}",
            f"Modified: {entry.get('modified_at') or 'unknown'}",
        ]
        if excerpt:
            body_lines.extend(["", excerpt])
        sections.append(
            ContextPackSection(
                title=f"Indexed Hit {offset}",
                body="\n".join(body_lines),
                priority=35 + offset,
                sources=[source] if source else [],
            )
        )
    return sections


def _embedding_section(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str,
    max_results: int,
) -> ContextPackSection | None:
    knowledge_root = resolve_mount_root(context_path, MountType.KNOWLEDGE, config=manager.config)
    candidates: list[Path] = []
    root_index = knowledge_root / "embedding_index.json"
    if root_index.exists():
        candidates.append(knowledge_root)
    if knowledge_root.exists():
        for child in sorted(knowledge_root.iterdir()):
            if child.is_dir() and (child / "embedding_index.json").exists():
                candidates.append(child)

    hits: list[tuple[float, str, str]] = []
    for index_root in candidates:
        try:
            results = search_embedding_index(
                index_root,
                query,
                top_k=max(1, max_results),
                min_score=0.15,
            )
        except (FileNotFoundError, ValueError):
            continue
        for result in results:
            source_path = str(result.source_path)
            if _path_blocked(Path(source_path), relative_path="", patterns=manager.config.sensitivity.never_export):
                continue
            hits.append((result.score, source_path, result.text_preview))

    if not hits:
        return None

    hits.sort(key=lambda item: item[0], reverse=True)
    lines: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for score, source_path, preview in hits:
        if source_path in seen:
            continue
        seen.add(source_path)
        lines.append(f"- {source_path} (score={score:.3f})")
        if preview.strip():
            lines.append(f"  {preview.strip()}")
        sources.append(source_path)
        if len(sources) >= max_results:
            break
    return ContextPackSection(
        title="Embedding Hits",
        body="\n".join(lines),
        priority=60,
        sources=sources,
    )


def _select_sections(
    sections: list[ContextPackSection],
    *,
    token_budget: int,
) -> tuple[list[ContextPackSection], list[ContextPackSection]]:
    chosen: list[ContextPackSection] = []
    omitted: list[ContextPackSection] = []
    used = 0
    for section in sorted(sections, key=lambda item: item.priority):
        cost = section.estimated_tokens()
        if chosen and used + cost > token_budget:
            omitted.append(section)
            continue
        chosen.append(section)
        used += cost
    return chosen, omitted


def _render_status_block(status: dict[str, Any], diff: dict[str, Any]) -> str:
    mount_health = status.get("mount_health", {})
    index_info = status.get("index", {})
    lines = [
        f"Mount files: {status.get('total_files', 0)}",
        f"Mount health: {'healthy' if mount_health.get('healthy') else 'needs repair'}",
    ]
    if index_info.get("enabled"):
        if index_info.get("built", index_info.get("has_entries", False)):
            freshness = "stale" if index_info.get("stale") else "fresh"
            lines.append(f"Index: {index_info.get('total_entries', 0)} entries ({freshness})")
        else:
            lines.append("Index: not built")
    else:
        lines.append("Index: disabled")
    if diff.get("available"):
        lines.append(f"Recent drift: {diff.get('total_changes', 0)} changes")
    elif diff.get("error"):
        lines.append(f"Recent drift: unavailable ({diff['error']})")
    return "\n".join(lines)


def _render_scratchpad_block(scratchpad: dict[str, Any]) -> str:
    chunks: list[str] = []
    state_text = str(scratchpad.get("state_text", "")).strip()
    deferred_text = str(scratchpad.get("deferred_text", "")).strip()
    if state_text:
        chunks.append("State:\n" + _trim_text(state_text, limit=1200))
    if deferred_text:
        chunks.append("Deferred:\n" + _trim_text(deferred_text, limit=1200))
    other_files = scratchpad.get("other_files") or []
    if other_files:
        chunks.append("Other files:\n" + "\n".join(f"- {name}" for name in other_files[:8]))
    return "\n\n".join(chunks) if chunks else "Scratchpad is empty."


def _render_handoff_block(handoff: dict[str, Any]) -> str:
    lines = [
        f"Session: {handoff.get('session_id', '')}",
        f"Agent: {handoff.get('agent_name', '')}",
        f"Timestamp: {handoff.get('timestamp', '')}",
    ]
    for label in ("accomplished", "blocked", "next_steps"):
        values = handoff.get(label) or []
        if values:
            lines.append(f"{label.replace('_', ' ').title()}:")
            lines.extend(f"- {value}" for value in values[:8])
    return "\n".join(lines)


def _render_tasks_block(tasks: dict[str, Any]) -> str:
    total = int(tasks.get("total", 0) or 0)
    if total <= 0:
        return "No queued tasks."
    counts = tasks.get("counts") or {}
    lines = [f"Total queued tasks: {total}"]
    if counts:
        lines.append(
            "Counts: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        )
    for item in (tasks.get("items") or [])[:8]:
        lines.append(
            f"- [{item.get('status', 'pending')}] p{item.get('priority', '?')} {item.get('title', '')}"
        )
    return "\n".join(lines)


def _render_hivemind_block(hivemind: dict[str, Any]) -> str:
    total = int(hivemind.get("recent_count", 0) or 0)
    if total <= 0:
        return "No recent hivemind messages."
    lines = [f"Recent messages: {total}"]
    for message in (hivemind.get("messages") or [])[:8]:
        target = f" -> {message.get('to')}" if message.get("to") else ""
        topic = f" #{message.get('topic')}" if message.get("topic") else ""
        lines.append(
            f"- {message.get('timestamp', '')[:19]} [{message.get('type', '')}] {message.get('from', '')}{target}{topic}"
        )
    return "\n".join(lines)


def _render_memory_block(memory: dict[str, Any]) -> str:
    lines = [f"Entries: {memory.get('entries_count', 0)}"]
    latest = str(memory.get("latest_markdown_path", "")).strip()
    if latest:
        lines.append(f"Latest summary: {latest}")
    excerpt = str(memory.get("latest_markdown_excerpt", "")).strip()
    if excerpt:
        lines.extend(["", _trim_text(excerpt, limit=1200)])
    return "\n".join(lines)


def _render_list(items: list[str], *, fallback: str) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return fallback
    return "\n".join(f"- {item}" for item in cleaned[:10])


def _trim_text(text: str, *, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _entry_blocked(entry: dict[str, Any], patterns: list[str]) -> bool:
    absolute_path = Path(str(entry.get("absolute_path", "")))
    relative_path = f"{entry.get('mount_type', '')}/{entry.get('relative_path', '')}".strip("/")
    return _path_blocked(absolute_path, relative_path=relative_path, patterns=patterns)


def _path_blocked(absolute_path: Path, *, relative_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return matches_path_rules(
        absolute_path,
        relative_path=relative_path,
        patterns=patterns,
    )
