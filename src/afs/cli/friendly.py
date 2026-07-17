"""Plain-language entry points over stable AFS APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..context_layout import LAYOUT_VERSION, detect_layout_version, resolve_system_path
from ..messages import MessageBus
from ..models import ContextCategory
from ..project_registry import COMMON_SCOPE_ID, ProjectRecord, ProjectRegistry
from ._utils import load_manager, resolve_context_paths


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project or working path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def _manager_context(args: argparse.Namespace):
    config = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config)
    project, context, _override, _directory = resolve_context_paths(args, manager)
    return manager, project, context


def _scope_for(
    context: Path,
    project: Path,
) -> tuple[str, ProjectRecord | None]:
    if detect_layout_version(context) != LAYOUT_VERSION:
        return COMMON_SCOPE_ID, None
    record = ProjectRegistry(context).resolve(project)
    if record is None:
        raise PermissionError(f"project is not registered in central context: {project}")
    return record.scope_id, record


def start_command(args: argparse.Namespace) -> int:
    """Build the normal session-start packet without exposing nested jargon."""
    from .core import session_bootstrap_command

    return session_bootstrap_command(args)


def _search_sources(
    context: Path,
    project: Path,
    *,
    semantic: bool,
    all_projects: bool,
) -> tuple[list[Any], str, str]:
    from ..hybrid_search import HybridSource

    version = detect_layout_version(context)
    sources: list[HybridSource] = []
    current_scope = COMMON_SCOPE_ID
    current_project_id = ""
    if version == LAYOUT_VERSION:
        registry = ProjectRegistry(context)
        current = registry.resolve(project)
        if current is None:
            raise PermissionError(f"project is not registered in central context: {project}")
        current_scope = current.scope_id
        current_project_id = current.project_id
        for record in registry.all_records():
            active = record.project_id == current.project_id
            sources.append(
                HybridSource(
                    Path(record.path),
                    scope_id=record.scope_id,
                    project_id=record.project_id,
                    project_terms=(record.name,),
                    active=active,
                    embed_allowed=semantic and (active or all_projects),
                )
            )
            for category in ContextCategory:
                scoped_root = context / category.value / "projects" / record.project_id
                if scoped_root.is_dir():
                    sources.append(
                        HybridSource(
                            scoped_root,
                            scope_id=record.scope_id,
                            project_id=record.project_id,
                            project_terms=(record.name, category.value),
                            active=active,
                            embed_allowed=semantic and (active or all_projects),
                        )
                    )
        for category in ContextCategory:
            common_root = context / category.value / "common"
            if common_root.is_dir():
                sources.append(
                    HybridSource(
                        common_root,
                        scope_id=COMMON_SCOPE_ID,
                        project_terms=("common", category.value),
                        embed_allowed=semantic,
                    )
                )
    else:
        sources.append(
            HybridSource(
                project,
                scope_id=COMMON_SCOPE_ID,
                project_terms=(project.name,),
                embed_allowed=semantic,
            )
        )
        for category in ContextCategory:
            category_root = context / category.value
            if category_root.is_dir():
                sources.append(
                    HybridSource(
                        category_root,
                        scope_id=COMMON_SCOPE_ID,
                        project_terms=(category.value,),
                        embed_allowed=semantic,
                    )
                )
    return sources, current_scope, current_project_id


def search_command(args: argparse.Namespace) -> int:
    """Run local-first scoped retrieval with explicit semantic opt-in."""
    from ..embeddings import DEFAULT_GEMINI_MODEL, create_embed_fn
    from ..hybrid_search import HybridSearchEngine

    _manager, project, context = _manager_context(args)
    sources, scope_id, project_id = _search_sources(
        context,
        project,
        semantic=bool(args.semantic),
        all_projects=bool(args.all_projects),
    )
    index_root = resolve_system_path(context, "search")
    engine = HybridSearchEngine(index_root)
    has_index = engine.current_path.is_file() or engine.metadata_path.is_file()
    build_payload: dict[str, Any] | None = None
    embed_fn = None
    requested_model = args.model or (
        DEFAULT_GEMINI_MODEL if args.provider == "gemini" else "nomic-embed-text"
    )
    semantic_index_ready = False
    if args.semantic and has_index:
        try:
            metadata = json.loads(engine.metadata_path.read_text(encoding="utf-8"))
            collection = metadata.get("collection", {})
            stats = metadata.get("stats", {})
            semantic_index_ready = (
                collection.get("health") == "healthy"
                and collection.get("provider") == args.provider
                and collection.get("model") == requested_model
                and int(stats.get("vectors", 0) or 0) > 0
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            semantic_index_ready = False
    if args.semantic:
        embed_fn = create_embed_fn(
            args.provider,
            model=requested_model,
        )
    if args.rebuild or not has_index or (args.semantic and not semantic_index_ready):
        build = engine.build(sources, embed_fn=embed_fn)
        build_payload = {
            "total_files": build.total_files,
            "indexed_files": build.indexed_files,
            "chunks_written": build.chunks_written,
            "skipped": build.skipped,
            "vector_count": build.vector_count,
            "vector_dimension": build.vector_dimension,
            "semantic_status": build.semantic_status,
            "capped_sources": build.capped_sources,
            "denied": build.denied,
            "errors": build.errors,
        }
    response = engine.search(
        args.query,
        scope_ids=[scope_id] if scope_id != COMMON_SCOPE_ID else [],
        include_common=True,
        all_projects=bool(args.all_projects),
        mode="hybrid" if args.semantic else args.mode,
        top_k=args.limit,
        recreate_query_embedder=bool(args.semantic),
    )
    payload = response.to_dict()
    payload.update(
        {
            "context_root": str(context),
            "project_path": str(project),
            "project_id": project_id,
            "index_root": str(index_root),
            "rebuilt": build_payload is not None,
            "build": build_payload,
        }
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    mode = payload["mode"]
    print(
        f"search: {len(response.results)} result(s)  mode={mode}  "
        f"semantic={response.semantic_status}"
    )
    for hit in response.results:
        location = f"{hit.source_path}:{hit.line_start}"
        print(f"{hit.score:.6f}  [{hit.scope_id}]  {location}")
        if hit.text_preview:
            print(f"  {hit.text_preview.replace(chr(10), ' ')[:240]}")
    return 0


def projects_current_command(args: argparse.Namespace) -> int:
    manager, project, context = _manager_context(args)
    scope_id, record = _scope_for(context, project)
    payload: dict[str, Any] = {
        "context_root": str(context),
        "layout_version": detect_layout_version(context),
        "project_path": str(project),
        "registered": record is not None,
        "scope_id": scope_id,
        "project": record.to_dict() if record else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"context_root: {payload['context_root']}")
        print(f"layout_version: {payload['layout_version']}")
        print(f"project_path: {payload['project_path']}")
        print(f"scope_id: {payload['scope_id']}")
        print(f"registered: {str(payload['registered']).lower()}")
    return 0


def _registry_from_args(args: argparse.Namespace) -> ProjectRegistry:
    config = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config)
    root = (
        Path(args.context_root).expanduser().resolve()
        if getattr(args, "context_root", None)
        else manager.config.general.context_root.expanduser().resolve()
    )
    return ProjectRegistry(root)


def projects_list_command(args: argparse.Namespace) -> int:
    registry = _registry_from_args(args)
    records = registry.all_records()
    payload = [record.to_dict() for record in records]
    if args.json:
        print(json.dumps(payload, indent=2))
    elif not records:
        print("no registered projects")
    else:
        for record in records:
            print(f"{record.project_id}\t{record.name}\t{record.path}")
    return 0


def projects_register_command(args: argparse.Namespace) -> int:
    registry = _registry_from_args(args)
    project = Path(args.project_path).expanduser().resolve()
    record = registry.register(project, name=args.name)
    if args.json:
        print(json.dumps(record.to_dict(), indent=2))
    else:
        print(f"registered: {record.project_id}")
        print(f"scope_id: {record.scope_id}")
        print(f"path: {record.path}")
    return 0


def projects_import_command(args: argparse.Namespace) -> int:
    from ..workspace_sync import load_workspace_entries

    registry = _registry_from_args(args)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    entries = load_workspace_entries(
        workspace_root,
        include_sections=False,
        include_items=True,
        include_local=not args.no_local,
    )
    existing_paths = {
        str(root.resolve()) for record in registry.all_records() for root in record.roots()
    }
    candidates = [
        entry for entry in entries if str(entry.path.expanduser().resolve()) not in existing_paths
    ]
    registered: list[dict[str, Any]] = []
    if args.apply:
        for entry in candidates:
            record = registry.register(entry.path, name=entry.path.name)
            registered.append(record.to_dict())
    payload = {
        "workspace_root": str(workspace_root),
        "applied": bool(args.apply),
        "candidate_count": len(candidates),
        "candidates": [str(entry.path.expanduser().resolve()) for entry in candidates],
        "registered": registered,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"mode: {'applied' if args.apply else 'dry-run'}")
        print(f"candidate_count: {len(candidates)}")
        for entry in candidates:
            print(f"  - {entry.path.expanduser().resolve()}")
        if candidates and not args.apply:
            print("run again with --apply to register these project paths")
    return 0


def _artifact_context(
    args: argparse.Namespace,
) -> tuple[Any, Path, Path, str, str]:
    manager, project, context = _manager_context(args)
    scope_id, record = _scope_for(context, project)
    if bool(getattr(args, "common", False)):
        scope_id = COMMON_SCOPE_ID
        record = None
    project_id = record.project_id if record is not None else ""
    return manager, project, context, scope_id, project_id


def _artifact_body(args: argparse.Namespace) -> str:
    body = getattr(args, "body", None)
    body_file = getattr(args, "body_file", None)
    if body is not None and body_file:
        raise ValueError("provide only one of --body or --body-file")
    if body_file:
        return Path(body_file).expanduser().read_text(encoding="utf-8")
    if body is not None:
        return str(body)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("note body is required; use --body, --body-file, or stdin")


def _render_artifact(artifact: Any, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(artifact.to_dict(), indent=2))
        return
    print(f"id: {artifact.metadata.artifact_id}")
    print(f"title: {artifact.metadata.title}")
    print(f"scope_id: {artifact.metadata.scope_id}")
    print(f"path: {artifact.path}")
    print()
    sys.stdout.write(artifact.body)


def notes_create_command(args: argparse.Namespace) -> int:
    from ..artifacts import NoteStore

    manager, _project, context, scope_id, project_id = _artifact_context(args)
    note = NoteStore(context, scope_id=scope_id, config=manager.config).create(
        title=args.title,
        body=_artifact_body(args),
        project_id=project_id,
        task_id=args.task_id or "",
        agent_name=args.agent_name or "",
        author_kind=args.author_kind,
        sensitivity=args.sensitivity,
    )
    _render_artifact(note, json_output=args.json)
    return 0


def notes_list_command(args: argparse.Namespace) -> int:
    from ..artifacts import NoteStore

    manager, _project, context, scope_id, _project_id = _artifact_context(args)
    notes = NoteStore(context, scope_id=scope_id, config=manager.config).list(limit=args.limit)
    if args.json:
        print(json.dumps([note.to_dict() for note in notes], indent=2))
    elif not notes:
        print("no notes")
    else:
        for note in notes:
            print(
                f"{note.metadata.created_at[:19]}  {note.metadata.artifact_id}  "
                f"{note.metadata.title}"
            )
    return 0


def notes_read_command(args: argparse.Namespace) -> int:
    from ..artifacts import NoteStore

    manager, _project, context, scope_id, _project_id = _artifact_context(args)
    note = NoteStore(context, scope_id=scope_id, config=manager.config).read(args.identifier)
    if note is None:
        print(f"note not found: {args.identifier}")
        return 1
    _render_artifact(note, json_output=args.json)
    return 0


def notes_draft_command(args: argparse.Namespace) -> int:
    from ..scratchpad import ScratchpadStore

    manager, _project, context, scope_id, project_id = _artifact_context(args)
    draft = ScratchpadStore(context, scope_id=scope_id, config=manager.config).create(
        title=args.title,
        body=_artifact_body(args),
        project_id=project_id,
        task_id=args.task_id or "",
        agent_name=args.agent_name or "",
        author_kind=args.author_kind,
        sensitivity=args.sensitivity,
    )
    _render_artifact(draft, json_output=args.json)
    return 0


def notes_drafts_command(args: argparse.Namespace) -> int:
    from ..scratchpad import ScratchpadStore

    manager, _project, context, scope_id, _project_id = _artifact_context(args)
    drafts = ScratchpadStore(context, scope_id=scope_id, config=manager.config).list(
        archived=args.archived,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([draft.to_dict() for draft in drafts], indent=2))
    elif not drafts:
        print("no archived drafts" if args.archived else "no drafts")
    else:
        for draft in drafts:
            print(
                f"{draft.metadata.created_at[:19]}  {draft.metadata.artifact_id}  "
                f"{draft.metadata.title}"
            )
    return 0


def notes_promote_command(args: argparse.Namespace) -> int:
    from ..scratchpad import ScratchpadStore

    manager, _project, context, scope_id, _project_id = _artifact_context(args)
    note = ScratchpadStore(context, scope_id=scope_id, config=manager.config).promote(
        args.identifier
    )
    _render_artifact(note, json_output=args.json)
    return 0


def notes_archive_command(args: argparse.Namespace) -> int:
    from ..scratchpad import ScratchpadStore

    manager, _project, context, scope_id, _project_id = _artifact_context(args)
    draft = ScratchpadStore(context, scope_id=scope_id, config=manager.config).archive(
        args.identifier
    )
    _render_artifact(draft, json_output=args.json)
    return 0


def _handoff_items(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        items.extend(part.strip() for part in value.split(";") if part.strip())
    return items


def _handoff_store(args: argparse.Namespace):
    from ..handoff import HandoffStore

    manager, _project, context, scope_id, project_id = _artifact_context(args)
    store = HandoffStore(context, scope_id=scope_id, config=manager.config)
    return store, project_id


def _create_handoff_revision(
    args: argparse.Namespace,
    *,
    stream_id: str | None = None,
    supersedes: str | None = None,
) -> Any:
    store, project_id = _handoff_store(args)
    return store.create_revision(
        title=args.title,
        agent_name=args.agent_name,
        stream_id=stream_id,
        supersedes=supersedes,
        accomplished=_handoff_items(args.accomplished),
        blocked=_handoff_items(args.blocked),
        next_steps=_handoff_items(args.next_steps),
        target_agent=args.target_agent,
        priority=args.priority,
        project_id=project_id,
    )


def handoff_create_command(args: argparse.Namespace) -> int:
    packet = _create_handoff_revision(args)
    if args.json:
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(f"handoff created: {packet.revision_id}")
        print(f"thread: {packet.stream_id}")
        print(f"path: {packet.artifact_path}")
    return 0


def handoff_revise_command(args: argparse.Namespace) -> int:
    store, _project_id = _handoff_store(args)
    parent = store.read(session_id=args.revision_id)
    if parent is None:
        print(f"handoff revision not found: {args.revision_id}")
        return 1
    packet = _create_handoff_revision(
        args,
        stream_id=parent.stream_id,
        supersedes=parent.revision_id,
    )
    if args.json:
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(f"handoff revised: {packet.revision_id}")
        print(f"thread: {packet.stream_id}")
        print(f"supersedes: {parent.revision_id}")
    return 0


def handoff_list_command(args: argparse.Namespace) -> int:
    store, _project_id = _handoff_store(args)
    packets = store.list(limit=args.limit, stream_id=args.thread)
    if args.json:
        print(json.dumps([packet.to_dict() for packet in packets], indent=2))
    elif not packets:
        print("no handoffs")
    else:
        for packet in packets:
            state = "closed" if packet.closed else "open"
            print(f"{packet.timestamp[:19]}  {packet.revision_id}  {packet.title}  [{state}]")
    return 0


def handoff_read_command(args: argparse.Namespace) -> int:
    store, _project_id = _handoff_store(args)
    packet = (
        store.read(stream_id=args.identifier)
        if args.thread
        else store.read(session_id=args.identifier)
    )
    if packet is None:
        print(f"handoff not found: {args.identifier}")
        return 1
    if args.json:
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(f"revision: {packet.revision_id}")
        print(f"thread: {packet.stream_id}")
        print(f"title: {packet.title}")
        print(f"agent: {packet.agent_name}")
        print(f"status: {'closed' if packet.closed else 'open'}")
        for label, values in (
            ("accomplished", packet.accomplished),
            ("blocked", packet.blocked),
            ("next", packet.next_steps),
        ):
            if values:
                print(f"{label}:")
                for value in values:
                    print(f"  - {value}")
    return 0


def handoff_threads_command(args: argparse.Namespace) -> int:
    store, _project_id = _handoff_store(args)
    threads = store.list_streams(limit=args.limit)
    if args.json:
        print(json.dumps([thread.to_dict() for thread in threads], indent=2))
    elif not threads:
        print("no handoff threads")
    else:
        for thread in threads:
            state = "closed" if thread.closed else "open"
            print(
                f"{thread.updated_at[:19]}  {thread.stream_id}  "
                f"{thread.title}  {thread.revision_count} revision(s)  [{state}]"
            )
    return 0


def handoff_ack_command(args: argparse.Namespace) -> int:
    store, _project_id = _handoff_store(args)
    if not store.acknowledge(args.revision_id, args.by):
        print(f"handoff revision not found: {args.revision_id}")
        return 1
    print(f"acknowledged: {args.revision_id}")
    return 0


def handoff_close_command(args: argparse.Namespace) -> int:
    store, _project_id = _handoff_store(args)
    if not store.close(args.identifier, actor=args.by, reason=args.reason or ""):
        print(f"handoff thread or revision not found: {args.identifier}")
        return 1
    print(f"closed: {args.identifier}")
    return 0


def _message_bus(args: argparse.Namespace) -> MessageBus:
    manager, project, context = _manager_context(args)
    scope_id, _record = _scope_for(context, project)
    return MessageBus(
        context,
        scope_id=scope_id,
        config=manager.config,
        all_projects=bool(getattr(args, "all_projects", False)),
        include_legacy=bool(getattr(args, "include_legacy", False)),
    )


def messages_list_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    messages = bus.read(
        agent_name=args.agent,
        msg_type=args.type,
        topic=args.topic,
        limit=args.limit,
    )
    payload = [message.to_dict() for message in messages]
    if args.json:
        print(json.dumps(payload, indent=2))
    elif not messages:
        print("no messages")
    else:
        for message in messages:
            target = f" -> {message.to}" if message.to else ""
            topic = f" #{message.topic}" if message.topic else ""
            print(
                f"{message.timestamp[:19]}  [{message.msg_type}]  "
                f"{message.from_agent}{target}{topic}  ({message.scope_id or 'legacy'})"
            )
    return 0


def _payload(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--payload must be a JSON object")
    return parsed


def messages_send_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    message = bus.send(
        args.from_agent,
        args.type,
        _payload(args.payload),
        to=args.to,
        topic=args.topic,
        ttl_hours=args.ttl_hours,
        scope_id=args.scope,
    )
    if args.json:
        print(json.dumps(message.to_dict(), indent=2))
    else:
        print(f"sent: {message.id}")
        print(f"scope_id: {message.scope_id}")
    return 0


def messages_subscribe_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    subscription = bus.subscribe(
        args.agent,
        args.topic,
        ttl_hours=args.ttl_hours,
    )
    if args.json:
        print(json.dumps(subscription.to_dict(), indent=2))
    else:
        print(f"subscribed {subscription.agent_name}: {', '.join(subscription.topics)}")
    return 0


def messages_unsubscribe_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    subscription = bus.unsubscribe(args.agent, args.topic)
    if args.json:
        print(json.dumps(subscription.to_dict(), indent=2))
    else:
        remaining = ", ".join(subscription.topics) or "(none)"
        print(f"subscribed topics for {subscription.agent_name}: {remaining}")
    return 0


def messages_clean_command(args: argparse.Namespace) -> int:
    if not args.all_projects:
        raise PermissionError("message cleanup is queue-wide; pass --all-projects to authorize it")
    bus = _message_bus(args)
    result = bus.reap(
        max_age_hours=args.max_age_hours,
        dry_run=not args.apply,
    )
    result["applied"] = bool(args.apply)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        mode = "applied" if args.apply else "dry-run"
        print(f"mode: {mode}")
        print(f"would_remove: {result['removed_count']}")
        print(f"remaining: {result['remaining_count']}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    start = subparsers.add_parser(
        "start",
        help="Start or resume work with the current scoped context.",
    )
    _add_context_args(start)
    start.add_argument("--task-limit", type=int, default=10)
    start.add_argument("--message-limit", type=int, default=10)
    start.add_argument("--agent-name", default="cli")
    start.add_argument("--skills-prompt", default="")
    start.add_argument("--skills-top-k", type=int, default=5)
    start.add_argument("--no-write-artifacts", action="store_true")
    start.add_argument("--engage", action="store_true")
    start.add_argument("--json", action="store_true")
    start.set_defaults(func=start_command)

    search = subparsers.add_parser(
        "search",
        help="Search the current project and shared context with scoped hybrid retrieval.",
    )
    _add_context_args(search)
    search.add_argument("query", help="Non-empty search query.")
    search.add_argument(
        "--mode",
        choices=["text", "symbol"],
        default="text",
        help="Local retrieval mode when --semantic is not enabled.",
    )
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the local index before searching.",
    )
    search.add_argument(
        "--semantic",
        action="store_true",
        help="Explicitly enable semantic embeddings for this rebuild/query.",
    )
    search.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "ollama"],
        help="Embedding provider used only with --semantic (default: gemini).",
    )
    search.add_argument(
        "--model",
        help="Embedding model override; Gemini defaults to stable gemini-embedding-2.",
    )
    search.add_argument(
        "--all-projects",
        action="store_true",
        help="Explicitly authorize results from every registered project.",
    )
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=search_command)

    projects = subparsers.add_parser(
        "projects",
        help="Inspect projects registered with the central context.",
    )
    project_commands = projects.add_subparsers(dest="projects_command")
    current = project_commands.add_parser("current", help="Show the current project and scope.")
    _add_context_args(current)
    current.add_argument("--json", action="store_true")
    current.set_defaults(func=projects_current_command)
    listing = project_commands.add_parser("list", help="List project registry metadata.")
    listing.add_argument("--config")
    listing.add_argument("--context-root")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=projects_list_command)
    register = project_commands.add_parser("register", help="Register a project path.")
    register.add_argument("project_path")
    register.add_argument("--name")
    register.add_argument("--config")
    register.add_argument("--context-root")
    register.add_argument("--json", action="store_true")
    register.set_defaults(func=projects_register_command)
    project_import = project_commands.add_parser(
        "import",
        help="Preview or import project items from a workspace catalog.",
    )
    project_import.add_argument(
        "--workspace-root",
        default=str(Path.home() / "src"),
        help="Directory containing WORKSPACE.toml (default: ~/src).",
    )
    project_import.add_argument("--config")
    project_import.add_argument("--context-root")
    project_import.add_argument(
        "--no-local",
        action="store_true",
        help="Ignore WORKSPACE.local.toml entries.",
    )
    project_import.add_argument(
        "--apply",
        action="store_true",
        help="Register the previewed project paths; default is dry-run.",
    )
    project_import.add_argument("--json", action="store_true")
    project_import.set_defaults(func=projects_import_command)

    notes = subparsers.add_parser(
        "notes",
        help="Create durable notes and manage temporary drafts.",
    )
    note_commands = notes.add_subparsers(dest="notes_command")

    def add_artifact_context(parser: argparse.ArgumentParser) -> None:
        _add_context_args(parser)
        parser.add_argument(
            "--common",
            action="store_true",
            help="Use the shared common scope instead of the current project.",
        )

    def add_note_content(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("title", help="Readable note title (required).")
        parser.add_argument("--body", help="Note body; reads stdin when omitted.")
        parser.add_argument("--body-file", help="Read the note body from a UTF-8 file.")
        parser.add_argument("--task-id")
        parser.add_argument("--agent-name")
        parser.add_argument(
            "--author-kind",
            choices=("human", "agent", "import", "system"),
            default="human",
        )
        parser.add_argument(
            "--sensitivity",
            choices=("internal", "private", "public", "restricted"),
            default="internal",
        )
        parser.add_argument("--json", action="store_true")

    note_create = note_commands.add_parser("create", help="Create a durable note.")
    add_artifact_context(note_create)
    add_note_content(note_create)
    note_create.set_defaults(func=notes_create_command)

    note_list = note_commands.add_parser("list", help="List durable notes.")
    add_artifact_context(note_list)
    note_list.add_argument("--limit", type=int, default=100)
    note_list.add_argument("--json", action="store_true")
    note_list.set_defaults(func=notes_list_command)

    note_read = note_commands.add_parser("read", help="Read a durable note.")
    add_artifact_context(note_read)
    note_read.add_argument("identifier", help="Full note UUID or contained filename/path.")
    note_read.add_argument("--json", action="store_true")
    note_read.set_defaults(func=notes_read_command)

    note_draft = note_commands.add_parser(
        "draft",
        help="Create a temporary note with a unique readable filename.",
    )
    add_artifact_context(note_draft)
    add_note_content(note_draft)
    note_draft.set_defaults(func=notes_draft_command)

    note_drafts = note_commands.add_parser("drafts", help="List temporary notes.")
    add_artifact_context(note_drafts)
    note_drafts.add_argument("--archived", action="store_true")
    note_drafts.add_argument("--limit", type=int, default=100)
    note_drafts.add_argument("--json", action="store_true")
    note_drafts.set_defaults(func=notes_drafts_command)

    note_promote = note_commands.add_parser(
        "promote",
        help="Copy a draft into durable notes while preserving provenance.",
    )
    add_artifact_context(note_promote)
    note_promote.add_argument("identifier")
    note_promote.add_argument("--json", action="store_true")
    note_promote.set_defaults(func=notes_promote_command)

    note_archive = note_commands.add_parser(
        "archive",
        help="Explicitly move a draft out of the active scratchpad.",
    )
    add_artifact_context(note_archive)
    note_archive.add_argument("identifier")
    note_archive.add_argument("--json", action="store_true")
    note_archive.set_defaults(func=notes_archive_command)

    handoff = subparsers.add_parser(
        "handoff",
        help="Create and follow immutable cross-session handoff threads.",
    )
    handoff_commands = handoff.add_subparsers(dest="handoff_command")

    def add_handoff_content(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--title", required=True, help="Readable handoff title.")
        parser.add_argument("--agent-name", default="cli")
        parser.add_argument("--accomplished", action="append")
        parser.add_argument("--blocked", action="append")
        parser.add_argument("--next", dest="next_steps", action="append")
        parser.add_argument("--target-agent")
        parser.add_argument(
            "--priority",
            choices=("low", "normal", "high", "critical"),
            default="normal",
        )
        parser.add_argument("--json", action="store_true")

    handoff_create = handoff_commands.add_parser(
        "create", help="Start a handoff thread with an immutable revision."
    )
    add_artifact_context(handoff_create)
    add_handoff_content(handoff_create)
    handoff_create.set_defaults(func=handoff_create_command)

    handoff_revise = handoff_commands.add_parser(
        "revise", help="Append a revision that supersedes an earlier handoff."
    )
    add_artifact_context(handoff_revise)
    handoff_revise.add_argument("revision_id", help="Revision to supersede.")
    add_handoff_content(handoff_revise)
    handoff_revise.set_defaults(func=handoff_revise_command)

    handoff_list = handoff_commands.add_parser("list", help="List handoff revisions.")
    add_artifact_context(handoff_list)
    handoff_list.add_argument("--thread", help="Restrict to one thread ID.")
    handoff_list.add_argument("--limit", type=int, default=20)
    handoff_list.add_argument("--json", action="store_true")
    handoff_list.set_defaults(func=handoff_list_command)

    handoff_read = handoff_commands.add_parser("read", help="Read a handoff revision.")
    add_artifact_context(handoff_read)
    handoff_read.add_argument("identifier", help="Revision ID, or thread ID with --thread.")
    handoff_read.add_argument("--thread", action="store_true")
    handoff_read.add_argument("--json", action="store_true")
    handoff_read.set_defaults(func=handoff_read_command)

    handoff_threads = handoff_commands.add_parser("threads", help="List logical handoff threads.")
    add_artifact_context(handoff_threads)
    handoff_threads.add_argument("--limit", type=int, default=100)
    handoff_threads.add_argument("--json", action="store_true")
    handoff_threads.set_defaults(func=handoff_threads_command)

    handoff_ack = handoff_commands.add_parser("ack", help="Acknowledge one revision.")
    add_artifact_context(handoff_ack)
    handoff_ack.add_argument("revision_id")
    handoff_ack.add_argument("--by", required=True)
    handoff_ack.set_defaults(func=handoff_ack_command)

    handoff_close = handoff_commands.add_parser("close", help="Close a handoff thread.")
    add_artifact_context(handoff_close)
    handoff_close.add_argument("identifier", help="Thread or revision ID.")
    handoff_close.add_argument("--by", required=True)
    handoff_close.add_argument("--reason")
    handoff_close.set_defaults(func=handoff_close_command)

    messages = subparsers.add_parser(
        "messages",
        help="Send and read scoped inter-agent messages.",
    )
    message_commands = messages.add_subparsers(dest="messages_command")

    def add_message_context(parser: argparse.ArgumentParser) -> None:
        _add_context_args(parser)
        parser.add_argument(
            "--all-projects",
            action="store_true",
            help="Explicitly authorize all registered project scopes.",
        )
        parser.add_argument(
            "--include-legacy",
            action="store_true",
            help="Include unscoped compatibility records.",
        )

    listing = message_commands.add_parser("list", help="List visible messages.")
    add_message_context(listing)
    listing.add_argument("--agent")
    listing.add_argument("--type")
    listing.add_argument("--topic")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=messages_list_command)

    send = message_commands.add_parser("send", help="Send a scoped message.")
    add_message_context(send)
    send.add_argument("--from", dest="from_agent", required=True)
    send.add_argument("--type", default="status")
    send.add_argument("--payload", help="JSON object payload.")
    send.add_argument("--to")
    send.add_argument("--topic")
    send.add_argument("--ttl-hours", type=int)
    send.add_argument("--scope", help="Destination scope (default: current project).")
    send.add_argument("--json", action="store_true")
    send.set_defaults(func=messages_send_command)

    subscribe = message_commands.add_parser("subscribe", help="Subscribe to topics.")
    add_message_context(subscribe)
    subscribe.add_argument("--agent", required=True)
    subscribe.add_argument("--topic", action="append", required=True)
    subscribe.add_argument("--ttl-hours", type=int)
    subscribe.add_argument("--json", action="store_true")
    subscribe.set_defaults(func=messages_subscribe_command)

    unsubscribe = message_commands.add_parser("unsubscribe", help="Unsubscribe from topics.")
    add_message_context(unsubscribe)
    unsubscribe.add_argument("--agent", required=True)
    unsubscribe.add_argument("--topic", action="append", required=True)
    unsubscribe.add_argument("--json", action="store_true")
    unsubscribe.set_defaults(func=messages_unsubscribe_command)

    clean = message_commands.add_parser("clean", help="Preview or apply retention cleanup.")
    add_message_context(clean)
    clean.add_argument("--max-age-hours", type=int)
    clean.add_argument("--apply", action="store_true", help="Apply removals; default is dry-run.")
    clean.add_argument("--json", action="store_true")
    clean.set_defaults(func=messages_clean_command)
