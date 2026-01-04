"""Warm AFS context data and optional embedding indexes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..core import resolve_context_root
from ..discovery import discover_contexts, get_project_stats
from ..embeddings import build_embedding_index, create_ollama_embed_fn
from ..workspace_sync import load_workspace_entries, resolve_config_output, sync_workspace_config
from ..cli._utils import write_config
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

AGENT_NAME = "context-warm"
AGENT_DESCRIPTION = "Sync workspace paths, discover contexts, and refresh embeddings."


@dataclass
class EmbeddingProject:
    name: str
    path: Path
    enabled: bool
    provider: str | None
    model: str | None
    include_patterns: list[str]
    exclude_patterns: list[str]
    max_files: int | None
    knowledge_roots: list[Path]


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Sync workspace paths, discover contexts, and refresh embeddings.")
    parser.add_argument(
        "--workspace-root",
        default=str(Path.home() / "src"),
        help="Workspace root for WORKSPACE.toml (default: ~/src).",
    )
    parser.add_argument(
        "--discover-path",
        action="append",
        help="Search path override for context discovery (repeatable).",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Discovery depth.")
    parser.add_argument("--ignore", action="append", help="Directory names to ignore.")
    parser.add_argument("--skip-workspace-sync", action="store_true", help="Skip workspace sync.")
    parser.add_argument("--skip-discover", action="store_true", help="Skip context discovery.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding refresh.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write config updates.")
    parser.add_argument(
        "--embedding-config",
        default=str(Path.home() / ".context" / "embedding_service" / "projects.json"),
        help="Embedding projects config path.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "none"],
        help="Override embedding provider.",
    )
    parser.add_argument("--embedding-model", help="Override embedding model.")
    parser.add_argument(
        "--embedding-host",
        default="http://localhost:11435",
        help="Ollama host (default: http://localhost:11435).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Seconds between runs (0 = run once).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum runs (0 = unlimited when interval > 0).",
    )
    parser.add_argument(
        "--sleep-first",
        action="store_true",
        help="Sleep for interval before first run.",
    )
    return parser


def _sync_workspace(args: argparse.Namespace, config) -> tuple[int, list[str]]:
    notes: list[str] = []
    root = Path(args.workspace_root).expanduser().resolve()
    try:
        entries = load_workspace_entries(
            root,
            include_sections=True,
            include_items=True,
            include_local=True,
        )
    except FileNotFoundError as exc:
        notes.append(str(exc))
        return 0, notes

    sync_workspace_config(config, entries, merge=True)
    if not args.dry_run:
        output = resolve_config_output(Path(args.config) if args.config else None)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_config(output, config)
    return len(entries), notes


def _discover_contexts(args: argparse.Namespace, config) -> tuple[list[dict], dict, list[str]]:
    notes: list[str] = []
    search_paths = args.discover_path
    paths = [Path(path).expanduser() for path in search_paths] if search_paths else None
    projects = discover_contexts(
        search_paths=paths,
        max_depth=args.max_depth,
        ignore_names=args.ignore,
        config=config,
    )
    stats = get_project_stats(projects) if projects else {"total_projects": 0, "total_mounts": 0}
    stats["invalid_projects"] = sum(1 for project in projects if not project.is_valid)
    contexts = [
        {
            "path": str(project.path),
            "project_name": project.project_name,
            "is_valid": project.is_valid,
            "total_mounts": project.total_mounts,
        }
        for project in projects
    ]
    return contexts, stats, notes


def _refresh_embeddings(args: argparse.Namespace, config) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    config_path = Path(args.embedding_config).expanduser().resolve()
    projects = _load_embedding_projects(config_path)
    if not projects:
        notes.append(f"no embedding projects at {config_path}")
        return [], notes

    context_root = resolve_context_root(config, None)
    results: list[dict] = []

    for project in projects:
        if not project.enabled:
            continue
        provider = args.embedding_provider or project.provider
        if not provider:
            notes.append(f"embedding disabled for {project.name}")
            continue
        embed_fn = None
        if provider == "ollama":
            model = args.embedding_model or project.model or "nomic-embed-text"
            try:
                embed_fn = create_ollama_embed_fn(model=model, host=args.embedding_host)
            except RuntimeError as exc:
                notes.append(f"{project.name}: {exc}")
                continue
        elif provider == "none":
            embed_fn = None
        else:
            notes.append(f"{project.name}: unsupported provider {provider}")
            continue

        sources = [project.path, *project.knowledge_roots]
        output_dir = context_root / "knowledge" / project.name
        result = build_embedding_index(
            sources,
            output_dir,
            include_patterns=project.include_patterns or None,
            exclude_patterns=project.exclude_patterns or None,
            max_files=project.max_files,
            embed_fn=embed_fn,
        )
        results.append(
            {
                "project": project.name,
                "provider": provider,
                "output_dir": str(output_dir),
                "summary": result.summary(),
                "indexed": result.indexed,
                "skipped": result.skipped,
                "errors": result.errors[:5],
            }
        )
        if result.errors:
            notes.append(f"{project.name}: {len(result.errors)} embedding errors")

    return results, notes


def _load_embedding_projects(path: Path) -> list[EmbeddingProject]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    entries: Iterable[dict]
    if isinstance(payload, dict):
        entries = [value for value in payload.values() if isinstance(value, dict)]
    elif isinstance(payload, list):
        entries = [value for value in payload if isinstance(value, dict)]
    else:
        entries = []

    projects: list[EmbeddingProject] = []
    for entry in entries:
        name = entry.get("name") or entry.get("project")
        path_value = entry.get("path")
        if not isinstance(name, str) or not isinstance(path_value, str):
            continue
        root = Path(path_value).expanduser().resolve()
        if not root.exists():
            continue
        knowledge_roots = [
            Path(p).expanduser().resolve()
            for p in entry.get("knowledge_roots", [])
            if isinstance(p, str)
        ]
        projects.append(
            EmbeddingProject(
                name=name,
                path=root,
                enabled=bool(entry.get("enabled", True)),
                provider=entry.get("embedding_provider"),
                model=entry.get("embedding_model"),
                include_patterns=[
                    pattern for pattern in entry.get("include_patterns", []) if isinstance(pattern, str)
                ],
                exclude_patterns=[
                    pattern for pattern in entry.get("exclude_patterns", []) if isinstance(pattern, str)
                ],
                max_files=entry.get("max_files") if isinstance(entry.get("max_files"), int) else None,
                knowledge_roots=knowledge_roots,
            )
        )

    return projects


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    config = load_agent_config(args.config)
    runs = 0

    if args.interval > 0 and args.sleep_first:
        time.sleep(args.interval)

    while True:
        runs += 1
        started_at = now_iso()
        start = time.monotonic()
        notes: list[str] = []
        metrics: dict[str, int | float] = {}
        payload: dict[str, object] = {}

        if not args.skip_workspace_sync:
            count, sync_notes = _sync_workspace(args, config)
            metrics["workspace_entries"] = count
            notes.extend(sync_notes)

        if not args.skip_discover:
            contexts, stats, discover_notes = _discover_contexts(args, config)
            payload["contexts"] = contexts
            payload["context_stats"] = stats
            metrics["contexts"] = stats.get("total_projects", 0)
            metrics["invalid_contexts"] = stats.get("invalid_projects", 0)
            notes.extend(discover_notes)

        if not args.skip_embeddings:
            embedding_results, embed_notes = _refresh_embeddings(args, config)
            payload["embedding_results"] = embedding_results
            metrics["embedding_projects"] = len(embedding_results)
            notes.extend(embed_notes)

        result = AgentResult(
            name=AGENT_NAME,
            status="ok" if not notes else "warn",
            started_at=started_at,
            finished_at=now_iso(),
            duration_seconds=time.monotonic() - start,
            metrics=metrics,
            notes=notes[:5],
            payload=payload,
        )

        output_path = Path(args.output).expanduser().resolve() if args.output else None
        emit_result(
            result,
            output_path=output_path,
            force_stdout=args.stdout,
            pretty=args.pretty,
        )

        if args.interval <= 0:
            break
        if args.max_runs and runs >= args.max_runs:
            break
        time.sleep(args.interval)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
