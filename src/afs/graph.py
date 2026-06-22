"""Build a graph export for AFS contexts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from . import __version__
from .config import load_config_model
from .discovery import discover_contexts
from .mapping import resolve_directory_name
from .models import ContextRoot, MountType
from .schema import AFSConfig

_PREFERRED_PROJECT_SECTIONS = (
    "src",
    "apps",
    "docs",
    "knowledge",
    "skills",
    "scripts",
    "tools",
    "tests",
    "models",
    "evals",
)


def build_graph(
    search_paths: Iterable[Path] | None = None,
    *,
    max_depth: int = 3,
    ignore_names: Iterable[str] | None = None,
    config: AFSConfig | None = None,
) -> dict[str, object]:
    config = config or load_config_model()
    workspace_roots = _resolve_workspace_roots(search_paths, config)
    contexts = discover_contexts(
        search_paths=search_paths,
        max_depth=max_depth,
        ignore_names=ignore_names,
        config=config,
    )

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    contexts_payload: list[dict[str, object]] = []
    mounts_summary: dict[str, int] = {}
    seen_node_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict[str, object]) -> None:
        node_id = str(node["id"])
        if node_id in seen_node_ids:
            return
        seen_node_ids.add(node_id)
        nodes.append(node)

    def add_edge(from_id: str, to_id: str, kind: str) -> None:
        edge_key = (from_id, to_id, kind)
        if edge_key in seen_edges:
            return
        seen_edges.add(edge_key)
        edges.append({"from": from_id, "to": to_id, "kind": kind})

    for workspace_root in workspace_roots:
        add_node(
            {
                "id": _workspace_root_id(workspace_root),
                "type": "workspace_root",
                "label": workspace_root.name or str(workspace_root),
                "path": str(workspace_root),
            }
        )

    for context in contexts:
        context_workspace = _workspace_for_context(context.path, workspace_roots)
        ctx_id = _context_id(context)
        context_label = (
            f"{context.project_name}/.context"
            if context_workspace is not None
            and context.path.parent == context_workspace
            else context.project_name
        )
        add_node(
            {
                "id": ctx_id,
                "type": "context",
                "label": context_label,
                "path": str(context.path),
            }
        )
        if context_workspace is not None:
            add_edge(_workspace_root_id(context_workspace), ctx_id, "contains")

        dir_ids: dict[str, str] = {}
        for mount_type in MountType:
            dir_name = resolve_directory_name(
                mount_type,
                afs_directories=config.directories,
                metadata=context.metadata,
            )
            dir_id = _dir_id(context, mount_type)
            dir_ids[mount_type.value] = dir_id
            add_node(
                {
                    "id": dir_id,
                    "type": "mount_dir",
                    "label": dir_name,
                    "mount_type": mount_type.value,
                    "path": str(context.path / dir_name),
                }
            )
            add_edge(ctx_id, dir_id, "contains")

        mounts_payload: list[dict[str, object]] = []
        for mount_type, mounts in context.mounts.items():
            dir_name = resolve_directory_name(
                mount_type,
                afs_directories=config.directories,
                metadata=context.metadata,
            )
            for mount in mounts:
                mount_id = _mount_id(context, mount_type, mount.name)
                mount_path = context.path / dir_name / mount.name
                add_node(
                    {
                        "id": mount_id,
                        "type": "mount",
                        "label": mount.name,
                        "mount_type": mount_type.value,
                        "path": str(mount_path),
                        "source": str(mount.source),
                        "is_symlink": mount.is_symlink,
                    }
                )
                add_edge(dir_ids.get(mount_type.value, ctx_id), mount_id, "contains")
                if mount.is_symlink:
                    _append_source_topology(
                        add_node=add_node,
                        add_edge=add_edge,
                        mount_dir_id=dir_ids.get(mount_type.value, ctx_id),
                        mount_id=mount_id,
                        mount_source=mount.source,
                        workspace_roots=workspace_roots,
                    )
                mounts_payload.append(
                    {
                        "id": mount_id,
                        "name": mount.name,
                        "mount_type": mount_type.value,
                        "path": str(mount_path),
                        "source": str(mount.source),
                        "is_symlink": mount.is_symlink,
                    }
                )
                mounts_summary[mount_type.value] = (
                    mounts_summary.get(mount_type.value, 0) + 1
                )

        contexts_payload.append(
            {
                "id": ctx_id,
                "name": context.project_name,
                "path": str(context.path),
                "metadata": context.metadata.to_dict(),
                "mounts": mounts_payload,
            }
        )

    summary = {
        "total_contexts": len(contexts),
        "total_mounts": sum(mounts_summary.values()),
        "mounts_by_type": mounts_summary,
    }

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "afs_version": __version__,
            "context_root": str(config.general.context_root),
            "max_depth": max_depth,
            "ignore": list(ignore_names or config.general.discovery_ignore),
        },
        "workspaces": [
            {
                "path": str(ws.path),
                "description": ws.description,
            }
            for ws in config.general.workspace_directories
        ],
        "contexts": contexts_payload,
        "nodes": nodes,
        "edges": edges,
        "summary": summary,
    }


def write_graph(graph: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    return output_path


def default_graph_path(config: AFSConfig | None = None) -> Path:
    config = config or load_config_model()
    return config.general.context_root / "index" / "afs_graph.json"


def _context_id(context: ContextRoot) -> str:
    return f"ctx:{context.path}"


def _dir_id(context: ContextRoot, mount_type: MountType) -> str:
    return f"dir:{context.path}:{mount_type.value}"


def _mount_id(context: ContextRoot, mount_type: MountType, name: str) -> str:
    return f"mount:{context.path}:{mount_type.value}:{name}"


def _workspace_root_id(path: Path) -> str:
    return f"workspace:{path}"


def _source_bucket_id(workspace_root: Path, bucket: str) -> str:
    return f"bucket:{workspace_root}:{bucket}"


def _source_project_id(workspace_root: Path, bucket: str, project: str) -> str:
    return f"project:{workspace_root}:{bucket}:{project}"


def _source_path_id(workspace_root: Path, bucket: str, project: str, suffix: str) -> str:
    return f"source_path:{workspace_root}:{bucket}:{project}:{suffix}"


def _project_section_id(workspace_root: Path, bucket: str, project: str, section: str) -> str:
    return f"project_section:{workspace_root}:{bucket}:{project}:{section}"


def _resolve_workspace_roots(
    search_paths: Iterable[Path] | None,
    config: AFSConfig,
) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    candidates = list(search_paths or [ws.path for ws in config.general.workspace_directories])
    for candidate in candidates:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _workspace_for_context(context_path: Path, workspace_roots: Iterable[Path]) -> Path | None:
    context_path = context_path.expanduser().resolve()
    for workspace_root in workspace_roots:
        try:
            context_path.relative_to(workspace_root)
        except ValueError:
            continue
        return workspace_root
    return None


def _source_components(
    source: Path,
    workspace_roots: Iterable[Path],
) -> tuple[Path, str, str, str, tuple[str, ...]] | None:
    source = source.expanduser().resolve()
    if source.is_file():
        source = source.parent

    for workspace_root in workspace_roots:
        try:
            relative = source.relative_to(workspace_root)
        except ValueError:
            continue
        parts = [part for part in relative.parts if part not in ("", ".")]
        if not parts:
            return workspace_root, workspace_root.name or str(workspace_root), "", "", ()
        bucket = parts[0]
        project = parts[1] if len(parts) > 1 else parts[0]
        suffix = tuple(parts[2:]) if len(parts) > 2 else ()
        return workspace_root, workspace_root.name or str(workspace_root), bucket, project, suffix
    return None


def _append_source_topology(
    *,
    add_node,
    add_edge,
    mount_dir_id: str,
    mount_id: str,
    mount_source: Path,
    workspace_roots: Iterable[Path],
) -> None:
    components = _source_components(mount_source, workspace_roots)
    if components is None:
        return

    workspace_root, workspace_label, bucket, project, suffix = components
    workspace_id = _workspace_root_id(workspace_root)
    bucket_id = _source_bucket_id(workspace_root, bucket)
    project_id = _source_project_id(workspace_root, bucket, project)

    add_node(
        {
            "id": workspace_id,
            "type": "workspace_root",
            "label": workspace_label,
            "path": str(workspace_root),
        }
    )
    if bucket:
        add_node(
            {
                "id": bucket_id,
                "type": "source_bucket",
                "label": bucket,
                "path": str(workspace_root / bucket),
            }
        )
        add_edge(workspace_id, bucket_id, "contains")
    add_node(
        {
            "id": project_id,
            "type": "source_project",
            "label": project,
            "path": str(workspace_root / bucket / project) if bucket else str(workspace_root / project),
            "source": str(mount_source),
        }
    )
    source_anchor_id = project_id
    if suffix:
        suffix_path = "/".join(suffix)
        source_path_id = _source_path_id(workspace_root, bucket, project, suffix_path)
        add_node(
            {
                "id": source_path_id,
                "type": "source_path",
                "label": suffix[-1],
                "path": str(mount_source),
                "source": str(mount_source),
            }
        )
        add_edge(project_id, source_path_id, "contains")
        source_anchor_id = source_path_id

    add_edge(bucket_id if bucket else workspace_id, project_id, "contains")
    for section_name, section_path in _project_sections(workspace_root, bucket, project):
        section_id = _project_section_id(workspace_root, bucket, project, section_name)
        add_node(
            {
                "id": section_id,
                "type": "project_section",
                "label": section_name,
                "path": str(section_path),
            }
        )
        add_edge(project_id, section_id, "contains")
    add_edge(mount_dir_id, source_anchor_id, "source")
    add_edge(mount_id, source_anchor_id, "source")


def _project_sections(
    workspace_root: Path,
    bucket: str,
    project: str,
) -> list[tuple[str, Path]]:
    project_root = workspace_root / bucket / project if bucket else workspace_root / project
    if not project_root.exists() or not project_root.is_dir():
        return []

    sections: list[tuple[str, Path]] = []
    for name in _PREFERRED_PROJECT_SECTIONS:
        candidate = project_root / name
        if candidate.exists():
            sections.append((name, candidate))
    return sections
