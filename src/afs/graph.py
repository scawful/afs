"""Build a graph export for AFS contexts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import load_config_model
from .discovery import discover_contexts
from .mapping import resolve_directory_name
from .models import ContextRoot, MountType
from .schema import AFSConfig
from . import __version__


def build_graph(
    search_paths: Iterable[Path] | None = None,
    *,
    max_depth: int = 3,
    ignore_names: Iterable[str] | None = None,
    config: AFSConfig | None = None,
) -> dict[str, object]:
    config = config or load_config_model()
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

    for context in contexts:
        ctx_id = _context_id(context)
        nodes.append(
            {
                "id": ctx_id,
                "type": "context",
                "label": context.project_name,
                "path": str(context.path),
            }
        )

        dir_ids: dict[str, str] = {}
        for mount_type in MountType:
            dir_name = resolve_directory_name(
                mount_type,
                afs_directories=config.directories,
                metadata=context.metadata,
            )
            dir_id = _dir_id(context, mount_type)
            dir_ids[mount_type.value] = dir_id
            nodes.append(
                {
                    "id": dir_id,
                    "type": "mount_dir",
                    "label": dir_name,
                    "mount_type": mount_type.value,
                    "path": str(context.path / dir_name),
                }
            )
            edges.append({"from": ctx_id, "to": dir_id, "kind": "contains"})

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
                nodes.append(
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
                edges.append(
                    {
                        "from": dir_ids.get(mount_type.value, ctx_id),
                        "to": mount_id,
                        "kind": "contains",
                    }
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
