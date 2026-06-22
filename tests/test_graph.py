from __future__ import annotations

from pathlib import Path

from afs.graph import build_graph
from afs.manager import AFSManager
from afs.schema import AFSConfig, GeneralConfig, WorkspaceDirectory


def test_build_graph_includes_source_bucket_and_project_nodes(tmp_path: Path) -> None:
    workspace_root = tmp_path / "src"
    workspace_root.mkdir()
    project_root = workspace_root / "lab" / "afs"
    project_root.mkdir(parents=True)

    config = AFSConfig(
        general=GeneralConfig(
            context_root=workspace_root / ".context",
            workspace_directories=[WorkspaceDirectory(path=workspace_root)],
        )
    )
    manager = AFSManager(config=config)
    manager.ensure(path=project_root)

    source_project = workspace_root / "lab" / "afs-scawful" / "knowledge"
    source_project.mkdir(parents=True)

    mount_path = (
        project_root
        / ".context"
        / "knowledge"
        / "profile-knowledge-dev-0-knowledge"
    )
    mount_path.symlink_to(source_project, target_is_directory=True)

    graph = build_graph(search_paths=[workspace_root], config=config, max_depth=4)

    nodes_by_type: dict[str, list[dict[str, object]]] = {}
    for node in graph["nodes"]:
        nodes_by_type.setdefault(str(node["type"]), []).append(node)

    workspace_nodes = nodes_by_type.get("workspace_root", [])
    assert any(node["path"] == str(workspace_root) for node in workspace_nodes)

    bucket_nodes = nodes_by_type.get("source_bucket", [])
    assert any(node["label"] == "lab" for node in bucket_nodes)

    project_nodes = nodes_by_type.get("source_project", [])
    assert any(node["label"] == "afs-scawful" for node in project_nodes)
    source_path_nodes = nodes_by_type.get("source_path", [])
    assert any(node["label"] == "knowledge" for node in source_path_nodes)
    project_section_nodes = nodes_by_type.get("project_section", [])
    assert any(node["label"] == "knowledge" for node in project_section_nodes)

    edges = {(edge["from"], edge["to"], edge["kind"]) for edge in graph["edges"]}
    workspace_id = f"workspace:{workspace_root}"
    context_id = f"ctx:{project_root / '.context'}"
    bucket_id = f"bucket:{workspace_root}:lab"
    project_id = f"project:{workspace_root}:lab:afs-scawful"
    source_path_id = f"source_path:{workspace_root}:lab:afs-scawful:knowledge"
    project_section_id = f"project_section:{workspace_root}:lab:afs-scawful:knowledge"
    mount_id = (
        f"mount:{project_root / '.context'}:knowledge:profile-knowledge-dev-0-knowledge"
    )
    mount_dir_id = f"dir:{project_root / '.context'}:knowledge"

    assert (workspace_id, context_id, "contains") in edges
    assert (workspace_id, bucket_id, "contains") in edges
    assert (bucket_id, project_id, "contains") in edges
    assert (project_id, source_path_id, "contains") in edges
    assert (project_id, project_section_id, "contains") in edges
    assert (mount_dir_id, source_path_id, "source") in edges
    assert (mount_id, source_path_id, "source") in edges
