from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import afs.cli.review as review_module
from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import (
    AFSConfig,
    DirectoryConfig,
    GeneralConfig,
    WorkspaceDirectory,
    default_directory_configs,
)


def _remap_directories(**overrides: str) -> list[DirectoryConfig]:
    directories: list[DirectoryConfig] = []
    for directory in default_directory_configs():
        name = (
            overrides.get(directory.role.value, directory.name)
            if directory.role
            else directory.name
        )
        directories.append(
            DirectoryConfig(
                name=name,
                policy=directory.policy,
                description=directory.description,
                role=directory.role,
            )
        )
    return directories


def _make_manager(
    tmp_path: Path,
    *,
    workspace_root: Path | None = None,
    remap: dict[str, str] | None = None,
) -> AFSManager:
    general = GeneralConfig(
        context_root=tmp_path / "global-context",
        workspace_directories=(
            [WorkspaceDirectory(path=workspace_root.resolve())]
            if workspace_root is not None
            else []
        ),
    )
    config = AFSConfig(
        general=general,
        directories=_remap_directories(**(remap or {})),
    )
    return AFSManager(config=config)


def _ensure_project_context(manager: AFSManager, project_path: Path) -> Path:
    project_path.mkdir(parents=True, exist_ok=True)
    context = manager.ensure(path=project_path)
    return context.path


def _queue_review_doc(
    manager: AFSManager,
    context_path: Path,
    *,
    category: str,
    filename: str,
    content: str = "draft",
    legacy_queue: bool = False,
) -> Path:
    if legacy_queue:
        queue_root = context_path / "review"
    else:
        queue_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD) / "review"
    category_root = queue_root / category
    category_root.mkdir(parents=True, exist_ok=True)
    target = category_root / filename
    target.write_text(content, encoding="utf-8")
    return target


def test_review_list_reads_context_local_queue(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    context_path = _ensure_project_context(manager, project_path)
    _queue_review_doc(manager, context_path, category="plans", filename="plan.md")
    _queue_review_doc(
        manager,
        context_path,
        category="automated_reports",
        filename="report.md",
    )
    monkeypatch.setattr(review_module, "load_manager", lambda _config_path=None: manager)

    exit_code = review_module.handle_list(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            json=True,
            category=None,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_path"] == str(context_path)
    assert payload["categories"]["plans"] == ["plan.md"]
    assert payload["categories"]["automated_reports"] == ["report.md"]


def test_review_approve_moves_plan_to_memory_reviewed(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    context_path = _ensure_project_context(manager, project_path)
    source = _queue_review_doc(manager, context_path, category="plans", filename="plan.md")
    monkeypatch.setattr(review_module, "load_manager", lambda _config_path=None: manager)

    exit_code = review_module.handle_approve(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            json=False,
            category=None,
            target="plan.md",
            filename=None,
        )
    )

    assert exit_code == 0
    capsys.readouterr()
    destination = (
        manager.resolve_mount_root(context_path, MountType.MEMORY)
        / "reviewed"
        / "plans"
        / "plan.md"
    )
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "draft"


def test_review_reject_moves_doc_to_history_rejected(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    context_path = _ensure_project_context(manager, project_path)
    source = _queue_review_doc(
        manager,
        context_path,
        category="walkthroughs",
        filename="walkthrough.md",
    )
    monkeypatch.setattr(review_module, "load_manager", lambda _config_path=None: manager)

    exit_code = review_module.handle_reject(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            json=False,
            category=None,
            target="walkthrough.md",
            filename=None,
            reason="needs revision",
        )
    )

    assert exit_code == 0
    capsys.readouterr()
    destination = (
        manager.resolve_mount_root(context_path, MountType.HISTORY)
        / "rejected"
        / "walkthroughs"
        / "walkthrough.md"
    )
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "draft"
    assert destination.with_name("walkthrough.md.reason.txt").read_text(encoding="utf-8") == "needs revision"


def test_review_approve_legacy_project_argument_uses_workspace_directory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace_root = tmp_path / "workspace"
    manager = _make_manager(tmp_path, workspace_root=workspace_root)
    project_path = workspace_root / "project-a"
    context_path = _ensure_project_context(manager, project_path)
    _queue_review_doc(manager, context_path, category="plans", filename="plan.md")
    monkeypatch.setattr(review_module, "load_manager", lambda _config_path=None: manager)

    exit_code = review_module.handle_approve(
        Namespace(
            config=None,
            path=None,
            context_root=None,
            context_dir=None,
            json=False,
            category=None,
            target="project-a",
            filename="plan.md",
        )
    )

    assert exit_code == 0
    capsys.readouterr()
    destination = (
        manager.resolve_mount_root(context_path, MountType.MEMORY)
        / "reviewed"
        / "plans"
        / "plan.md"
    )
    assert destination.exists()


def test_review_approve_uses_remapped_mount_roots(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(
        tmp_path,
        remap={"scratchpad": "notes", "memory": "journal", "history": "ledger"},
    )
    project_path = tmp_path / "project"
    context_path = _ensure_project_context(manager, project_path)
    source = _queue_review_doc(manager, context_path, category="plans", filename="plan.md")
    monkeypatch.setattr(review_module, "load_manager", lambda _config_path=None: manager)

    exit_code = review_module.handle_approve(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            json=False,
            category=None,
            target="plan.md",
            filename=None,
        )
    )

    assert exit_code == 0
    capsys.readouterr()
    destination = context_path / "journal" / "reviewed" / "plans" / "plan.md"
    assert not source.exists()
    assert destination.exists()


def test_review_approve_supports_legacy_queue_root(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    context_path = _ensure_project_context(manager, project_path)
    source = _queue_review_doc(
        manager,
        context_path,
        category="plans",
        filename="legacy.md",
        legacy_queue=True,
    )
    monkeypatch.setattr(review_module, "load_manager", lambda _config_path=None: manager)

    exit_code = review_module.handle_approve(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            json=False,
            category=None,
            target="legacy.md",
            filename=None,
        )
    )

    assert exit_code == 0
    capsys.readouterr()
    destination = (
        manager.resolve_mount_root(context_path, MountType.MEMORY)
        / "reviewed"
        / "plans"
        / "legacy.md"
    )
    assert not source.exists()
    assert destination.exists()


def _make_v2_manager(tmp_path: Path) -> tuple[AFSManager, Path, Path]:
    context_path = tmp_path / ".context"
    project_path = tmp_path / "project-v2"
    project_path.mkdir()
    scaffold_v2(context_path)
    ProjectRegistry(context_path).register(project_path)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_path))
    )
    return manager, context_path, project_path


def test_v2_review_reads_copy_migrated_scratchpad_queue(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    queue = context_path / "scratchpad" / "common" / "review" / "plans"
    queue.mkdir(parents=True)
    source = queue / "migrated-from-v1.md"
    source.write_text("pending plan", encoding="utf-8")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    args = Namespace(
        config=None,
        path=str(project_path),
        context_root=None,
        context_dir=None,
        json=True,
        category=None,
        target=source.name,
        filename=None,
    )
    assert review_module.handle_approve(args) == 0
    payload = json.loads(capsys.readouterr().out)
    destination = (
        context_path
        / "memory"
        / "common"
        / "reviewed"
        / "plans"
        / source.name
    )
    assert payload["destination"] == str(destination)
    assert destination.read_text(encoding="utf-8") == "pending plan"
    assert not source.exists()


def test_v2_review_reads_migrated_queue_and_writes_common_destination(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    canonical = context_path / "human" / "common" / "review" / "plans"
    canonical.mkdir(parents=True)
    source = canonical / "readable-plan.md"
    source.write_text("migrated plan", encoding="utf-8")
    # A copy-migrated duplicate left in the old queue must not make the
    # canonical document ambiguous.
    legacy = context_path / "review" / "plans"
    legacy.mkdir(parents=True)
    (legacy / source.name).write_text("migrated plan", encoding="utf-8")
    pre_fix = context_path / "scratchpad" / "review" / "plans"
    pre_fix.mkdir(parents=True)
    (pre_fix / source.name).write_text("stale pre-fix copy", encoding="utf-8")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    args = Namespace(
        config=None,
        path=str(project_path),
        context_root=None,
        context_dir=None,
        json=True,
        category=None,
        target=source.name,
        filename=None,
    )
    assert review_module.handle_approve(args) == 0
    payload = json.loads(capsys.readouterr().out)
    destination = context_path / "memory" / "common" / "reviewed" / "plans" / source.name
    assert payload["destination"] == str(destination)
    assert destination.read_text(encoding="utf-8") == "migrated plan"
    assert not source.exists()
    assert not (context_path / "memory" / "reviewed").exists()


def test_v2_review_reject_writes_common_history_and_reason(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    queue = context_path / "human" / "common" / "review" / "walkthroughs"
    queue.mkdir(parents=True)
    source = queue / "walkthrough.md"
    source.write_text("needs work", encoding="utf-8")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    assert (
        review_module.handle_reject(
            Namespace(
                config=None,
                path=str(project_path),
                context_root=None,
                context_dir=None,
                json=True,
                category=None,
                target=source.name,
                filename=None,
                reason="add verification",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    destination = (
        context_path
        / "history"
        / "common"
        / "rejected"
        / "walkthroughs"
        / source.name
    )
    assert payload["destination"] == str(destination)
    assert destination.read_text(encoding="utf-8") == "needs work"
    assert destination.with_name("walkthrough.md.reason.txt").read_text(
        encoding="utf-8"
    ) == "add verification"
    assert not (context_path / "history" / "rejected").exists()


def test_v2_review_rejects_a_linked_queue_leaf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    category = context_path / "human" / "common" / "review" / "plans"
    category.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        (category / "linked.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        review_module.handle_approve(
            Namespace(
                config=None,
                path=str(project_path),
                context_root=None,
                context_dir=None,
                json=False,
                category=None,
                target="linked.md",
                filename=None,
            )
        )
    assert outside.read_text(encoding="utf-8") == "outside"


def test_v2_review_rejects_a_linked_queue_category(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    queue_root = context_path / "human" / "common" / "review"
    queue_root.mkdir(parents=True)
    outside = tmp_path / "outside-category"
    outside.mkdir()
    (outside / "leak.md").write_text("outside", encoding="utf-8")
    try:
        (queue_root / "plans").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        review_module.handle_list(
            Namespace(
                config=None,
                path=str(project_path),
                context_root=None,
                context_dir=None,
                json=True,
                category=None,
            )
        )


def test_v2_review_rejects_linked_reason_before_moving_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    queue = context_path / "human" / "common" / "review" / "plans"
    queue.mkdir(parents=True)
    source = queue / "plan.md"
    source.write_text("pending plan", encoding="utf-8")
    destination = context_path / "history" / "common" / "rejected" / "plans" / source.name
    destination.parent.mkdir(parents=True)
    outside = tmp_path / "outside-reason.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        destination.with_name(f"{source.name}.reason.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        review_module.handle_reject(
            Namespace(
                config=None,
                path=str(project_path),
                context_root=None,
                context_dir=None,
                json=False,
                category=None,
                target=source.name,
                filename=None,
                reason="needs work",
            )
        )

    assert source.read_text(encoding="utf-8") == "pending plan"
    assert not destination.exists()
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("linked_leaf", [False, True])
def test_v2_review_rejects_linked_destination_path(
    tmp_path: Path,
    monkeypatch,
    linked_leaf: bool,
) -> None:
    manager, context_path, project_path = _make_v2_manager(tmp_path)
    queue = context_path / "human" / "common" / "review" / "plans"
    queue.mkdir(parents=True)
    (queue / "plan.md").write_text("safe source", encoding="utf-8")
    destination_root = context_path / "memory" / "common" / "reviewed"
    outside = tmp_path / "outside-destination"
    outside.mkdir()
    try:
        if linked_leaf:
            destination_parent = destination_root / "plans"
            destination_parent.mkdir(parents=True)
            (destination_parent / "plan.md").symlink_to(outside / "plan.md")
        else:
            destination_root.parent.mkdir(parents=True, exist_ok=True)
            destination_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(review_module, "load_manager", lambda _path=None: manager)

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        review_module.handle_approve(
            Namespace(
                config=None,
                path=str(project_path),
                context_root=None,
                context_dir=None,
                json=False,
                category=None,
                target="plan.md",
                filename=None,
            )
        )
    assert not (outside / "plan.md").exists()
