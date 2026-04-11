from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import afs.cli.core as cli_core_module
import afs.config as config_module
import afs.core as core_module
from afs.cli._utils import AFS_DIRS
from afs.cli.core import agents_watch_command, status_command
from afs.context_index import ContextSQLiteIndex
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, DirectoryConfig, GeneralConfig, default_directory_configs


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


def _build_context(tmp_path: Path) -> tuple[AFSConfig, Path]:
    context_root = tmp_path / ".context"
    for name in AFS_DIRS:
        (context_root / name).mkdir(parents=True, exist_ok=True)
    (context_root / "scratchpad" / "note.md").write_text("status coverage", encoding="utf-8")

    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        )
    )
    manager = AFSManager(config=config)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "README.md").write_text("knowledge mount", encoding="utf-8")
    manager.mount(docs_root, MountType.KNOWLEDGE, alias="docs", context_path=context_root)
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.SCRATCHPAD, MountType.KNOWLEDGE],
        include_content=True,
    )
    return config, context_root


def _clear_profile_env(monkeypatch) -> None:  # noqa: ANN001
    for name in (
        "AFS_PROFILE",
        "AFS_ENABLED_EXTENSIONS",
        "AFS_KNOWLEDGE_MOUNTS",
        "AFS_SKILL_ROOTS",
        "AFS_MODEL_REGISTRIES",
        "AFS_POLICIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_status_command_json_reports_index_and_mount_counts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _clear_profile_env(monkeypatch)
    config, context_root = _build_context(tmp_path)

    monkeypatch.setattr(config_module, "load_config_model", lambda *args, **kwargs: config)
    monkeypatch.setattr(core_module, "find_root", lambda _start_dir=None: context_root)
    monkeypatch.setattr(
        core_module,
        "resolve_context_root",
        lambda _config, linked_root: linked_root or context_root,
    )

    exit_code = status_command(Namespace(start_dir=None, json=True))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_root"] == str(context_root)
    assert payload["valid"] is True
    assert payload["mount_counts"]["scratchpad"] == 1
    assert payload["mount_counts"]["knowledge"] == 1
    assert payload["total_files"] >= 2
    assert payload["mount_health"]["healthy"] is True
    assert payload["index"]["available"] is True
    assert payload["index"]["has_entries"] is True
    assert payload["index"]["total_entries"] >= 1
    assert "maintenance" in payload


def test_context_index_rebuild_is_visible_to_fresh_connections(tmp_path: Path) -> None:
    config, context_root = _build_context(tmp_path)
    scratchpad = context_root / "scratchpad"
    for index in range(64):
        (scratchpad / f"note_{index:03d}.md").write_text(
            f"checkpoint visibility {index}",
            encoding="utf-8",
        )

    manager = AFSManager(config=config)
    index = ContextSQLiteIndex(manager, context_root)
    summary = index.rebuild(
        mount_types=[MountType.SCRATCHPAD, MountType.KNOWLEDGE],
        include_content=True,
    )

    with sqlite3.connect(index.db_path, timeout=5.0) as connection:
        row_count = connection.execute(
            "SELECT COUNT(1) FROM file_index WHERE context_path = ?",
            (str(context_root),),
        ).fetchone()[0]

    assert row_count == summary.rows_written
    wal_path = Path(f"{index.db_path}-wal")
    if wal_path.exists():
        assert wal_path.stat().st_size == 0


def test_status_command_prints_index_rebuild_hint_when_index_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _clear_profile_env(monkeypatch)
    context_root = tmp_path / ".context"
    for name in AFS_DIRS:
        (context_root / name).mkdir(parents=True, exist_ok=True)

    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        )
    )

    monkeypatch.setattr(config_module, "load_config_model", lambda *args, **kwargs: config)
    monkeypatch.setattr(core_module, "find_root", lambda _start_dir=None: context_root)
    monkeypatch.setattr(
        core_module,
        "resolve_context_root",
        lambda _config, linked_root: linked_root or context_root,
    )

    exit_code = status_command(Namespace(start_dir=None, json=False))

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "afs index rebuild --path .  # build context index" in out


def test_status_command_ignores_volatile_index_drift(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _clear_profile_env(monkeypatch)
    config, context_root = _build_context(tmp_path)
    (context_root / "scratchpad" / "note.md").write_text(
        "updated after rebuild",
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "load_config_model", lambda *args, **kwargs: config)
    monkeypatch.setattr(core_module, "find_root", lambda _start_dir=None: context_root)
    monkeypatch.setattr(
        core_module,
        "resolve_context_root",
        lambda _config, linked_root: linked_root or context_root,
    )

    exit_code = status_command(Namespace(start_dir=None, json=True))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["index"]["stale"] is False


def test_agents_watch_command_uses_remapped_history_dir(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        ),
        directories=_remap_directories(history="ledger"),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    (context_root / "ledger" / "events_20260317.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-17T12:00:00+00:00",
                "source": "agent.worker-1",
                "op": "progress",
                "metadata": {"detail": "queued task"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_core_module, "load_manager", lambda _config_path=None: manager)

    exit_code = agents_watch_command(
        Namespace(
            name="worker-1",
            limit=10,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    )

    assert exit_code == 0
    assert "queued task" in capsys.readouterr().out
