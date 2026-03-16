from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.config as config_module
import afs.core as core_module
from afs.cli._utils import AFS_DIRS
from afs.cli.core import status_command
from afs.context_index import ContextSQLiteIndex
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig


def _build_context(tmp_path: Path) -> tuple[AFSConfig, Path]:
    context_root = tmp_path / ".context"
    for name in AFS_DIRS:
        (context_root / name).mkdir(parents=True, exist_ok=True)
    (context_root / "scratchpad" / "note.md").write_text("status coverage", encoding="utf-8")

    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
            agent_workspaces_dir=context_root / "workspaces",
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
