from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.cli.core as cli_core_module
from afs.cli.core import session_bootstrap_command
from afs.hivemind import HivemindBus
from afs.manager import AFSManager
from afs.schema import AFSConfig, DirectoryConfig, GeneralConfig, default_directory_configs
from afs.tasks import TaskQueue


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


def test_session_bootstrap_command_outputs_json_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        ),
        directories=_remap_directories(
            scratchpad="notes",
            items="queue",
            hivemind="bus",
            memory="brain",
        ),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)

    notes_root = context_root / "notes"
    notes_root.mkdir(exist_ok=True)
    (notes_root / "state.md").write_text("bootstrap state", encoding="utf-8")
    (notes_root / "deferred.md").write_text("bootstrap deferred", encoding="utf-8")

    queue = TaskQueue(context_root)
    queue.create("bootstrap task", created_by="tester", priority=2)

    bus = HivemindBus(context_root)
    bus.send("tester", "status", {"detail": "handoff ready"})

    brain_root = context_root / "brain"
    summary_dir = brain_root / "history_consolidation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (brain_root / "entries.jsonl").write_text(
        json.dumps({"id": "mem-1"}) + "\n",
        encoding="utf-8",
    )
    (summary_dir / "mem-1.md").write_text("latest durable memory", encoding="utf-8")

    monkeypatch.setattr(cli_core_module, "load_manager", lambda _config_path=None: manager)

    exit_code = session_bootstrap_command(
        Namespace(
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
            task_limit=5,
            message_limit=5,
            no_write_artifacts=False,
            json=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_path"] == str(context_root)
    assert payload["scratchpad"]["path"].endswith("/notes")
    assert payload["tasks"]["total"] == 1
    assert payload["hivemind"]["recent_count"] == 1
    assert payload["memory"]["entries_count"] == 1
    assert payload["artifact_paths"]["json"].endswith("session_bootstrap.json")
    assert payload["artifact_paths"]["markdown"].endswith("session_bootstrap.md")
    assert Path(payload["artifact_paths"]["json"]).exists()
    assert Path(payload["artifact_paths"]["markdown"]).exists()
    assert any("afs context query" in step for step in payload["startup_sequence"])
    assert any("afs index rebuild" in action for action in payload["recommended_actions"])
    assert any("scratchpad state" in action.lower() for action in payload["recommended_actions"])


def test_build_session_bootstrap_does_not_mutate_hivemind_or_memory(
    tmp_path: Path,
) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        )
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)

    bus = HivemindBus(context_root, config=config)
    message = bus.send("tester", "status", {"detail": "expired soon"}, ttl_hours=1)
    msg_path = context_root / "hivemind" / "tester" / f"{message.id}.json"
    data = json.loads(msg_path.read_text(encoding="utf-8"))
    data["expires_at"] = "2000-01-01T00:00:00+00:00"
    msg_path.write_text(json.dumps(data), encoding="utf-8")

    checkpoint_dir = context_root / "scratchpad" / "afs_agents"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "history_memory_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps({"timestamp": "2024-01-01T00:00:00", "event_id": "evt-1"}),
        encoding="utf-8",
    )

    summary = build_session_bootstrap(manager, context_root)

    assert isinstance(summary["memory"]["status"].get("stale"), bool)
    assert msg_path.exists()
    assert not (context_root / "memory" / "entries.jsonl").exists()
