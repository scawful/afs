from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import afs.cli.core as cli_core_module
from afs.cli.core import memory_consolidate_command
from afs.context_layout import scaffold_v2
from afs.history import append_history_event
from afs.manager import AFSManager
from afs.memory_consolidation import (
    check_consolidation_gates,
    consolidate_history_to_memory,
    memory_status,
    search_memory,
)
from afs.models import MountType
from afs.schema import (
    AFSConfig,
    DirectoryConfig,
    GeneralConfig,
    MemoryConsolidationConfig,
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


def _build_context(
    tmp_path: Path,
    *,
    directories: list[DirectoryConfig] | None = None,
) -> tuple[AFSConfig, Path]:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        ),
        directories=directories or default_directory_configs(),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return config, context_root


def test_consolidate_history_to_memory_writes_entries_and_markdown(tmp_path: Path) -> None:
    directories = _remap_directories(history="ledger", memory="brain", scratchpad="notes")
    config, context_root = _build_context(tmp_path, directories=directories)
    history_root = context_root / "ledger"
    base = datetime.now(timezone.utc).replace(microsecond=0)

    append_history_event(
        history_root,
        "fs",
        "afs.context_fs",
        op="write",
        context_root=context_root,
        metadata={
            "mount_type": "scratchpad",
            "relative_path": "state.md",
            "context_path": str(context_root),
        },
        timestamp=(base + timedelta(seconds=60)).isoformat(),
        event_id="evt-001",
    )
    append_history_event(
        history_root,
        "context",
        "afs.manager",
        op="mount",
        context_root=context_root,
        metadata={
            "mount_type": "knowledge",
            "alias": "docs",
            "context_path": str(context_root),
        },
        timestamp=(base + timedelta(seconds=120)).isoformat(),
        event_id="evt-002",
    )

    result = consolidate_history_to_memory(context_root, config=config)

    assert result.entries_written == 1
    assert result.markdown_written == 1
    assert result.consolidated_events >= 2
    assert result.entries_path == context_root / "brain" / "entries.jsonl"
    assert result.checkpoint_path == context_root / "notes" / "afs_agents" / "history_memory_checkpoint.json"

    entries = [
        json.loads(line)
        for line in result.entries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["source"] == "history.consolidation"
    assert "history-consolidated" in entry["tags"]
    assert entry["_metadata"]["event_types"]["fs"] == 1
    assert entry["_metadata"]["event_types"]["context"] >= 1
    assert "scratchpad/state.md" in entry["output"]
    assert result.markdown_paths[0].exists()


def test_consolidation_checkpoint_skips_old_events(tmp_path: Path) -> None:
    config, context_root = _build_context(tmp_path)
    history_root = context_root / MountType.HISTORY.value
    base = datetime.now(timezone.utc).replace(microsecond=0)

    append_history_event(
        history_root,
        "fs",
        "afs.context_fs",
        op="write",
        context_root=context_root,
        metadata={"mount_type": "scratchpad", "relative_path": "note.md"},
        timestamp=(base + timedelta(seconds=60)).isoformat(),
        event_id="evt-001",
    )

    first = consolidate_history_to_memory(
        context_root,
        config=config,
        write_markdown=False,
    )
    second = consolidate_history_to_memory(
        context_root,
        config=config,
        write_markdown=False,
    )

    append_history_event(
        history_root,
        "review",
        "afs.cli.review",
        op="approve",
        context_root=context_root,
        metadata={"category": "docs", "filename": "design.md"},
        timestamp=(base + timedelta(seconds=120)).isoformat(),
        event_id="evt-002",
    )
    third = consolidate_history_to_memory(
        context_root,
        config=config,
        write_markdown=False,
    )

    entries = [
        json.loads(line)
        for line in (context_root / "memory" / "entries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert first.entries_written == 1
    assert second.entries_written == 0
    assert "no new events" in second.notes
    assert third.entries_written == 1
    assert len(entries) == 2


def test_memory_consolidate_command_outputs_json(tmp_path: Path, monkeypatch, capsys) -> None:
    config, context_root = _build_context(tmp_path)
    history_root = context_root / "history"
    base = datetime.now(timezone.utc).replace(microsecond=0)
    append_history_event(
        history_root,
        "hook",
        "afs.grounding_hooks",
        op="before_context_read",
        context_root=context_root,
        metadata={"status": "ok"},
        timestamp=(base + timedelta(seconds=60)).isoformat(),
        event_id="evt-001",
    )

    manager = AFSManager(config=config)
    monkeypatch.setattr(cli_core_module, "load_manager", lambda _config_path=None: manager)

    exit_code = memory_consolidate_command(
        Namespace(
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
            max_events=None,
            max_events_per_entry=None,
            event_types=None,
            no_markdown=True,
            json=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries_written"] == 1
    assert payload["markdown_written"] == 0
    assert payload["memory_root"] == str(context_root / "memory")


def _build_v2_context(tmp_path: Path) -> tuple[AFSConfig, Path]:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    return config, context_root


def test_v2_consolidation_uses_common_history_memory_and_checkpoint(
    tmp_path: Path,
) -> None:
    config, context_root = _build_v2_context(tmp_path)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    append_history_event(
        context_root / "history",
        "fs",
        "afs.context_fs",
        op="write",
        context_root=context_root,
        metadata={"mount_type": "scratchpad", "relative_path": "note.md"},
        timestamp=timestamp,
        event_id="evt-v2-001",
    )

    result = consolidate_history_to_memory(
        context_root,
        config=config,
        write_markdown=True,
    )

    assert result.history_root == context_root / "history" / "common"
    assert result.memory_root == context_root / "memory" / "common"
    assert result.entries_path == context_root / "memory" / "common" / "entries.jsonl"
    assert result.checkpoint_path == (
        context_root
        / "scratchpad"
        / "common"
        / "afs_agents"
        / "history_memory_checkpoint.json"
    )
    assert result.entries_written == 1
    assert not (context_root / "memory" / "entries.jsonl").exists()


def test_v2_memory_status_and_search_read_common_and_dedupe_prefix_data(
    tmp_path: Path,
) -> None:
    config, context_root = _build_v2_context(tmp_path)
    entry = {
        "id": "history-memory-migrated",
        "instruction": "Recall migrated context",
        "output": "semantic dream result",
        "tags": ["history-consolidated"],
    }
    rendered = json.dumps(entry) + "\n"
    common = context_root / "memory" / "common"
    common.mkdir(parents=True, exist_ok=True)
    (common / "entries.jsonl").write_text(rendered, encoding="utf-8")
    # Early v2 builds wrote the same record at the category root. A copied
    # transition record must not be counted or returned twice.
    (context_root / "memory" / "entries.jsonl").write_text(rendered, encoding="utf-8")

    status = memory_status(context_root, config=config)
    results = search_memory(context_root, "semantic dream", config=config)

    assert status["entries_path"] == str(common / "entries.jsonl")
    assert status["entries_count"] == 1
    assert [result["id"] for result in results] == ["history-memory-migrated"]


def test_v2_consolidation_reads_pre_fix_checkpoint_and_writes_canonical(
    tmp_path: Path,
) -> None:
    config, context_root = _build_v2_context(tmp_path)
    first_timestamp = "2026-07-17T01:00:00+00:00"
    second_timestamp = "2026-07-17T02:00:00+00:00"
    for event_id, timestamp in (
        ("already-consolidated", first_timestamp),
        ("new-after-upgrade", second_timestamp),
    ):
        append_history_event(
            context_root / "history",
            "fs",
            "afs.context_fs",
            op="write",
            context_root=context_root,
            timestamp=timestamp,
            event_id=event_id,
        )
    pre_fix = (
        context_root
        / "scratchpad"
        / "afs_agents"
        / "history_memory_checkpoint.json"
    )
    pre_fix.parent.mkdir(parents=True)
    pre_fix_payload = json.dumps(
        {"timestamp": first_timestamp, "event_id": "already-consolidated"},
        indent=2,
    ) + "\n"
    pre_fix.write_text(pre_fix_payload, encoding="utf-8")

    result = consolidate_history_to_memory(
        context_root,
        config=config,
        write_markdown=False,
    )

    canonical = (
        context_root
        / "scratchpad"
        / "common"
        / "afs_agents"
        / "history_memory_checkpoint.json"
    )
    assert result.consolidated_events == 1
    assert json.loads(canonical.read_text(encoding="utf-8"))["event_id"] == (
        "new-after-upgrade"
    )
    assert pre_fix.read_text(encoding="utf-8") == pre_fix_payload


def test_v1_memory_reads_ignore_non_file_entries_path(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    (context_root / "memory" / "entries.jsonl").mkdir(parents=True)
    config = AFSConfig(general=GeneralConfig(context_root=context_root))

    assert memory_status(context_root, config=config)["entries_count"] == 0
    assert search_memory(context_root, "anything", config=config) == []


def test_v2_consolidation_rejects_a_linked_entries_leaf(tmp_path: Path) -> None:
    config, context_root = _build_v2_context(tmp_path)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    append_history_event(
        context_root / "history",
        "fs",
        "afs.context_fs",
        op="write",
        context_root=context_root,
        metadata={"mount_type": "scratchpad", "relative_path": "note.md"},
        timestamp=timestamp,
        event_id="evt-v2-linked",
    )
    memory_common = context_root / "memory" / "common"
    memory_common.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        (memory_common / "entries.jsonl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        consolidate_history_to_memory(
            context_root,
            config=config,
            write_markdown=False,
        )
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_v2_search_uses_canonical_precedence_for_duplicate_ids(
    tmp_path: Path,
) -> None:
    config, context_root = _build_v2_context(tmp_path)
    canonical = context_root / "memory" / "common"
    canonical.mkdir(parents=True)
    (canonical / "entries.jsonl").write_text(
        json.dumps(
            {
                "id": "same-id",
                "instruction": "canonical record",
                "output": "safe content",
                "tags": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (context_root / "memory" / "entries.jsonl").write_text(
        json.dumps(
            {
                "id": "same-id",
                "instruction": "legacy duplicate",
                "output": "must-not-surface",
                "tags": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert search_memory(
        context_root,
        "must-not-surface",
        config=config,
    ) == []


@pytest.mark.parametrize(
    ("relative_path", "gate_config"),
    [
        (
            Path("scratchpad/common/afs_agents/history_memory.lock"),
            MemoryConsolidationConfig(
                gate_min_hours=0,
                gate_min_events=0,
                gate_min_sessions=0,
            ),
        ),
        (
            Path(
                "scratchpad/common/afs_agents/history_memory_checkpoint.json"
            ),
            MemoryConsolidationConfig(
                gate_min_hours=24,
                gate_min_events=0,
                gate_min_sessions=0,
            ),
        ),
    ],
)
def test_v2_consolidation_gates_reject_linked_managed_paths(
    tmp_path: Path,
    relative_path: Path,
    gate_config: MemoryConsolidationConfig,
) -> None:
    _base_config, context_root = _build_v2_context(tmp_path)
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        memory_consolidation=gate_config,
    )
    linked_path = context_root / relative_path
    linked_path.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / f"outside-{linked_path.name}"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        linked_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        check_consolidation_gates(context_root, config=config)


def test_v2_consolidation_session_gate_counts_canonical_history_events(
    tmp_path: Path,
) -> None:
    _base_config, context_root = _build_v2_context(tmp_path)
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        memory_consolidation=MemoryConsolidationConfig(
            gate_min_hours=0,
            gate_min_events=0,
            gate_min_sessions=1,
        ),
    )
    append_history_event(
        context_root / "history",
        "session",
        "afs.session",
        op="bootstrap",
        context_root=context_root,
        event_id="session-bootstrap-v2",
    )

    gate = check_consolidation_gates(context_root, config=config)

    assert gate.passed is True
    assert gate.gate == "all"
