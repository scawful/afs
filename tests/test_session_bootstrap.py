from __future__ import annotations

import json
import shutil
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

import afs.cli.core as cli_core_module
from afs.agent_jobs import AgentJobQueue
from afs.agent_runs import AgentRunStore
from afs.cli.core import session_bootstrap_command
from afs.context_index import ContextSQLiteIndex
from afs.context_layout import scaffold_v2
from afs.hivemind import HivemindBus
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import (
    AFSConfig,
    DirectoryConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    default_directory_configs,
)
from afs.tasks import TaskQueue
from afs.work_assistant import WorkAssistantStore


def test_v2_bootstrap_does_not_append_through_linked_daily_history_log(
    tmp_path: Path,
) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    common = context_root / "history" / "common"
    common.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("do not append\n", encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    try:
        (common / f"events_{stamp}.jsonl").symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    summary = build_session_bootstrap(manager, context_root)

    assert summary["context_path"] == str(context_root)
    assert outside.read_text(encoding="utf-8") == "do not append\n"


def test_v2_bootstrap_rejects_symlinked_work_assistant_database(tmp_path: Path) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    outside = tmp_path / "outside.sqlite3"
    outside.write_text("do not modify", encoding="utf-8")
    db_path = context_root / ".afs" / "compat" / "global" / "work_assistant.sqlite3"
    try:
        db_path.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    summary = build_session_bootstrap(manager, context_root, record_event=False)

    assert "symbolic link or reparse point" in summary["work_assistant"]["error"]
    assert outside.read_text(encoding="utf-8") == "do not modify"


def test_v2_bootstrap_rejects_symlinked_compat_metadata(tmp_path: Path) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    outside = tmp_path / "metadata.json"
    outside.write_text(
        json.dumps({"description": "outside-private-metadata"}),
        encoding="utf-8",
    )
    metadata_path = context_root / ".afs" / "compat" / "metadata.json"
    try:
        metadata_path.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        build_session_bootstrap(manager, context_root, record_event=False)
    assert "outside-private-metadata" in outside.read_text(encoding="utf-8")


def test_v2_bootstrap_ignores_compat_metadata_directory_redirects(tmp_path: Path) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("outside-directory-private", encoding="utf-8")
    metadata_path = context_root / ".afs" / "compat" / "metadata.json"
    metadata_path.write_text(
        json.dumps({"directories": {"scratchpad": str(outside)}}),
        encoding="utf-8",
    )
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    summary = build_session_bootstrap(manager, context_root, record_event=False)

    assert "outside-directory-private" not in json.dumps(summary)
    assert manager.resolve_mount_root(context_root, MountType.SCRATCHPAD) == (
        context_root / "scratchpad"
    )


def test_v2_bootstrap_status_diff_freshness_and_scratchpad_are_scope_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))

    for scope_path, marker in (
        (Path("common"), "common-visible"),
        (Path("projects") / alpha_record.project_id, "alpha-visible"),
        (Path("projects") / beta_record.project_id, "beta-private"),
    ):
        for category in ("scratchpad", "knowledge", "memory"):
            root = context_root / category / scope_path
            root.mkdir(parents=True, exist_ok=True)
            (root / "state.md").write_text(marker, encoding="utf-8")

    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.SCRATCHPAD, MountType.KNOWLEDGE, MountType.MEMORY],
        include_content=True,
    )
    common_reports = context_root / "scratchpad" / "common" / "afs_agents"
    beta_reports = (
        context_root
        / "scratchpad"
        / "projects"
        / beta_record.project_id
        / "afs_agents"
    )
    common_reports.mkdir(parents=True)
    beta_reports.mkdir(parents=True)
    (common_reports / "context_warm.json").write_text(
        '{"status":"common-report-visible"}', encoding="utf-8"
    )
    (beta_reports / "context_warm.json").write_text(
        '{"status":"beta-report-private"}', encoding="utf-8"
    )
    alpha_scratchpad = (
        context_root / "scratchpad" / "projects" / alpha_record.project_id
    )
    beta_scratchpad = context_root / "scratchpad" / "projects" / beta_record.project_id
    beta_memory = context_root / "memory" / "projects" / beta_record.project_id
    alpha_memory = context_root / "memory" / "projects" / alpha_record.project_id
    (beta_memory / "entries.jsonl").write_text(
        '{"id":"beta-memory-linked","domain":"private"}\n', encoding="utf-8"
    )
    alpha_reports = alpha_scratchpad / "afs_agents"
    alpha_reports.mkdir(parents=True)
    (alpha_scratchpad / "deferred.md").write_text(
        "alpha-visible", encoding="utf-8"
    )
    (beta_reports / "context_watch.json").write_text(
        '{"status":"beta-report-linked"}', encoding="utf-8"
    )
    (alpha_scratchpad / "state.md").unlink()
    try:
        (alpha_scratchpad / "state.md").symlink_to(beta_scratchpad / "state.md")
        (alpha_memory / "entries.jsonl").symlink_to(beta_memory / "entries.jsonl")
        (alpha_reports / "context_watch.json").symlink_to(
            beta_reports / "context_watch.json"
        )
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setenv("AFS_SESSION_ID", "alpha-scope-test")
    first = build_session_bootstrap(manager, context_root, project_path=alpha)
    first_text = json.dumps(first)
    assert "alpha-visible" in first_text
    assert "common-visible" in first_text
    assert "common-report-visible" in first_text
    assert "beta-private" not in first_text
    assert "beta-report-private" not in first_text
    assert "beta-memory-linked" not in first_text
    assert "beta-report-linked" not in first_text
    assert first["status"]["mount_counts"]["knowledge"] == 2
    assert first["mount_freshness"]["knowledge"]["file_count"] == 2

    alpha_file = context_root / "knowledge" / "projects" / alpha_record.project_id / "state.md"
    beta_file = context_root / "knowledge" / "projects" / beta_record.project_id / "state.md"
    alpha_file.write_text("alpha-visible changed", encoding="utf-8")
    beta_file.write_text("beta-private changed", encoding="utf-8")

    second = build_session_bootstrap(
        manager,
        context_root,
        project_path=alpha,
        record_event=False,
    )
    second_text = json.dumps(second)
    assert any(
        item["relative_path"] == f"projects/{alpha_record.project_id}/state.md"
        for item in second["diff"]["modified"]
    )
    assert "beta-private" not in second_text
    assert all(
        beta_record.project_id not in str(item)
        for key in ("added", "modified", "deleted")
        for item in second["diff"][key]
    )
    assert beta_record.project_id not in json.dumps(second["session_changes"])


def test_v2_bootstrap_rejects_a_linked_project_scope_root(tmp_path: Path) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    beta_root = context_root / "scratchpad" / "projects" / beta_record.project_id
    beta_root.mkdir(parents=True)
    (beta_root / "state.md").write_text("beta-root-private", encoding="utf-8")
    alpha_root = context_root / "scratchpad" / "projects" / alpha_record.project_id
    try:
        alpha_root.symlink_to(beta_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        build_session_bootstrap(
            manager,
            context_root,
            project_path=alpha,
            record_event=False,
        )


def test_v2_bootstrap_artifacts_and_snapshots_reject_linked_output_roots(
    tmp_path: Path,
) -> None:
    from afs.context_freshness import save_context_snapshot
    from afs.context_paths import resolve_agent_output_root
    from afs.scopes import resolve_scope
    from afs.session_bootstrap import (
        build_session_bootstrap,
        write_session_bootstrap_artifacts,
    )

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    outside = tmp_path / "outside"
    alpha.mkdir()
    outside.mkdir()
    scaffold_v2(context_root)
    record = ProjectRegistry(context_root).register(alpha)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    summary = build_session_bootstrap(
        manager,
        context_root,
        project_path=alpha,
        record_event=False,
    )
    output_root = resolve_agent_output_root(
        context_root,
        config=manager.config,
        scope_id=record.scope_id,
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        write_session_bootstrap_artifacts(manager, context_root, summary)
    assert list(outside.iterdir()) == []

    output_root.unlink()
    output_root.mkdir()
    snapshot_outside = tmp_path / "snapshot-outside"
    snapshot_outside.mkdir()
    (output_root / "context_snapshots").symlink_to(
        snapshot_outside,
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        save_context_snapshot(
            context_root,
            "linked-output",
            config=manager.config,
            scoped=resolve_scope(context_root, requester_path=alpha),
        )
    assert list(snapshot_outside.iterdir()) == []


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

    bootstrap_job = AgentJobQueue(context_root).create(
        "bootstrap background job",
        "Check background state.",
        created_by="tester",
        priority=1,
    )
    AgentJobQueue(context_root).move(bootstrap_job.id, "done", result="ready")
    AgentRunStore(context_root).start(
        "bootstrap recorded run",
        harness="codex",
        workspace=str(project_path),
    )
    work_store = WorkAssistantStore(context_root, config=config)
    work_store.create_approval(
        target_system="zendesk",
        target_id="ticket-1",
        action="post_ticket_comment",
        summary="Send drafted support reply",
        preview={"text": "Thanks for the report."},
        permission_required="ticket comment approval",
    )

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
    assert payload["agent_manifest"]["available"] is True
    assert payload["agent_jobs"]["total"] == 1
    assert payload["agent_jobs"]["inbox_attention_count"] == 1
    assert "afs agent-jobs inbox" in payload["agent_jobs"]["inbox_command"]
    assert payload["agent_runs"]["recent_count"] == 1
    assert payload["work_assistant"]["summary"]["pending_approvals"] == 1
    assert payload["work_assistant"]["pending_approvals"][0]["summary"] == "Send drafted support reply"
    assert payload["work_assistant"]["communication_guidance"]["sample_count"] == 0
    assert payload["work_assistant"]["communication_preflight"]["missing_style_evidence"] is True
    assert (
        payload["work_assistant"]["communication_preflight"]["approval_guardrail"][
            "requires_explicit_approval"
        ]
        is True
    )
    assert any(
        "style evidence is missing" in line
        for line in payload["work_assistant"]["communication_guidance"]["guidance"]
    )
    assert payload["hivemind"]["recent_count"] == 1
    assert payload["memory"]["entries_count"] == 1
    assert payload["artifact_paths"]["json"].endswith("session_bootstrap.json")
    assert payload["artifact_paths"]["markdown"].endswith("session_bootstrap.md")
    assert Path(payload["artifact_paths"]["json"]).exists()
    assert Path(payload["artifact_paths"]["markdown"]).exists()
    assert any("afs context query" in step for step in payload["startup_sequence"])
    assert any("afs index rebuild" in action for action in payload["recommended_actions"])
    assert any("scratchpad state" in action.lower() for action in payload["recommended_actions"])
    assert any("afs agent-jobs inbox" in action for action in payload["recommended_actions"])
    assert any("afs work approvals list" in action for action in payload["recommended_actions"])


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


def test_v2_bootstrap_reads_only_current_and_common_scopes(tmp_path: Path) -> None:
    from afs.context_layout import scaffold_v2
    from afs.handoff import HandoffStore
    from afs.messages import MessageBus
    from afs.project_registry import ProjectRegistry
    from afs.scratchpad import ScratchpadStore
    from afs.session_bootstrap import build_session_bootstrap, render_session_bootstrap

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    alpha_alias = tmp_path / "alpha-alias"
    beta = tmp_path / "beta"
    alpha.mkdir()
    alpha_alias.mkdir()
    beta.mkdir()
    alpha_subdir = alpha / "src"
    alpha_subdir.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    registry.add_alias(
        alpha_record.project_id,
        alpha_alias,
        requester_path=alpha,
    )
    beta_record = registry.register(beta)
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    manager = AFSManager(config=config)

    MessageBus(context_root, scope_id=alpha_record.scope_id, config=config).send(
        "alpha-agent", "status", {"detail": "alpha-visible"}
    )
    MessageBus(context_root, scope_id=beta_record.scope_id, config=config).send(
        "beta-agent", "status", {"detail": "beta-hidden"}
    )
    MessageBus(context_root, scope_id="common", config=config).send(
        "common-agent", "status", {"detail": "common-visible"}
    )
    ScratchpadStore(
        context_root, scope_id=alpha_record.scope_id, config=config
    ).create(title="Alpha draft", body="alpha")
    ScratchpadStore(
        context_root, scope_id=beta_record.scope_id, config=config
    ).create(title="Beta draft", body="beta")
    HandoffStore(context_root, scope_id=alpha_record.scope_id, config=config).create_revision(
        title="Alpha handoff", agent_name="alpha-agent"
    )
    HandoffStore(context_root, scope_id=beta_record.scope_id, config=config).create_revision(
        title="Beta handoff", agent_name="beta-agent"
    )
    run_store = AgentRunStore(context_root)
    run_store.start("Common run", prompt="common-visible-run")
    run_store.start("Alpha run", workspace=str(alpha), prompt="alpha-visible-run")
    run_store.start("Alpha alias run", workspace=str(alpha_alias))
    run_store.start("Beta run", workspace=str(beta), prompt="beta-hidden-run")

    summary = build_session_bootstrap(
        manager,
        context_root,
        project_path=alpha_subdir,
        record_event=False,
    )

    assert summary["scope_id"] == alpha_record.scope_id
    assert summary["layout_version"] == 2
    assert summary["project"] == "alpha"
    assert {item["from"] for item in summary["hivemind"]["messages"]} == {
        "alpha-agent",
        "common-agent",
    }
    assert [item["title"] for item in summary["scratchpad"]["drafts"]] == [
        "Alpha draft"
    ]
    assert summary["handoff"]["title"] == "Alpha handoff"
    assert {item["task"] for item in summary["agent_runs"]["items"]} == {
        "Alpha alias run",
        "Alpha run",
        "Common run",
    }
    assert "beta-hidden-run" not in json.dumps(summary["agent_runs"])
    assert any("afs search" in step for step in summary["startup_sequence"])
    assert not any("afs context query" in step for step in summary["startup_sequence"])
    assert any("afs search" in action for action in summary["recommended_actions"])
    rendered = render_session_bootstrap(summary)
    assert "## Messages" in rendered
    assert "Hivemind" not in rendered


def test_build_session_bootstrap_ignores_volatile_index_drift(
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

    knowledge_root = context_root / "knowledge"
    scratchpad_root = context_root / "scratchpad"
    (knowledge_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (scratchpad_root / "note.md").write_text("first draft\n", encoding="utf-8")

    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(
        mount_types=[MountType.KNOWLEDGE, MountType.SCRATCHPAD],
        include_content=True,
    )

    (scratchpad_root / "note.md").write_text("updated draft\n", encoding="utf-8")

    summary = build_session_bootstrap(manager, context_root)

    assert summary["status"]["index"]["stale"] is False
    assert not any(
        "afs index rebuild" in action for action in summary["recommended_actions"]
    )


def test_build_session_bootstrap_includes_codebase_summary(
    tmp_path: Path,
) -> None:
    from afs.session_bootstrap import build_session_bootstrap, render_session_bootstrap

    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        )
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path)

    (project_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (project_path / "src").mkdir()
    (project_path / "src" / "demo.py").write_text("def demo() -> int:\n    return 1\n", encoding="utf-8")
    (project_path / "tests").mkdir()
    (project_path / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")

    summary = build_session_bootstrap(manager, project_path / ".context")

    assert "src" in summary["codebase"]["source_roots"]
    assert "tests" in summary["codebase"]["test_roots"]
    assert any("afs context overview" in step for step in summary["startup_sequence"])
    rendered = render_session_bootstrap(summary)
    assert "## Codebase" in rendered
    assert "source_roots: src" in rendered


def test_v2_session_bootstrap_prunes_nested_project_and_visible_context(
    tmp_path: Path,
) -> None:
    from afs.context_layout import scaffold_v2
    from afs.project_registry import ProjectRegistry
    from afs.session_bootstrap import build_session_bootstrap

    alpha = tmp_path / "workspace"
    beta = alpha / "nested-beta"
    context_root = alpha / "central-context"
    beta.mkdir(parents=True)
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    registry.register(alpha)
    registry.register(beta)
    (alpha / "alpha_safe.py").write_text("ALPHA_SAFE = True\n", encoding="utf-8")
    (beta / "beta_confidential_canary.py").write_text(
        "BETA_PRIVATE = True\n",
        encoding="utf-8",
    )
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    summary = build_session_bootstrap(
        manager,
        context_root,
        project_path=alpha,
    )
    rendered = json.dumps(summary["codebase"])

    assert "alpha_safe.py" in rendered
    assert "nested-beta" not in rendered
    assert "beta_confidential_canary" not in rendered
    assert "central-context" not in rendered


def test_session_bootstrap_delivers_explicitly_matched_skill_body(
    tmp_path: Path,
) -> None:
    from afs.session_bootstrap import (
        build_session_bootstrap,
        render_session_bootstrap,
        write_session_bootstrap_artifacts,
    )

    skill_root = tmp_path / "skills"
    skill = skill_root / "quantum" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\n"
        "name: quantum-frobnicate\n"
        "triggers: [quantumfrobnicate]\n"
        "profiles: [general]\n"
        "---\n\n"
        "# Quantum Frobnication\n\n"
        "Frobnicate only after validating the flux boundary.\n",
        encoding="utf-8",
    )
    context_root = tmp_path / ".context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            profiles=ProfilesConfig(
                active_profile="default",
                profiles={"default": ProfileConfig(skill_roots=[skill_root])},
            ),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)

    summary = build_session_bootstrap(
        manager,
        context_root,
        skills_prompt="quantumfrobnicate",
        record_event=False,
    )
    expected_body = (
        "# Quantum Frobnication\n\n"
        "Frobnicate only after validating the flux boundary."
    )
    assert summary["skills"]["matches"][0]["name"] == "quantum-frobnicate"
    assert summary["skills"]["matches"][0]["body"] == expected_body
    rendered = render_session_bootstrap(summary)
    assert "## Relevant Skills" in rendered
    assert expected_body in rendered

    artifacts = write_session_bootstrap_artifacts(manager, context_root, summary)
    persisted = json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
    assert persisted["skills"]["matches"][0]["body"] == expected_body
    assert expected_body in Path(artifacts["markdown"]).read_text(encoding="utf-8")

    no_match = build_session_bootstrap(
        manager,
        context_root,
        skills_prompt="",
        record_event=False,
    )
    assert no_match["skills"]["matches"] == []
    assert "## Relevant Skills" not in render_session_bootstrap(no_match)

    TaskQueue(context_root).create(
        "quantumfrobnicate the next flux boundary",
        created_by="tester",
        priority=1,
    )
    inferred = build_session_bootstrap(
        manager,
        context_root,
        record_event=False,
    )
    assert inferred["skills"]["prompt_source"] == "session_state"
    assert inferred["skills"]["matches"][0]["name"] == "quantum-frobnicate"
    assert inferred["skills"]["matches"][0]["body"] == expected_body


def test_session_bootstrap_surfaces_skill_warnings_without_losing_matches(
    tmp_path: Path,
) -> None:
    from afs.session_bootstrap import (
        build_session_bootstrap,
        render_session_bootstrap,
    )

    skill_root = tmp_path / "skills"
    good = skill_root / "good" / "SKILL.md"
    good.parent.mkdir(parents=True)
    good.write_text(
        "---\nname: good\ntriggers: [diagnosticprobe]\n---\n\n# Good\n",
        encoding="utf-8",
    )
    invalid_skills: list[Path] = []
    for skill_index in range(25):
        directory = (
            "invalid-\x1b\x85`[entry]"
            if skill_index == 0
            else f"invalid-{skill_index:02d}"
        )
        invalid = skill_root / directory / "SKILL.md"
        invalid.parent.mkdir(parents=True)
        invalid.write_text(
            f"---\nname: invalid-{skill_index:02d}\nenforcement:\n"
            + "".join(f"  - rule {index}\n" for index in range(17))
            + "---\n",
            encoding="utf-8",
        )
        invalid_skills.append(invalid)
    context_root = tmp_path / ".context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            profiles=ProfilesConfig(
                active_profile="default",
                profiles={"default": ProfileConfig(skill_roots=[skill_root])},
            ),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)

    summary = build_session_bootstrap(
        manager,
        context_root,
        skills_prompt="diagnosticprobe",
        record_event=False,
    )

    assert summary["skills"]["matches"][0]["name"] == "good"
    assert summary["skills"]["diagnostic_count"] == 25
    assert len(summary["skills"]["diagnostics"]) == 20
    assert summary["skills"]["diagnostics_omitted"] == 5
    assert summary["skills"]["diagnostics"][0]["code"] == "skill_invalid"
    rendered = render_session_bootstrap(summary)
    assert "## Relevant Skills" in rendered
    assert "## Skill Discovery Warnings" in rendered
    from afs.skills import escape_skill_diagnostic_text

    escaped_path = escape_skill_diagnostic_text(
        invalid_skills[0],
        max_chars=240,
        markdown=True,
    )
    warning_section = rendered.split("## Skill Discovery Warnings", 1)[1].split(
        "## Agent Jobs",
        1,
    )[0]
    assert f"[skill\\_invalid] {escaped_path}" in warning_section
    assert "\x1b" not in warning_section
    assert "\x85" not in warning_section
    assert "\\u001b" in warning_section
    assert "\\u0085" in warning_section
    assert "\\`\\[entry\\]" in warning_section
    assert "25 configured skill entry or root" in rendered


def test_session_bootstrap_survives_unresolvable_skill_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    healthy_root = tmp_path / "healthy"
    healthy = healthy_root / "healthy" / "SKILL.md"
    healthy.parent.mkdir(parents=True)
    healthy.write_text(
        "---\nname: healthy\ntriggers: [resolutionprobe]\n---\n\n# Healthy\n",
        encoding="utf-8",
    )
    broken_root = tmp_path / "broken-loop"
    context_root = tmp_path / ".context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            profiles=ProfilesConfig(
                active_profile="default",
                profiles={
                    "default": ProfileConfig(
                        skill_roots=[healthy_root, broken_root]
                    )
                },
            ),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    original_resolve = Path.resolve

    def simulated_resolve(path: Path, *args, **kwargs) -> Path:
        if path == broken_root:
            raise RuntimeError("simulated symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulated_resolve)

    summary = build_session_bootstrap(
        manager,
        context_root,
        skills_prompt="resolutionprobe",
        record_event=False,
    )

    assert summary["skills"]["available"] is True
    assert summary["skills"]["matches"][0]["name"] == "healthy"
    assert summary["skills"]["diagnostic_count"] == 1
    assert summary["skills"]["diagnostics"][0]["code"] == "root_unreadable"


def test_bootstrap_token_budget_sheds_bodies_before_skill_pointers() -> None:
    from afs.session_bootstrap import _apply_token_budget, _estimate_tokens

    summary = {
        "scratchpad": {"state_text": "keep me"},
        "skills": {
            "available": True,
            "matches": [
                {
                    "name": "large",
                    "path": "/skills/large/SKILL.md",
                    "body": "x" * 4000,
                    "body_chars": 4000,
                }
            ],
        },
    }
    truncated = _apply_token_budget(summary, budget=200)
    assert truncated["scratchpad"] == {"state_text": "keep me"}
    assert truncated["skills"]["matches"][0]["name"] == "large"
    assert truncated["skills"]["matches"][0]["body"] == ""
    assert truncated["skills"]["matches"][0]["body_chars"] == 0
    assert truncated["skills"]["matches"][0]["body_omitted"] == "token_budget"
    assert "skills.bodies" in truncated["_budget_info"]["truncated_sections"]
    actual_tokens = _estimate_tokens(json.dumps(truncated, default=str))
    assert truncated["_budget_info"]["estimated_tokens"] == actual_tokens
    assert actual_tokens <= 200


def test_bootstrap_skill_signal_ignores_completed_tasks() -> None:
    from afs.session_bootstrap import _skill_signal

    signal = _skill_signal(
        handoff={"available": False},
        missions={"active": []},
        tasks={
            "items": [
                {"status": "done", "title": "stale completed quantumfrobnicate"},
                {"status": "pending", "title": "active python review"},
            ]
        },
    )
    assert "stale completed" not in signal
    assert "active python review" in signal


def test_bootstrap_skill_signal_prioritizes_open_tasks_before_display_limit(
    tmp_path: Path,
) -> None:
    from afs.session_bootstrap import build_session_bootstrap

    skill_root = tmp_path / "skills"
    skill = skill_root / "queue" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: queue-focus\ntriggers: [pendingneedle]\n---\n\n"
        "Handle the pending work first.\n",
        encoding="utf-8",
    )
    context_root = tmp_path / ".context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            profiles=ProfilesConfig(
                active_profile="default",
                profiles={"default": ProfileConfig(skill_roots=[skill_root])},
            ),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    queue = TaskQueue(context_root)
    for index in range(10):
        completed = queue.create(
            f"completed filler {index}",
            created_by="tester",
            priority=0,
        )
        queue.update_status(completed.id, "done")
    queue.create(
        "pendingneedle remains actionable",
        created_by="tester",
        priority=9,
    )

    summary = build_session_bootstrap(
        manager,
        context_root,
        task_limit=10,
        record_event=False,
    )
    assert summary["tasks"]["items"][0]["status"] == "pending"
    assert summary["skills"]["matches"][0]["name"] == "queue-focus"


def test_bootstrap_renders_bounded_human_confirmed_mission_acceptance(
    tmp_path: Path,
) -> None:
    from afs.human_provenance import _broker_for_reader
    from afs.missions import MissionStore
    from afs.session_bootstrap import build_session_bootstrap, render_session_bootstrap

    context_root = tmp_path / ".context"
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    store = MissionStore(context_root, config=manager.config)
    title = "Ship bounded mission delivery"
    acceptance = "human-visible done criteria"
    authorization = _broker_for_reader(lambda _prompt: "human").confirm_token(
        "human",
        "prompt",
        scope=store.human_acceptance_scope("create", title, acceptance),
    )
    assert authorization is not None
    store.create(
        title=title,
        summary="s" * 3000,
        acceptance=acceptance,
        acceptance_authorization=authorization,
        next_steps=[f"step {index} " + "x" * 600 for index in range(8)],
        metadata={"unbounded": "secret" * 1000},
    )

    summary = build_session_bootstrap(manager, context_root, record_event=False)
    mission = summary["missions"]["active"][0]
    assert len(mission["summary"]) <= 1500
    assert len(mission["next_steps"]) == 5
    assert all(len(step) <= 400 for step in mission["next_steps"])
    assert "metadata" not in mission
    assert "log" not in mission
    assert mission["acceptance_human_confirmed"] is True

    rendered = render_session_bootstrap(summary)
    assert "## Active Missions" in rendered
    assert title in rendered
    assert f"done_when: {acceptance}" in rendered


def test_bootstrap_token_budget_can_shed_missions() -> None:
    from afs.session_bootstrap import _apply_token_budget

    summary = {
        "missions": {
            "available": True,
            "active_count": 1,
            "active": [{"title": "mission", "summary": "x" * 5000}],
        }
    }
    truncated = _apply_token_budget(summary, budget=80)
    assert truncated["missions"] == {"truncated": True, "reason": "token_budget"}
    assert "missions" in truncated["_budget_info"]["truncated_sections"]
