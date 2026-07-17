from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.cli.core as cli_core_module
from afs.agent_jobs import AgentJobQueue
from afs.agent_runs import AgentRunStore
from afs.cli.core import session_bootstrap_command
from afs.context_index import ContextSQLiteIndex
from afs.hivemind import HivemindBus
from afs.manager import AFSManager
from afs.models import MountType
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
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
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

    summary = build_session_bootstrap(
        manager,
        context_root,
        project_path=alpha,
        record_event=False,
    )

    assert summary["scope_id"] == alpha_record.scope_id
    assert summary["project"] == "alpha"
    assert {item["from"] for item in summary["hivemind"]["messages"]} == {
        "alpha-agent",
        "common-agent",
    }
    assert [item["title"] for item in summary["scratchpad"]["drafts"]] == [
        "Alpha draft"
    ]
    assert summary["handoff"]["title"] == "Alpha handoff"
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
