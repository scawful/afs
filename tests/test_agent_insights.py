from __future__ import annotations

import json
from pathlib import Path

import afs.agents.insights as insights_agent
from afs.agent_defaults import default_agent_configs
from afs.agents import get_agent_registry
from afs.agents.insights import main
from afs.context_layout import scaffold_v2
from afs.history import append_history_event, resolve_history_root
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig


def _config(tmp_path: Path, context: Path, *, project: Path | None = None) -> Path:
    config = tmp_path / "afs.toml"
    agent = ""
    if project is not None:
        agent = (
            "\n[profiles]\n"
            'active_profile = "work"\n'
            "\n[profiles.work]\n"
            "[[profiles.work.agent_configs]]\n"
            'name = "insights-reflect"\n'
            'module = "afs.agents.insights"\n'
            'schedule = "weekly"\n'
            f'project_path = "{project}"\n'
        )
    config.write_text(
        f'[general]\ncontext_root = "{context}"\n'
        "\n[agents]\ndefault_set = false\n"
        f"{agent}",
        encoding="utf-8",
    )
    return config


def test_insights_agent_is_registered_but_never_in_default_set() -> None:
    assert "insights-reflect" in get_agent_registry()
    assert "insights-reflect" not in {
        agent.name for agent in default_agent_configs(AFSConfig())
    }


def test_insights_agent_skips_without_explicit_scope(
    tmp_path: Path,
    capsys,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    config = _config(tmp_path, context)

    assert main(["--config", str(config), "--stdout"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "skipped"
    assert payload["metrics"]["candidates"] == 0
    assert payload["payload"]["network_used"] is False
    assert payload["payload"]["model_used"] is False


def test_scheduled_insights_agent_uses_configured_project_without_network(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(context)
    record = ProjectRegistry(context).register(project)
    config = _config(tmp_path, context, project=project)
    for index in (1, 2):
        append_history_event(
            resolve_history_root(context),
            "agent_lifecycle",
            "agent.worker",
            op="failed",
            event_id=f"scheduled-failure-{index}",
            context_root=context,
            metadata={
                "scope_id": record.scope_id,
                "project_id": record.project_id,
                "scope_attribution": "registry",
                "status": "failed",
            },
        )

    monkeypatch.setattr(
        "afs.insights.iter_history_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduled reflection must not scan full history")
        ),
    )

    assert main(["--config", str(config), "--stdout"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["metrics"] == {"evidence_events": 2, "candidates": 1}
    assert payload["payload"]["scope_id"] == record.scope_id
    assert payload["payload"]["network_used"] is False
    assert payload["payload"]["model_used"] is False
    assert Path(payload["payload"]["candidate"]["path"]).is_file()

    append_history_event(
        resolve_history_root(context),
        "agent_lifecycle",
        "agent.worker",
        op="failed",
        event_id="scheduled-failure-3",
        context_root=context,
        metadata={
            "scope_id": record.scope_id,
            "project_id": record.project_id,
            "scope_attribution": "registry",
            "status": "failed",
        },
    )
    assert main(["--config", str(config), "--stdout"]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["metrics"]["candidates"] == 0
    assert repeated["payload"]["candidate_created"] is False
    assert repeated["payload"]["bound_evidence_digest"] != (
        repeated["payload"]["inspected_evidence_digest"]
    )


def test_scheduled_insights_agent_uses_bounded_recent_history_window(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(context)
    record = ProjectRegistry(context).register(project)
    config = _config(tmp_path, context, project=project)
    history_root = resolve_history_root(context)
    metadata = {
        "scope_id": record.scope_id,
        "project_id": record.project_id,
        "scope_attribution": "registry",
    }
    for index in (1, 2):
        append_history_event(
            history_root,
            "agent_lifecycle",
            "agent.worker",
            op="failed",
            event_id=f"old-failure-{index}",
            context_root=context,
            metadata={**metadata, "status": "failed"},
        )
    for index in (1, 2):
        append_history_event(
            history_root,
            "agent_lifecycle",
            "agent.worker",
            op="completed",
            event_id=f"recent-completion-{index}",
            context_root=context,
            metadata={**metadata, "status": "completed"},
        )

    monkeypatch.setattr(
        insights_agent,
        "DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW",
        2,
    )
    assert main(["--config", str(config), "--stdout"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    assert payload["metrics"] == {"evidence_events": 2, "candidates": 0}
    assert payload["payload"]["history_window"] == 2
    assert "candidate" not in payload["payload"]
