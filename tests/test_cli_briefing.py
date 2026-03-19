from __future__ import annotations

import json
from pathlib import Path

from afs.agent_registry import AgentRegistry
from afs.agents.base import AgentResult, emit_result
from afs.cli import briefing


def test_briefing_reads_agents_from_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(briefing, "PROJECTS", {})
    monkeypatch.setattr(briefing, "_fetch_halext_tasks", lambda: [])
    monkeypatch.setattr(briefing, "_latest_weekly_carryover", lambda: [])

    emit_result(
        AgentResult(
            name="context-audit",
            status="ok",
            started_at="2026-03-19T08:00:00",
            finished_at="2026-03-19T08:00:05",
            duration_seconds=5.0,
        ),
        output_path=tmp_path / "audit.json",
        force_stdout=False,
        pretty=False,
    )

    payload = briefing._build_briefing(days=1)
    assert payload["active_agents"]
    assert payload["active_agents"][0]["name"] == "context-audit"


def test_briefing_registry_loader_accepts_list_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    registry = AgentRegistry()
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text(
        json.dumps(
            [
                {
                    "name": "history-memory",
                    "task": "Consolidate recent history events into durable memory summaries.",
                    "status": "running",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = briefing._read_agent_registry()
    assert loaded == [
        {
            "name": "history-memory",
            "task": "Consolidate recent history events into durable memory summaries.",
            "status": "running",
        }
    ]


def test_briefing_skips_gws_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(briefing, "PROJECTS", {})
    monkeypatch.setattr(briefing, "_fetch_halext_tasks", lambda: [])
    monkeypatch.setattr(briefing, "_latest_weekly_carryover", lambda: [])
    monkeypatch.setattr(
        briefing,
        "_get_gws_client",
        lambda: (_ for _ in ()).throw(AssertionError("gws client should not be used")),
    )

    payload = briefing._build_briefing(days=1, include_gws=False)

    assert payload["gws_available"] is False
    assert payload["calendar_agenda"] == []
    assert payload["gmail_unread"] == []


def test_briefing_reads_gws_when_authenticated(monkeypatch) -> None:
    monkeypatch.setattr(briefing, "PROJECTS", {})
    monkeypatch.setattr(briefing, "_fetch_halext_tasks", lambda: [])
    monkeypatch.setattr(briefing, "_latest_weekly_carryover", lambda: [])

    class _FakeGWSClient:
        available = True
        authenticated = True

        def calendar_agenda(self) -> list[dict[str, str]]:
            return [{"summary": "Sync", "start": {"dateTime": "2026-03-19T09:00:00"}}]

        def gmail_unread(self) -> list[dict[str, str]]:
            return [{"id": "abc123", "snippet": "Need a response"}]

    monkeypatch.setattr(briefing, "_get_gws_client", lambda: _FakeGWSClient())

    payload = briefing._build_briefing(days=1, include_gws=True)

    assert payload["gws_available"] is True
    assert payload["calendar_agenda"][0]["summary"] == "Sync"
    assert payload["gmail_unread"][0]["snippet"] == "Need a response"


def test_briefing_skips_gws_when_client_is_not_authenticated(monkeypatch) -> None:
    monkeypatch.setattr(briefing, "PROJECTS", {})
    monkeypatch.setattr(briefing, "_fetch_halext_tasks", lambda: [])
    monkeypatch.setattr(briefing, "_latest_weekly_carryover", lambda: [])

    class _FakeGWSClient:
        available = True
        authenticated = False

        def calendar_agenda(self) -> list[dict[str, str]]:
            raise AssertionError("calendar_agenda should not run without auth")

        def gmail_unread(self) -> list[dict[str, str]]:
            raise AssertionError("gmail_unread should not run without auth")

    monkeypatch.setattr(briefing, "_get_gws_client", lambda: _FakeGWSClient())

    payload = briefing._build_briefing(days=1, include_gws=True)

    assert payload["gws_available"] is False
    assert payload["calendar_agenda"] == []
    assert payload["gmail_unread"] == []
