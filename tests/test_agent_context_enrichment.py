from __future__ import annotations

from datetime import datetime, timedelta, timezone

from afs.agents.context_warm import _WarmAgent
from afs.agents.journal_agent import _JournalAgent


def _iso_ago(*, minutes: int = 0, days: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes, days=days)).isoformat()


def test_warm_agent_only_flags_recent_changes() -> None:
    agent = _WarmAgent()
    agent.get_recent_events = lambda **_kwargs: [  # type: ignore[method-assign]
        {"timestamp": _iso_ago(minutes=90), "type": "context_write"},
        {"timestamp": _iso_ago(minutes=10), "type": "context_write"},
    ]

    assert agent.has_recent_changes(since_minutes=30) is True
    assert agent.has_recent_changes(since_minutes=5) is False


def test_journal_agent_activity_summary_respects_day_window() -> None:
    agent = _JournalAgent()
    agent.get_recent_events = lambda **_kwargs: [  # type: ignore[method-assign]
        {
            "timestamp": _iso_ago(days=20),
            "source": "workspace-analyst",
            "op": "old report",
            "metadata": {},
        },
        {
            "timestamp": _iso_ago(days=2),
            "source": "mission-runner",
            "op": "recent mission",
            "metadata": {},
        },
    ]

    summary = agent.get_agent_activity_summary(days=7)

    assert summary == ["- mission-runner: recent mission"]
