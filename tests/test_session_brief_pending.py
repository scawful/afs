"""The short session brief surfaces what is waiting for the named agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from afs.handoff import HandoffStore
from afs.manager import AFSManager
from afs.messages import MessageBus
from afs.schema import AFSConfig, GeneralConfig
from afs.session_bootstrap import build_session_bootstrap, render_session_bootstrap


def _setup(tmp_path: Path) -> tuple[AFSManager, Path]:
    context = tmp_path / "context"
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context)))
    manager.ensure(path=tmp_path, context_root=context)
    return manager, context


def test_brief_lists_messages_handoffs_inbox_board_and_lease(tmp_path: Path) -> None:
    manager, context = _setup(tmp_path)
    scratchpad = context / "scratchpad"
    lock = tmp_path / "heavy.lock"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (scratchpad / "inbox.md").write_text(
        "Current goal: Ship 1.0\n"
        "FORMAT: YYYY-MM-DD HH:MMZ from -> to: ask (path)\n"
        f"{today} 01:00Z codex -> cursor: Add the CI flag (handoffs/ci.md)\n"
        f"DONE {today} 01:05Z codex -> cursor: Old ask\n"
        f"2026-01-01 01:00Z codex -> cursor: Ancient ask\n"
        f"{today} 01:10Z codex -> claude: Not for cursor\n",
        encoding="utf-8",
    )
    (scratchpad / "agent_board.md").write_text(
        "# Agent board\n"
        f"Heavy lock: truth is {lock}/owner\n\n"
        "## Codex — 2026-10-20 01:20Z\nResult: x\nRan: y\nNext: z\n"
        "Holds: kydrog_boss.asm, ROM $3AA800-$3AAFFF\nHandoff: h.md\n\n"
        "## Cursor — 2026-10-20 01:00Z\nResult: a\nRan: b\nNext: c\nHolds: none\nHandoff: h2.md\n",
        encoding="utf-8",
    )
    bus = MessageBus(context, scope_id="common", include_legacy=True)
    ask = bus.send("codex", "request", {"text": "please review #300"}, to="cursor")
    bus.send("codex", "request", {"text": "for claude"}, to="claude")
    HandoffStore(context, scope_id="common", config=manager.config).create_revision(
        title="Take over CI", agent_name="codex", next_steps=["run ci"], target_agent="cursor"
    )

    result = build_session_bootstrap(manager, context, short=True, agent_name="cursor")
    pending = result["pending"]

    assert pending["agent"] == "cursor"
    assert [item["id"] for item in pending["messages"]["items"]] == [ask.id]
    assert pending["handoffs"]["items"][0]["title"] == "Take over CI"
    assert pending["inbox"]["goal"] == "Ship 1.0"
    assert pending["inbox"]["open_for_agent"] == 1
    assert "Add the CI flag" in pending["inbox"]["lines"][0]
    assert [hold["owner"] for hold in pending["board"]["holds"]] == ["Codex"]
    assert pending["board"]["own_section"]["owner"] == "Cursor"
    assert pending["leases"] == [f"{lock}: free"]
    assert len(json.dumps(result)) < 8000

    rendered = render_session_bootstrap(result)
    assert "Goal: Ship 1.0" in rendered
    assert "please review #300" in rendered
    assert "Take over CI" in rendered


def test_brief_without_agent_skips_personal_sections(tmp_path: Path) -> None:
    manager, context = _setup(tmp_path)
    MessageBus(context, scope_id="common", include_legacy=True).send(
        "codex", "request", {"text": "hi"}, to="cursor"
    )
    result = build_session_bootstrap(manager, context, short=True)
    assert result["pending"] == {"agent": ""}
