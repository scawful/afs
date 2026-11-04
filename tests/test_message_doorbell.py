from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import afs.cli.friendly as friendly
import afs.message_doorbell as doorbell
from afs.cli import main as cli_main
from afs.message_doorbell import addresses_host, hook_main, resolve_message_bus, ring_codex
from afs.messages import MessageBus

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> Path:
    root = tmp_path / ".context"
    (root / "hivemind").mkdir(parents=True, exist_ok=True)
    return root


def _bus(tmp_path: Path, scope_id: str = "project:alpha") -> MessageBus:
    return MessageBus(_context(tmp_path), scope_id=scope_id)


def _wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        doorbell,
        "resolve_message_bus",
        lambda _project, config_path=None: _bus(tmp_path),
    )


def _hook(monkeypatch, capsys, host: str, payload: dict, *, event: str | None = None):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert doorbell.run_hook(host, event=event) == 0
    out = capsys.readouterr().out
    return json.loads(out) if out.strip() else None


def _event(name: str, session_id: str = "sess-1", **extra) -> dict:
    return {"hook_event_name": name, "session_id": session_id, "cwd": "/unused", **extra}


def _start(monkeypatch, capsys, host: str = "codex", session_id: str = "sess-1") -> None:
    assert _hook(monkeypatch, capsys, host, _event("SessionStart", session_id)) is None


def test_session_start_registers_session_without_output(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)

    record = _bus(tmp_path).session_record("codex", "sess-1")
    assert record is not None
    assert record["scope_id"] == "project:alpha"


def test_post_tool_use_delivers_new_message_once(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    sent = _bus(tmp_path).send("claude-kydrog", "request", {"text": "rebase onto main"}, to="codex")

    first = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))
    second = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))

    assert first is not None
    out = first["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    assert sent.id in out["additionalContext"]
    assert "rebase onto main" in out["additionalContext"]
    assert "not instructions from the user" in out["additionalContext"]
    assert "afs messages ack <id> --by codex" in out["additionalContext"]
    assert "does not ack them" in out["additionalContext"]
    assert second is None
    # Injection is not an ack: the message stays unread for the recipient.
    assert [entry["state"] for entry in _bus(tmp_path).inbox("codex")] == ["unread"]


def test_user_prompt_submit_uses_additional_context(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys, host="claude")
    _bus(tmp_path).send("codex", "status", {"text": "tests green"}, to="claude")

    emitted = _hook(monkeypatch, capsys, "claude", _event("UserPromptSubmit"))

    assert emitted["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "tests green" in emitted["hookSpecificOutput"]["additionalContext"]


def test_backlog_from_before_session_start_is_not_pushed(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _bus(tmp_path).send("claude", "request", {"text": "old ask"}, to="codex")
    time.sleep(0.002)
    _start(monkeypatch, capsys)

    assert _hook(monkeypatch, capsys, "codex", _event("PostToolUse")) is None


def test_unregistered_session_is_baselined_silently(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _bus(tmp_path).send("claude", "request", {"text": "old ask"}, to="codex")

    assert _hook(monkeypatch, capsys, "codex", _event("PostToolUse")) is None
    assert _bus(tmp_path).session_record("codex", "sess-1") is not None

    _bus(tmp_path).send("claude", "request", {"text": "new ask"}, to="codex")
    emitted = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))
    context = emitted["hookSpecificOutput"]["additionalContext"]
    assert "new ask" in context
    assert "old ask" not in context


def test_stop_blocks_with_reason_and_does_not_loop(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    sent = _bus(tmp_path).send("claude", "request", {"text": "check the log"}, to="codex")

    blocked = _hook(monkeypatch, capsys, "codex", _event("Stop", stop_hook_active=False))
    again = _hook(monkeypatch, capsys, "codex", _event("Stop", stop_hook_active=False))

    assert blocked["decision"] == "block"
    assert sent.id in blocked["reason"]
    assert "check the log" in blocked["reason"]
    assert again is None


def test_stop_during_hook_continuation_leaves_message_unclaimed(
    tmp_path, monkeypatch, capsys
) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    sent = _bus(tmp_path).send("claude", "request", {"text": "late ask"}, to="codex")

    assert _hook(monkeypatch, capsys, "codex", _event("Stop", stop_hook_active=True)) is None
    emitted = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))
    assert sent.id in emitted["hookSpecificOutput"]["additionalContext"]


def test_subagent_tool_call_does_not_consume_message(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys, host="claude")
    sent = _bus(tmp_path).send("codex", "request", {"text": "for the main agent"}, to="claude")

    sub = _hook(monkeypatch, capsys, "claude", _event("PostToolUse", agent_id="agent-7"))
    main = _hook(monkeypatch, capsys, "claude", _event("PostToolUse"))

    assert sub is None
    assert sent.id in main["hookSpecificOutput"]["additionalContext"]


def test_each_session_gets_its_own_delivery(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys, session_id="sess-a")
    _start(monkeypatch, capsys, session_id="sess-b")
    sent = _bus(tmp_path).send("claude", "status", {"text": "fyi"}, to="codex")

    a = _hook(monkeypatch, capsys, "codex", _event("PostToolUse", "sess-a"))
    b = _hook(monkeypatch, capsys, "codex", _event("PostToolUse", "sess-b"))

    assert sent.id in a["hookSpecificOutput"]["additionalContext"]
    assert sent.id in b["hookSpecificOutput"]["additionalContext"]


def test_acked_own_and_other_recipient_messages_are_not_pushed(
    tmp_path, monkeypatch, capsys
) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    bus = _bus(tmp_path)
    acked = bus.send("claude", "request", {"text": "already read"}, to="codex")
    bus.ack(acked.id, "codex", status="read")
    bus.send("codex", "status", {"text": "my own broadcast"}, to="all")
    bus.send("codex", "status", {"text": "to claude only"}, to="claude")

    assert _hook(monkeypatch, capsys, "codex", _event("PostToolUse")) is None


def test_delivery_caps_per_hook_and_carries_the_rest(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    bus = _bus(tmp_path)
    cap = doorbell.MAX_MESSAGES_PER_DELIVERY
    sent = [bus.send("claude", "status", {"text": f"note {n}"}, to="codex") for n in range(cap + 2)]

    first = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))
    second = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))

    first_text = first["hookSpecificOutput"]["additionalContext"]
    second_text = second["hookSpecificOutput"]["additionalContext"]
    assert [m.id in first_text for m in sent] == [True] * cap + [False] * 2
    assert "More unread messages are waiting" in first_text
    assert [m.id in second_text for m in sent] == [False] * cap + [True] * 2
    assert "More unread messages are waiting" not in second_text


def test_digest_is_one_line_per_message_with_urgent_first(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    bus = _bus(tmp_path)
    cap = doorbell.MAX_MESSAGES_PER_DELIVERY
    chatter = [
        bus.send("claude", "status", {"text": f"status {n}\n" + "detail " * 80}, to="codex")
        for n in range(cap)
    ]
    ask = bus.send("claude", "request", {"text": "please rerun the eval"}, to="codex")

    emitted = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))

    text = emitted["hookSpecificOutput"]["additionalContext"]
    lines = text.splitlines()
    assert lines[1] == f"- [{ask.id}] claude [request]: please rerun the eval"
    assert "detail" not in text
    assert all(len(line) <= doorbell.PREVIEW_CHARS + 80 for line in lines[1:-2])
    assert sum(m.id in text for m in chatter) == cap - 1
    assert "More unread messages are waiting" in text


def test_stop_blocks_only_for_requests_and_blockers_to_this_host(
    tmp_path, monkeypatch, capsys
) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    bus = _bus(tmp_path)
    fyi = bus.send("claude", "status", {"text": "tests green"}, to="codex")
    finding = bus.send("claude", "finding", {"text": "flaky test"}, to="codex")
    broadcast_ask = bus.send("claude", "request", {"text": "anyone free?"}, to="all")

    assert _hook(monkeypatch, capsys, "codex", _event("Stop", turn_id="t1")) is None

    blocker = bus.send("claude", "blocker", {"text": "main is red"}, to="codex-rc")
    blocked = _hook(monkeypatch, capsys, "codex", _event("Stop", turn_id="t2"))
    assert blocked["decision"] == "block"
    assert blocker.id in blocked["reason"]
    assert fyi.id not in blocked["reason"]

    # Stop left the non-urgent messages unclaimed; the next digest carries them.
    emitted = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"))
    text = emitted["hookSpecificOutput"]["additionalContext"]
    assert all(m.id in text for m in (fyi, finding, broadcast_ask))
    assert blocker.id not in text


def test_stop_blocks_at_most_once_per_codex_turn(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    bus = _bus(tmp_path)
    first = bus.send("claude", "request", {"text": "one"}, to="codex")

    blocked = _hook(monkeypatch, capsys, "codex", _event("Stop", turn_id="t1"))
    second = bus.send("claude", "request", {"text": "two"}, to="codex")
    capped = _hook(monkeypatch, capsys, "codex", _event("Stop", turn_id="t1"))
    next_turn = _hook(monkeypatch, capsys, "codex", _event("Stop", turn_id="t2"))

    assert first.id in blocked["reason"]
    assert capped is None
    assert second.id in next_turn["reason"]


def test_stop_budget_resets_on_user_prompt_submit(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys, host="claude")
    bus = _bus(tmp_path)
    bus.send("codex", "request", {"text": "one"}, to="claude")

    assert _hook(monkeypatch, capsys, "claude", _event("Stop"))["decision"] == "block"
    second = bus.send("codex", "request", {"text": "two"}, to="claude")
    assert _hook(monkeypatch, capsys, "claude", _event("Stop")) is None

    prompt = _hook(monkeypatch, capsys, "claude", _event("UserPromptSubmit"))
    assert second.id in prompt["hookSpecificOutput"]["additionalContext"]
    third = bus.send("codex", "request", {"text": "three"}, to="claude")
    assert third.id in _hook(monkeypatch, capsys, "claude", _event("Stop"))["reason"]


def test_concurrent_claims_deliver_exactly_once(tmp_path) -> None:
    bus = _bus(tmp_path)
    bus.register_session("claude", "sess-1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _n: bus.claim_delivery("claude", "sess-1", "m-1"), range(8)))

    assert results.count(True) == 1


def test_hook_fails_open_on_bad_stdin_and_storage_errors(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    assert doorbell.run_hook("codex") == 0
    assert capsys.readouterr().out == ""

    def broken(_project, config_path=None):
        raise PermissionError("project is not registered in central context")

    monkeypatch.setattr(doorbell, "resolve_message_bus", broken)
    assert _hook(monkeypatch, capsys, "codex", _event("PostToolUse")) is None


def test_hook_without_session_id_or_unknown_event_is_silent(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)

    assert _hook(monkeypatch, capsys, "codex", {"hook_event_name": "PostToolUse"}) is None
    assert _hook(monkeypatch, capsys, "codex", _event("PreToolUse")) is None
    assert not (tmp_path / ".context" / "hivemind" / ".sessions").exists()


def test_event_flag_overrides_payload_event(tmp_path, monkeypatch, capsys) -> None:
    _wire(monkeypatch, tmp_path)
    _start(monkeypatch, capsys)
    _bus(tmp_path).send("claude", "request", {"text": "ping"}, to="codex")

    emitted = _hook(monkeypatch, capsys, "codex", _event("PostToolUse"), event="Stop")

    assert emitted["decision"] == "block"


def test_hook_main_exits_zero_on_bad_arguments(capsys) -> None:
    assert hook_main(["--host", "gemini"]) == 0
    assert capsys.readouterr().out == ""


def test_module_entry_point_dispatches_hook_fast_path(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "afs", "messages", "hook", "--host", "claude"],
        input=json.dumps({"hook_event_name": "PostToolUse"}),
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""


def test_resolve_message_bus_uses_project_context(tmp_path) -> None:
    project = tmp_path / "project"
    (project / ".context" / "hivemind").mkdir(parents=True)

    bus = resolve_message_bus(project)

    assert bus.scope_id == "common"
    bus.register_session("claude", "sess-1", cwd=str(project))
    assert (project / ".context" / "hivemind" / ".sessions" / "claude" / "sess-1.json").is_file()


@pytest.mark.parametrize("session_id", ["../escape", "a/b", ".", ""])
def test_register_session_rejects_path_like_ids(tmp_path, session_id) -> None:
    with pytest.raises(ValueError):
        _bus(tmp_path).register_session("codex", session_id)


def test_claim_delivery_rejects_path_like_message_ids(tmp_path) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "sess-1")
    with pytest.raises(ValueError):
        bus.claim_delivery("codex", "sess-1", "../../hivemind/claude/x")


def test_live_sessions_are_scope_filtered_and_newest_first(tmp_path) -> None:
    alpha = _bus(tmp_path, "project:alpha")
    alpha.register_session("codex", "older")
    time.sleep(0.002)
    alpha.register_session("codex", "newer")
    _bus(tmp_path, "project:beta").register_session("codex", "beta-thread")

    assert [r["session_id"] for r in alpha.live_sessions("codex")] == ["newer", "older"]
    assert [r["session_id"] for r in _bus(tmp_path, "project:beta").live_sessions("codex")] == [
        "beta-thread"
    ]
    with pytest.raises(PermissionError):
        alpha.live_sessions("codex", scope_id="project:beta")


def test_register_session_prunes_stale_records_and_claims(tmp_path) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "stale")
    bus.claim_delivery("codex", "stale", "m-1")
    sessions_dir = tmp_path / ".context" / "hivemind" / ".sessions" / "codex"
    old = time.time() - 15 * 86400
    os.utime(sessions_dir / "stale.json", (old, old))

    bus.register_session("codex", "fresh")

    assert not (sessions_dir / "stale.json").exists()
    assert not (sessions_dir / "stale.delivered").exists()
    assert (sessions_dir / "fresh.json").is_file()


@pytest.mark.parametrize(
    ("to", "host", "expected"),
    [
        ("codex", "codex", True),
        ("Codex-RC", "codex", True),
        ("claude,codex", "codex", True),
        ("claude/codex-lyra", "codex", True),
        ("claude-kydrog", "claude", True),
        ("all", "codex", False),
        ("*", "claude", False),
        ("claude", "codex", False),
        (None, "codex", False),
        ("codexian", "codex", False),
    ],
)
def test_addresses_host(to, host, expected) -> None:
    assert addresses_host(to, host) is expected


def _fake_codex(tmp_path: Path, *, exit_code: int = 0) -> tuple[Path, Path]:
    args_file = tmp_path / "codex-args.txt"
    script = tmp_path / "bin" / "codex"
    script.parent.mkdir(exist_ok=True)
    script.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args_file}'\n"
        f"echo 'queue said no' >&2\nexit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script, args_file


def test_ring_queues_pointer_on_newest_codex_thread(tmp_path) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "thread-old")
    time.sleep(0.002)
    bus.register_session("codex", "thread-new")
    message = bus.send("claude", "request", {"text": "first line\nsecond line"}, to="codex")
    script, args_file = _fake_codex(tmp_path)

    result = ring_codex(bus, message, codex_bin=str(script))

    assert result["status"] == "rang"
    assert result["thread"] == "thread-new"
    argv = args_file.read_text(encoding="utf-8").splitlines()
    assert argv[:4] == ["queue", "--thread", "thread-new", "--message"]
    assert argv[4].startswith(f"AFS message {message.id} from claude: first line")
    assert "second line" not in argv[4]
    assert "--resume" not in argv


def test_ring_resume_passes_resume_flag(tmp_path) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "thread-1")
    message = bus.send("claude", "request", {"text": "wake up"}, to="codex")
    script, args_file = _fake_codex(tmp_path)

    ring_codex(bus, message, codex_bin=str(script), resume=True)

    assert args_file.read_text(encoding="utf-8").splitlines()[-1] == "--resume"


def test_ring_reports_codex_failure(tmp_path) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "thread-1")
    message = bus.send("claude", "request", {"text": "x"}, to="codex")
    script, _args_file = _fake_codex(tmp_path, exit_code=3)

    result = ring_codex(bus, message, codex_bin=str(script))

    assert result == {
        "status": "failed",
        "thread": "thread-1",
        "returncode": 3,
        "stderr": "queue said no",
    }


def test_ring_skips_broadcasts_other_recipients_and_missing_threads(tmp_path) -> None:
    bus = _bus(tmp_path)
    script, args_file = _fake_codex(tmp_path)
    to_codex = bus.send("claude", "request", {"text": "x"}, to="codex")

    assert ring_codex(bus, to_codex, codex_bin=str(script))["reason"] == (
        "no codex thread recorded for this scope"
    )
    bus.register_session("codex", "thread-1")
    broadcast = bus.send("claude", "request", {"text": "x"}, to="all")
    to_claude = bus.send("codex", "request", {"text": "x"}, to="claude")
    assert ring_codex(bus, broadcast, codex_bin=str(script))["status"] == "skipped"
    assert ring_codex(bus, to_claude, codex_bin=str(script))["status"] == "skipped"
    assert not args_file.exists()


@pytest.mark.parametrize("msg_type", ["status", "finding", "ack", "fyi"])
def test_ring_only_for_requests_and_blockers(tmp_path, msg_type) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "thread-1")
    script, args_file = _fake_codex(tmp_path)

    quiet = ring_codex(
        bus, bus.send("claude", msg_type, {"text": "x"}, to="codex"), codex_bin=str(script)
    )
    assert quiet["status"] == "skipped"
    assert "does not ring" in quiet["reason"]
    assert not args_file.exists()

    loud = ring_codex(
        bus, bus.send("claude", "blocker", {"text": "x"}, to="codex"), codex_bin=str(script)
    )
    assert loud["status"] == "rang"


def test_ring_ignores_threads_from_other_scopes(tmp_path) -> None:
    _bus(tmp_path, "project:beta").register_session("codex", "beta-thread")
    bus = _bus(tmp_path, "project:alpha")
    message = bus.send("claude", "request", {"text": "x"}, to="codex")
    script, args_file = _fake_codex(tmp_path)

    assert ring_codex(bus, message, codex_bin=str(script))["status"] == "skipped"
    assert not args_file.exists()


def test_send_ring_cli_reports_ring_result(tmp_path, monkeypatch, capsys) -> None:
    bus = _bus(tmp_path)
    bus.register_session("codex", "thread-1")
    script, args_file = _fake_codex(tmp_path)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(friendly, "_message_bus", lambda _args: bus)

    code = cli_main(
        [
            "messages",
            "send",
            "--from",
            "claude",
            "--to",
            "codex",
            "--type",
            "request",
            "--text",
            "hi",
            "--ring",
            "--json",
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ring"]["status"] == "rang"
    assert result["ring"]["thread"] == "thread-1"
    assert args_file.read_text(encoding="utf-8").splitlines()[2] == "thread-1"


def test_sessions_cli_lists_recorded_sessions(tmp_path, monkeypatch, capsys) -> None:
    bus = _bus(tmp_path)
    bus.register_session("claude", "sess-c", cwd="/work/alpha")
    bus.register_session("codex", "thread-x", cwd="/work/alpha")
    monkeypatch.setattr(friendly, "_message_bus", lambda _args: bus)

    assert cli_main(["messages", "sessions", "--host", "codex", "--json"]) == 0

    listed = json.loads(capsys.readouterr().out)
    assert [record["session_id"] for record in listed] == ["thread-x"]
