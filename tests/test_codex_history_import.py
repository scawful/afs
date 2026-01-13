import json

from afs.history import iter_history_events
from afs.training.codex_export import import_codex_logs_to_history


def test_codex_history_import(tmp_path) -> None:
    root = tmp_path / ".codex"
    log_dir = root / "sessions" / "2026" / "01" / "07"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "rollout-2026-01-07T10-00-00-test.jsonl"

    lines = [
        {
            "type": "session_meta",
            "payload": {
                "id": "sess-1",
                "cli_version": "0.1",
                "originator": "codex_cli",
                "instructions": "System rules.",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-07T10:00:00Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Do it."}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-07T10:00:01Z",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec_command",
                "input": "{\"cmd\": \"ls\"}",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-07T10:00:02Z",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "{\"output\":\"ok\",\"metadata\":{\"exit_code\":0}}",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-07T10:00:03Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    history_root = tmp_path / "history"
    result = import_codex_logs_to_history(
        [root],
        history_root=history_root,
        include_system=True,
        redact=False,
    )

    assert result.imported_model == 1
    assert result.imported_tools == 1

    events = list(iter_history_events(history_root, include_payloads=True))
    event_types = {event.get("type") for event in events}
    assert "model" in event_types
    assert "tool" in event_types

    model_event = next(event for event in events if event.get("type") == "model")
    payload = model_event.get("payload") or {}
    assert payload.get("prompt") == "Do it."
    assert payload.get("response") == "Done."
    assert payload.get("system") == "System rules."
