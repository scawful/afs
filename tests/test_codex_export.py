import json

from afs.training.codex_export import export_codex_logs_to_dataset


def test_codex_export(tmp_path) -> None:
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
            "type": "turn_context",
            "payload": {"model": "gpt-5.2-codex", "cwd": "/tmp"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Do it."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec_command",
                "input": "{\"cmd\": \"ls\"}",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "{\"output\":\"ok\",\"metadata\":{\"exit_code\":0}}",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    output_path = tmp_path / "out.jsonl"
    result = export_codex_logs_to_dataset(
        [root],
        output_path,
        include_tools=True,
        include_system=True,
        require_quality=False,
        redact=False,
    )

    assert result.exported == 1
    data = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert data["instruction"] == "Do it."
    assert data["output"] == "Done."
    assert "Tool outputs" in data["input"]
    assert "System:" in data["input"]
