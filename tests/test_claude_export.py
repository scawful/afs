import json

from afs.training.claude_export import export_claude_logs_to_dataset


def test_claude_export(tmp_path) -> None:
    root = tmp_path / ".claude"
    log_dir = root / "projects" / "proj"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "agent-test.jsonl"

    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Do the thing."},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_test",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_test",
                        "content": "output",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "model": "claude-sonnet-4",
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    output_path = tmp_path / "out.jsonl"
    result = export_claude_logs_to_dataset(
        [root],
        output_path,
        include_tools=True,
        require_quality=False,
        redact=False,
    )

    assert result.exported == 1
    data = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert data["instruction"] == "Do the thing."
    assert data["output"] == "Done."
    assert "Tool outputs" in data["input"]
