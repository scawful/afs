import json

from afs.training.gemini_export import export_gemini_logs_to_dataset


def test_gemini_export(tmp_path) -> None:
    root = tmp_path / ".gemini"
    session_dir = root / "tmp" / "projecthash" / "chats"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "session-2026-01-07T00-00-00000000.json"

    session_data = {
        "sessionId": "session-1",
        "projectHash": "projecthash",
        "messages": [
            {
                "id": "user-1",
                "type": "user",
                "content": "Summarize the change.",
            },
            {
                "id": "assistant-1",
                "type": "gemini",
                "content": "Summary goes here.",
                "model": "gemini-3-pro-preview",
                "toolCalls": [
                    {
                        "name": "read_file",
                        "args": {"file_path": "README.md"},
                        "status": "success",
                        "result": [
                            {
                                "functionResponse": {
                                    "response": {"output": "README content"}
                                }
                            }
                        ],
                    }
                ],
            },
        ],
    }
    session_path.write_text(json.dumps(session_data), encoding="utf-8")

    output_path = tmp_path / "out.jsonl"
    result = export_gemini_logs_to_dataset(
        [root],
        output_path,
        include_tools=True,
        require_quality=False,
    )

    assert result.exported == 1
    data = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert data["instruction"] == "Summarize the change."
    assert data["output"] == "Summary goes here."
    assert "Tool outputs" in data["input"]
    assert data["_metadata"]["gemini_session_id"] == "session-1"
