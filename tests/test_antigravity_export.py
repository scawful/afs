import json

from afs.training.antigravity_export import _extract_json_objects, _payload_to_sample


def test_antigravity_payload_to_sample() -> None:
    task = {
        "Mode": "VERIFICATION",
        "PredictedTaskSize": 2,
        "TaskName": "Implement Backend API Catalog",
        "TaskStatus": "Notifying user of fix",
        "TaskSummary": "Added /admin/api-catalog endpoint and tests.",
    }
    response = {
        "BlockedOnUser": False,
        "ConfidenceJustification": "Build verified.",
        "ConfidenceScore": 1,
        "Message": "API catalog endpoint implemented and tested.",
        "PathsToReview": ["/tmp/does_not_exist.py"],
    }
    payload = (
        b"\x08\x01"
        + json.dumps(task).encode("utf-8")
        + b"\x00\xff"
        + json.dumps(response).encode("utf-8")
    )

    objects = _extract_json_objects(payload)
    assert any(obj.get("TaskName") == task["TaskName"] for obj in objects)
    assert any(obj.get("Message") == response["Message"] for obj in objects)

    sample = _payload_to_sample(
        payload,
        default_domain="general",
        include_paths_content=False,
        max_path_chars=2000,
    )
    assert sample is not None
    assert sample.instruction == task["TaskName"]
    assert sample.output == response["Message"]
    assert "Task summary" in sample.input
    assert sample._metadata["antigravity_task_name"] == task["TaskName"]
