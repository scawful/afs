from __future__ import annotations

import json

from afs.training.memory_export import export_memory_to_dataset


def test_memory_export_quality_filter(tmp_path) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()

    entries = [
        {"instruction": "Short", "output": "tiny"},
        {"instruction": "Long", "output": "x" * 100},
    ]
    (memory_root / "entries.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )

    output_path = tmp_path / "out.jsonl"
    result = export_memory_to_dataset(
        memory_root,
        output_path,
        require_quality=True,
        min_quality_score=0.5,
        score_profile="generic",
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert result.filtered == 1


def test_allow_raw_tags_gate(tmp_path) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "note.md").write_text("raw content", encoding="utf-8")

    output_path = tmp_path / "out.jsonl"
    result = export_memory_to_dataset(
        memory_root,
        output_path,
        allow_raw=True,
        allow_raw_tags=["approved"],
        require_quality=False,
    )

    assert result.exported == 0
    assert output_path.read_text(encoding="utf-8") == ""
