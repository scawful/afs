import json

from afs.training.rebalance import rebalance_dataset


def _write_samples(path, samples):
    path.write_text("\n".join(json.dumps(item) for item in samples), encoding="utf-8")


def test_rebalance_by_source(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    samples = (
        [{"instruction": "a", "output": "1", "source": "claude"} for _ in range(4)]
        + [{"instruction": "b", "output": "2", "source": "gemini"} for _ in range(2)]
        + [{"instruction": "c", "output": "3", "source": "codex"} for _ in range(1)]
    )
    _write_samples(input_path, samples)

    output_path = tmp_path / "out.jsonl"
    result = rebalance_dataset(
        [input_path],
        output_path,
        group_by="source",
        weights={"claude": 0.5, "gemini": 0.3, "codex": 0.2},
        allow_oversample=False,
        seed=1,
    )

    assert result.total_output == 5
    assert result.group_counts_out["codex"] == 1
    assert "claude" in result.group_counts_out
    assert "gemini" in result.group_counts_out
