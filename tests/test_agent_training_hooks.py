from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from afs.agent import hooks
from afs.agent.hooks import HookConfig, TrainingExportHook, TrainingSample
from afs.context_layout import audit_layout, scaffold_v2


def _sample() -> TrainingSample:
    return TrainingSample(
        input_text="Explain the scoped context layout.",
        output_text="Use the managed project scope.",
        domain="documentation",
        source="agent_harness",
        timestamp=datetime(2026, 7, 17),
    )


def test_default_training_hook_routes_v2_pool_under_runtime_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    monkeypatch.setattr(hooks, "_default_context_root", lambda: context_root)

    hook = TrainingExportHook()
    hook._append_to_pool(_sample(), 0.9)

    expected = context_root / ".afs" / "training" / "pools"
    assert hook.config.output_dir == expected
    assert (expected / "documentation_pool.jsonl").is_file()
    assert not (context_root / "training_pools").exists()
    assert audit_layout(context_root).valid is True


def test_default_training_hook_keeps_v1_training_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    monkeypatch.setattr(hooks, "_default_context_root", lambda: context_root)

    hook = TrainingExportHook()
    hook._append_to_pool(_sample(), 0.9)

    expected = context_root / "training_pools"
    assert hook.config.output_dir == expected
    assert (expected / "documentation_pool.jsonl").is_file()


def test_explicit_training_hook_output_remains_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    explicit = tmp_path / "explicit-pools"
    monkeypatch.setattr(hooks, "_default_context_root", lambda: context_root)

    hook = TrainingExportHook(HookConfig(output_dir=explicit))
    hook._append_to_pool(_sample(), 0.9)

    assert (explicit / "documentation_pool.jsonl").is_file()
    assert not (context_root / ".afs" / "training").exists()


def test_default_training_hook_rejects_linked_v2_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    training_root = context_root / ".afs" / "training"
    training_root.mkdir()
    outside = tmp_path / "outside-pools"
    outside.mkdir()
    try:
        (training_root / "pools").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(hooks, "_default_context_root", lambda: context_root)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        TrainingExportHook()

    assert list(outside.iterdir()) == []
