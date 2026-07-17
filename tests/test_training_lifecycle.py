from __future__ import annotations

from pathlib import Path

import pytest

from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.schema import AFSConfig, GeneralConfig
from afs.training.lifecycle import (
    dataset_artifact_root,
    run_root,
    write_dataset_artifacts,
)


def _manager(context: Path) -> AFSManager:
    return AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context)))


def test_v2_training_roots_use_common_scratchpad(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    manager = _manager(context)

    dataset_root = dataset_artifact_root(manager, context, "dataset-one")
    training_run = run_root(manager, context, "20260717T120000Z-run-one")

    assert dataset_root == (
        context / "scratchpad" / "common" / "training" / "datasets" / "dataset-one"
    )
    assert training_run == (
        context
        / "scratchpad"
        / "common"
        / "training"
        / "runs"
        / "20260717T120000Z-run-one"
    )
    write_dataset_artifacts(dataset_root, manifest={"dataset_id": "dataset-one"})
    assert (dataset_root / "manifest.json").is_file()


def test_v2_training_rejects_linked_output_root(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    outside = tmp_path / "outside"
    outside.mkdir()
    training_root = context / "scratchpad" / "common" / "training"
    training_root.parent.mkdir(parents=True)
    try:
        training_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        dataset_artifact_root(_manager(context), context, "dataset-one")

    assert list(outside.iterdir()) == []


def test_v2_training_rejects_linked_manifest_leaf(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    artifact_root = dataset_artifact_root(
        _manager(context),
        context,
        "dataset-one",
    )
    artifact_root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    try:
        (artifact_root / "manifest.json").symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        write_dataset_artifacts(artifact_root, manifest={"dataset_id": "dataset-one"})

    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_v1_training_paths_keep_legacy_shape(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    (context / "scratchpad").mkdir(parents=True)
    manager = _manager(context)

    assert dataset_artifact_root(manager, context, "dataset-one") == (
        context / "scratchpad" / "training" / "datasets" / "dataset-one"
    )
    assert run_root(manager, context, "run-one") == (
        context / "scratchpad" / "training" / "runs" / "run-one"
    )
