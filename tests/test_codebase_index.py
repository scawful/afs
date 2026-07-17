from __future__ import annotations

from pathlib import Path

import pytest

from afs.codebase_index import build_codebase_index, codebase_index_dir
from afs.context_layout import scaffold_v2


def test_v2_codebase_index_uses_common_scratchpad(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    (project / "sample.py").write_text("def sample():\n    return 1\n", encoding="utf-8")
    scaffold_v2(context)

    output = codebase_index_dir(context)
    result = build_codebase_index(project, output)

    assert output == context / "scratchpad" / "common" / "codebase"
    assert result.indexed == 1
    assert (output / "index.json").is_file()
    assert not (context / "scratchpad" / "codebase").exists()


def test_v2_codebase_index_rejects_linked_manifest(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    (project / "sample.py").write_text("def sample(): pass\n", encoding="utf-8")
    scaffold_v2(context)
    output = codebase_index_dir(context)
    output.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    try:
        (output / "index.json").symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        build_codebase_index(project, output)

    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_codebase_index_preserves_v1_and_explicit_output_paths(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    (context / "scratchpad").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    (project / "sample.py").write_text("def sample(): pass\n", encoding="utf-8")
    explicit = tmp_path / "trusted-output"

    assert codebase_index_dir(context) == context / "scratchpad" / "codebase"
    assert build_codebase_index(project, explicit).indexed == 1
    assert (explicit / "index.json").is_file()
