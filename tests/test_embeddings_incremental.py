"""Tests for incremental re-indexing (Feature 2)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from afs.embeddings import build_embedding_index


def test_full_build_writes_metadata(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("hello world", encoding="utf-8")
    output = tmp_path / "index_out"

    result = build_embedding_index([source], output)
    assert result.total_files == 1
    assert result.indexed == 1
    assert result.mode == "full"

    index_data = json.loads((output / "embedding_index.json").read_text(encoding="utf-8"))
    assert "_metadata" in index_data
    assert index_data["_metadata"]["mode"] == "full"


def test_incremental_reuses_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("hello world", encoding="utf-8")
    (source / "b.md").write_text("other content", encoding="utf-8")
    output = tmp_path / "index_out"

    # Full build
    result1 = build_embedding_index([source], output)
    assert result1.indexed == 2
    assert result1.reused == 0

    # Incremental build with no changes
    result2 = build_embedding_index([source], output, incremental=True)
    assert result2.total_files == 2
    assert result2.reused == 2
    assert result2.indexed == 0
    assert result2.mode == "incremental"


def test_incremental_re_embeds_changed(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("original content", encoding="utf-8")
    output = tmp_path / "index_out"

    build_embedding_index([source], output)

    # Modify the file (change size to ensure detection)
    time.sleep(0.05)
    (source / "a.md").write_text("modified content that is longer now", encoding="utf-8")

    result = build_embedding_index([source], output, incremental=True)
    assert result.indexed == 1
    assert result.reused == 0


def test_incremental_detects_deletions(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("file a", encoding="utf-8")
    (source / "b.md").write_text("file b", encoding="utf-8")
    output = tmp_path / "index_out"

    build_embedding_index([source], output)

    # Delete one file
    (source / "b.md").unlink()

    result = build_embedding_index([source], output, incremental=True)
    assert result.total_files == 1
    assert result.removed == 1


def test_search_skips_metadata_key(tmp_path: Path) -> None:
    from afs.embeddings import search_embedding_index

    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.md").write_text("hello search test", encoding="utf-8")
    output = tmp_path / "index_out"

    build_embedding_index([source], output)

    # Verify _metadata key doesn't break search
    results = search_embedding_index(output, "hello", min_score=0.0)
    assert len(results) >= 1
    for r in results:
        assert r.doc_id != "_metadata"


def test_embedding_index_result_summary_incremental() -> None:
    from afs.embeddings import EmbeddingIndexResult

    result = EmbeddingIndexResult(
        total_files=10, indexed=2, skipped=1, reused=7, removed=0, mode="incremental"
    )
    summary = result.summary()
    assert "reused=7" in summary
    assert "removed=0" in summary
