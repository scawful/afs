from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from afs.embeddings import EmbeddingCollectionMetadata
from afs.hybrid_search import HybridSearchEngine, HybridSource, SourcePolicy


def _semantic_vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "sprite" in lowered else 0.0,
        1.0 if "release" in lowered else 0.0,
        0.25,
    ]


def test_scope_filter_is_applied_before_all_ranking(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "guide.md").write_text("sprite alpha guide", encoding="utf-8")
    (beta / "secret.md").write_text("sprite beta secret", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    engine.build(
        [
            HybridSource(alpha, scope_id="project:alpha", project_id="alpha", embed_allowed=True),
            HybridSource(beta, scope_id="project:beta", project_id="beta", embed_allowed=True),
        ],
        embed_fn=_semantic_vector,
    )

    response = engine.search(
        "sprite",
        scope_ids=["project:alpha"],
        include_common=False,
        query_embed_fn=lambda _: [1.0, 0.0, 0.25],
    )

    assert response.semantic_status == "ready"
    assert response.results
    assert {hit.scope_id for hit in response.results} == {"project:alpha"}
    assert all("beta" not in hit.source_path for hit in response.results)


def test_rrf_ranking_is_deterministic_and_reports_signal_provenance(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "renderer.md").write_text(
        "# Renderer\nclass SpriteRenderer:\nrelease sprite pipeline", encoding="utf-8"
    )
    (source / "notes.md").write_text("sprite notes", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    engine.build(
        [
            HybridSource(
                source,
                scope_id="project:alpha",
                project_id="alpha",
                project_terms=("game", "renderer"),
                embed_allowed=True,
            )
        ],
        embed_fn=_semantic_vector,
    )

    kwargs = {
        "scope_ids": ["project:alpha"],
        "include_common": False,
        "query_embed_fn": lambda _: [1.0, 0.0, 0.25],
    }
    first = engine.search("SpriteRenderer", **kwargs)
    second = engine.search("SpriteRenderer", **kwargs)

    assert [hit.to_dict() for hit in first.results] == [hit.to_dict() for hit in second.results]
    assert first.results[0].relative_path == "renderer.md"
    assert {"exact", "symbol"}.issubset(first.results[0].signals)
    assert first.results[0].score == pytest.approx(sum(
        1.0 / (60 + int(signal["rank"])) for signal in first.results[0].signals.values()
    ))


def test_semantic_dimension_mismatch_falls_back_without_semantic_signal(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "release.md").write_text("release checklist", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    engine.build(
        [HybridSource(source, scope_id="project:alpha", embed_allowed=True)],
        embed_fn=_semantic_vector,
    )

    response = engine.search(
        "release",
        scope_ids=["project:alpha"],
        include_common=False,
        mode="semantic",
        query_embed_fn=lambda _: [1.0, 0.0],
    )

    assert response.semantic_status == "dimension_mismatch"
    assert response.results
    assert all("semantic" not in hit.signals for hit in response.results)
    assert any("text" in hit.signals for hit in response.results)


def test_partial_embedding_failure_is_unhealthy_and_never_partially_ranked(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "good.md").write_text("release good", encoding="utf-8")
    (source / "bad.md").write_text("release bad", encoding="utf-8")

    def partial(text: str) -> list[float]:
        if "bad" in text:
            raise RuntimeError("provider failure")
        return [0.0, 1.0, 0.25]

    engine = HybridSearchEngine(tmp_path / "index")
    result = engine.build(
        [HybridSource(source, scope_id="project:alpha", embed_allowed=True)],
        embed_fn=partial,
    )
    response = engine.search(
        "release",
        scope_ids=["project:alpha"],
        include_common=False,
        query_embed_fn=lambda _: [0.0, 1.0, 0.25],
    )

    assert result.semantic_status == "unhealthy"
    assert response.semantic_status == "unhealthy"
    assert len(response.results) == 2
    assert all("semantic" not in hit.signals for hit in response.results)


def test_safe_source_policy_denies_secrets_gitignored_and_escaping_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / ".gitignore").write_text("ignored.md\n", encoding="utf-8")
    (source / ".afsignore").write_text("afs-ignored.md\n", encoding="utf-8")
    (source / "safe.md").write_text("safe searchable notes", encoding="utf-8")
    (source / "ignored.md").write_text("ignored material", encoding="utf-8")
    (source / "afs-ignored.md").write_text("afs ignored material", encoding="utf-8")
    (source / ".env").write_text("API_KEY=not-for-indexing", encoding="utf-8")
    (source / "private.md").write_text("password=not-for-indexing", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside material", encoding="utf-8")
    os.symlink(outside, source / "escape.md")
    embed_calls: list[str] = []

    engine = HybridSearchEngine(tmp_path / "index")
    result = engine.build(
        [
            HybridSource(
                source,
                scope_id="project:alpha",
                embed_allowed=False,
                policy=SourcePolicy(),
            )
        ],
        embed_fn=lambda text: embed_calls.append(text) or [1.0, 0.0],
    )
    response = engine.search(
        "material API_KEY password searchable",
        scope_ids=["project:alpha"],
        include_common=False,
        mode="text",
    )

    assert embed_calls == []
    assert result.indexed_files == 1
    assert [hit.relative_path for hit in response.results] == ["safe.md"]


def test_hybrid_layout_uses_normalized_float32_vectors_and_is_discoverable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "guide.md").write_text("sprite guide", encoding="utf-8")
    index_root = tmp_path / ".afs" / "search"
    engine = HybridSearchEngine(index_root)
    collection = EmbeddingCollectionMetadata(
        provider="custom",
        model="fake",
        dimension=3,
        document_instruction="document",
        query_instruction="query",
    )
    result = engine.build(
        [HybridSource(source, scope_id="project:alpha", embed_allowed=True)],
        embed_fn=lambda _: [3.0, 4.0, 0.0],
        collection=collection,
    )

    matrix = np.load(engine.vector_path, allow_pickle=False)
    metadata = json.loads(engine.metadata_path.read_text(encoding="utf-8"))
    assert result.healthy
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert metadata["collection"]["dimension"] == 3
    assert metadata["collection"]["normalized"] is True
    assert HybridSearchEngine.discover(tmp_path) == [index_root]


def test_inactive_source_file_cap_is_enforced_before_unbounded_crawl(tmp_path: Path) -> None:
    source = tmp_path / "inactive"
    source.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (source / name).write_text(f"content {name}", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")

    result = engine.build(
        [
            HybridSource(
                source,
                scope_id="project:inactive",
                active=False,
                policy=SourcePolicy(max_files_inactive=1),
            )
        ]
    )
    response = engine.search(
        "content",
        scope_ids=["project:inactive"],
        include_common=False,
        mode="text",
    )

    assert result.indexed_files == 1
    assert result.capped_sources == [f"project:inactive:{source}"]
    assert [hit.relative_path for hit in response.results] == ["a.md"]
