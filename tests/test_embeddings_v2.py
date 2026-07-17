from __future__ import annotations

import json
from pathlib import Path

from afs.embeddings import (
    DEFAULT_GEMINI_DIMENSION,
    DEFAULT_GEMINI_MODEL,
    EMBEDDING_INDEX_VERSION,
    GEMINI_DOCUMENT_TASK,
    build_embedding_index,
    create_embed_fn,
    create_query_embed_fn_from_index,
    discover_embedding_indexes,
    register_embedding_backend,
    search_embedding_index_detailed,
)


def _register_fake_provider(calls: list[dict] | None = None) -> str:
    provider = "test-v2"

    def factory(
        model: str,
        task_type: str = GEMINI_DOCUMENT_TASK,
        dimension: int = 3,
        **kwargs,
    ):
        if calls is not None:
            calls.append(
                {
                    "model": model,
                    "task_type": task_type,
                    "dimension": dimension,
                    **kwargs,
                }
            )

        def embed(text: str) -> list[float]:
            return [float(len(text) > index) for index in range(dimension)]

        return embed

    register_embedding_backend(provider, factory)
    return provider


def test_versioned_collection_records_provider_model_dimension_and_instructions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("alpha beta gamma", encoding="utf-8")
    provider = _register_fake_provider()
    embed = create_embed_fn(
        provider,
        model="fake-2026",
        dimension=3,
        task_type=GEMINI_DOCUMENT_TASK,
    )

    result = build_embedding_index([source], tmp_path / "index", embed_fn=embed)

    payload = json.loads((tmp_path / "index" / "embedding_index.json").read_text())
    collection = payload["_metadata"]["collection"]
    assert result.errors == []
    assert collection == {
        "version": EMBEDDING_INDEX_VERSION,
        "provider": provider,
        "model": "fake-2026",
        "dimension": 3,
        "document_instruction": GEMINI_DOCUMENT_TASK,
        "query_instruction": GEMINI_DOCUMENT_TASK,
        "normalized": False,
        "health": "healthy",
    }


def test_query_embedder_is_recreated_from_collection_contract(tmp_path: Path) -> None:
    calls: list[dict] = []
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("retrieval guide", encoding="utf-8")
    provider = _register_fake_provider(calls)
    document_embed = create_embed_fn(
        provider,
        model="fake-2026",
        dimension=3,
        task_type=GEMINI_DOCUMENT_TASK,
    )
    build_embedding_index([source], tmp_path / "index", embed_fn=document_embed)
    index_path = tmp_path / "index" / "embedding_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["_metadata"]["collection"]["query_instruction"] = "QUERY_TASK"
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    calls.clear()

    query_embed = create_query_embed_fn_from_index(tmp_path / "index")
    assert len(query_embed("find guide")) == 3
    assert calls == [
        {
            "model": "fake-2026",
            "task_type": "QUERY_TASK",
            "dimension": 3,
        }
    ]


def test_dimension_mismatch_marks_collection_unhealthy_and_falls_back(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.md").write_text("alpha keyword", encoding="utf-8")
    (source / "b.md").write_text("beta keyword", encoding="utf-8")
    calls = 0

    def inconsistent(_: str) -> list[float]:
        nonlocal calls
        calls += 1
        return [1.0, 0.0] if calls == 1 else [1.0, 0.0, 0.0]

    build_embedding_index([source], tmp_path / "index", embed_fn=inconsistent)
    response = search_embedding_index_detailed(
        tmp_path / "index", "keyword", embed_fn=lambda _: [1.0, 0.0], min_score=0.0
    )

    assert response.collection is not None
    assert response.collection.health == "unhealthy"
    assert response.semantic_status == "fallback"
    assert "unhealthy" in (response.semantic_reason or "")
    assert len(response.results) == 2


def test_incremental_reuse_preserves_healthy_collection_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("alpha keyword", encoding="utf-8")
    provider = _register_fake_provider()
    embed = create_embed_fn(provider, model="fake-2026", dimension=3)
    build_embedding_index([source], tmp_path / "index", embed_fn=embed)

    result = build_embedding_index(
        [source], tmp_path / "index", embed_fn=embed, incremental=True
    )

    assert result.reused == 1
    assert result.semantic_status == "healthy"


def test_full_rebuild_garbage_collects_orphan_payloads(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.md"
    first.write_text("first", encoding="utf-8")
    output = tmp_path / "index"
    build_embedding_index([source], output)
    orphan = output / "embeddings" / "orphan.json"
    orphan.write_text("{}", encoding="utf-8")
    first.unlink()
    (source / "second.md").write_text("second", encoding="utf-8")

    result = build_embedding_index([source], output)

    assert result.orphans_removed >= 2
    assert not orphan.exists()
    payloads = list((output / "embeddings").glob("*.json"))
    assert len(payloads) == 1


def test_embedding_index_discovery_is_bounded_and_documented(tmp_path: Path) -> None:
    direct = tmp_path / ".afs" / "search"
    project = tmp_path / "knowledge" / "alpha"
    too_deep = tmp_path / "knowledge" / "alpha" / "nested"
    for root in (direct, project, too_deep):
        root.mkdir(parents=True, exist_ok=True)
        (root / "embedding_index.json").write_text("{}", encoding="utf-8")

    assert discover_embedding_indexes(tmp_path) == sorted([direct, project], key=str)


def test_stable_gemini_defaults_are_current() -> None:
    assert DEFAULT_GEMINI_MODEL == "gemini-embedding-2"
    assert DEFAULT_GEMINI_DIMENSION == 768
