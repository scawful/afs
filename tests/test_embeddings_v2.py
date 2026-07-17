from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

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

    result = build_embedding_index([source], tmp_path / "index", embed_fn=embed, incremental=True)

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
    payloads = list((output / "embeddings").rglob("*.json"))
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


def test_blank_query_is_rejected_before_embedder_invocation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("guide", encoding="utf-8")
    output = tmp_path / "index"
    build_embedding_index([source], output)
    calls: list[str] = []

    with pytest.raises(ValueError, match="query cannot be empty"):
        search_embedding_index_detailed(
            output,
            "   ",
            embed_fn=lambda text: calls.append(text) or [1.0],
        )

    assert calls == []


def test_output_inside_source_is_never_self_indexed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("stable source", encoding="utf-8")
    output = source / "generated-index"

    first = build_embedding_index([source], output, include_hidden=True)
    second = build_embedding_index([source], output, include_hidden=True)

    assert first.total_files == 1
    assert second.total_files == 1
    manifest = json.loads((output / "embedding_index.json").read_text(encoding="utf-8"))
    entries = {key: value for key, value in manifest.items() if key != "_metadata"}
    assert len(entries) == 1
    payload = json.loads(
        (output / "embeddings" / next(iter(entries.values()))).read_text(encoding="utf-8")
    )
    assert payload["source_path"] == str(source / "guide.md")


def test_output_parent_does_not_hide_registered_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("visible source", encoding="utf-8")

    result = build_embedding_index([source], tmp_path)

    assert result.total_files == 1
    assert result.indexed == 1


def test_manifest_publication_failure_preserves_previous_generation(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    document = source / "guide.md"
    document.write_text("old marker", encoding="utf-8")
    output = tmp_path / "index"
    build_embedding_index([source], output)
    manifest_path = output / "embedding_index.json"
    old_manifest = manifest_path.read_text(encoding="utf-8")
    document.write_text("new marker", encoding="utf-8")

    import afs.embeddings as embeddings_module

    real_atomic_write = embeddings_module.atomic_write_text

    def fail_manifest(path: Path, text: str) -> None:
        if path == manifest_path:
            raise RuntimeError("injected publication crash")
        real_atomic_write(path, text)

    monkeypatch.setattr(embeddings_module, "atomic_write_text", fail_manifest)
    with pytest.raises(RuntimeError, match="injected publication crash"):
        build_embedding_index([source], output)

    assert manifest_path.read_text(encoding="utf-8") == old_manifest
    response = search_embedding_index_detailed(output, "old", min_score=0.0)
    assert response.results[0].text_preview == "old marker"


def test_reader_holds_legacy_generation_until_search_completes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    document = source / "guide.md"
    document.write_text("old marker", encoding="utf-8")
    output = tmp_path / "index"
    build_embedding_index([source], output, embed_fn=lambda _: [1.0, 0.0])

    entered = threading.Event()
    release = threading.Event()
    search_results = []
    failures: list[BaseException] = []

    def query_embed(_: str) -> list[float]:
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release query embedder")
        return [1.0, 0.0]

    def run_search() -> None:
        try:
            search_results.append(
                search_embedding_index_detailed(output, "old", embed_fn=query_embed, min_score=0.0)
            )
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            failures.append(exc)

    def run_build() -> None:
        try:
            build_embedding_index([source], output, embed_fn=lambda _: [1.0, 0.0])
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            failures.append(exc)

    search_thread = threading.Thread(target=run_search)
    search_thread.start()
    assert entered.wait(5)
    document.write_text("new marker", encoding="utf-8")
    build_thread = threading.Thread(target=run_build)
    build_thread.start()
    time.sleep(0.1)
    assert build_thread.is_alive()
    release.set()
    search_thread.join(5)
    build_thread.join(5)

    assert failures == []
    assert not search_thread.is_alive()
    assert not build_thread.is_alive()
    assert search_results[0].results[0].text_preview == "old marker"
    current = search_embedding_index_detailed(output, "new", min_score=0.0)
    assert current.results[0].text_preview == "new marker"


def test_legacy_builders_are_serialized(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("serialized", encoding="utf-8")
    output = tmp_path / "index"
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    failures: list[BaseException] = []

    def slow_embed(_: str) -> list[float]:
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return [1.0, 0.0]

    def run_build() -> None:
        try:
            build_embedding_index([source], output, embed_fn=slow_embed)
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [threading.Thread(target=run_build) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert failures == []
    assert maximum_active == 1
    assert all(not thread.is_alive() for thread in threads)
