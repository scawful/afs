from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from afs.context_layout import scaffold_v2
from afs.embeddings import EmbeddingCollectionMetadata
from afs.hybrid_search import (
    HybridScopeCoverageError,
    HybridSearchEngine,
    HybridSearchResponse,
    HybridSource,
    SourcePolicy,
)


def _semantic_vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "sprite" in lowered else 0.0,
        1.0 if "release" in lowered else 0.0,
        0.25,
    ]


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")


def test_v2_system_search_root_rejects_link_escape(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    outside = tmp_path / "outside"
    scaffold_v2(context_root)
    outside.mkdir()
    search_root = context_root / ".afs" / "search"
    search_root.rmdir()
    search_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        HybridSearchEngine(search_root)
    assert not any(outside.iterdir())


def test_v2_hybrid_rejects_linked_current_for_search_and_rebuild(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / ".context"
    source = tmp_path / "project"
    scaffold_v2(context_root)
    source.mkdir()
    (source / "guide.md").write_text("sprite guide", encoding="utf-8")
    engine = HybridSearchEngine(context_root / ".afs" / "search")
    registered = HybridSource(source, scope_id="project:alpha")
    engine.build([registered])
    current = engine.current_path
    outside_current = tmp_path / "outside-current"
    outside_payload = current.read_text(encoding="utf-8")
    outside_current.write_text(outside_payload, encoding="utf-8")
    current.unlink()
    _symlink_or_skip(current, outside_current)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        engine.search(
            "sprite",
            scope_ids=["project:alpha"],
            include_common=False,
            mode="text",
        )
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        engine.build([registered])

    assert outside_current.read_text(encoding="utf-8") == outside_payload


def test_v2_hybrid_rejects_linked_active_generation_directory(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / ".context"
    source = tmp_path / "project"
    scaffold_v2(context_root)
    source.mkdir()
    (source / "guide.md").write_text("sprite guide", encoding="utf-8")
    engine = HybridSearchEngine(context_root / ".afs" / "search")
    engine.build([HybridSource(source, scope_id="project:alpha")])
    generation_id = engine.current_path.read_text(encoding="utf-8").strip()
    generation = engine.generations_root / generation_id
    outside_generation = tmp_path / "outside-generation"
    generation.rename(outside_generation)
    _symlink_or_skip(generation, outside_generation, directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        engine.search(
            "sprite",
            scope_ids=["project:alpha"],
            include_common=False,
            mode="text",
        )


def test_v2_hybrid_rejects_linked_generations_root_before_rebuild(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / ".context"
    source = tmp_path / "project"
    scaffold_v2(context_root)
    source.mkdir()
    (source / "guide.md").write_text("sprite guide", encoding="utf-8")
    engine = HybridSearchEngine(context_root / ".afs" / "search")
    outside = tmp_path / "outside-generations"
    outside.mkdir()
    _symlink_or_skip(engine.index_root / "generations", outside, directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        engine.build([HybridSource(source, scope_id="project:alpha")])

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("filename", "mode"),
    [
        ("search.sqlite3", "text"),
        ("vectors.npy", "semantic"),
        ("hybrid_index.json", "text"),
    ],
)
def test_v2_hybrid_rejects_linked_active_generation_leaf(
    tmp_path: Path,
    filename: str,
    mode: str,
) -> None:
    context_root = tmp_path / ".context"
    source = tmp_path / "project"
    scaffold_v2(context_root)
    source.mkdir()
    (source / "guide.md").write_text("sprite guide", encoding="utf-8")
    engine = HybridSearchEngine(context_root / ".afs" / "search")
    engine.build(
        [
            HybridSource(
                source,
                scope_id="project:alpha",
                embed_allowed=True,
            )
        ],
        embed_fn=_semantic_vector,
    )
    generation_id = engine.current_path.read_text(encoding="utf-8").strip()
    leaf = engine.generations_root / generation_id / filename
    outside_leaf = tmp_path / f"outside-{filename}"
    leaf.rename(outside_leaf)
    _symlink_or_skip(leaf, outside_leaf)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        engine.search(
            "sprite",
            scope_ids=["project:alpha"],
            include_common=False,
            mode=mode,
            query_embed_fn=lambda _: [1.0, 0.0, 0.25],
        )

    assert outside_leaf.exists()


@pytest.mark.parametrize(
    ("lock_name", "operation"),
    [
        (".hybrid-build.lock", "build"),
        (".hybrid-publication.lock", "search"),
    ],
)
def test_v2_hybrid_rejects_linked_lock_leaf(
    tmp_path: Path,
    lock_name: str,
    operation: str,
) -> None:
    context_root = tmp_path / ".context"
    source = tmp_path / "project"
    scaffold_v2(context_root)
    source.mkdir()
    (source / "guide.md").write_text("sprite guide", encoding="utf-8")
    engine = HybridSearchEngine(context_root / ".afs" / "search")
    registered = HybridSource(source, scope_id="project:alpha")
    engine.build([registered])
    lock_path = engine.index_root / lock_name
    lock_path.unlink()
    outside_lock = tmp_path / f"outside-{lock_name}"
    outside_lock.write_bytes(b"outside")
    _symlink_or_skip(lock_path, outside_lock)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        if operation == "build":
            engine.build([registered])
        else:
            engine.search(
                "sprite",
                scope_ids=["project:alpha"],
                include_common=False,
                mode="text",
            )

    assert outside_lock.read_bytes() == b"outside"


def test_legacy_flat_hybrid_index_without_current_remains_searchable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "guide.md").write_text("legacy sprite guide", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "legacy-index")
    engine.build([HybridSource(source, scope_id="project:alpha")])
    generation_id = engine.current_path.read_text(encoding="utf-8").strip()
    generation = engine.generations_root / generation_id
    for filename in ("search.sqlite3", "vectors.npy", "hybrid_index.json"):
        shutil.copy2(generation / filename, engine.index_root / filename)
    engine.current_path.unlink()

    response = engine.search(
        "legacy sprite",
        scope_ids=["project:alpha"],
        include_common=False,
        mode="text",
    )

    assert [hit.relative_path for hit in response.results] == ["guide.md"]


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
    assert first.results[0].score == pytest.approx(
        sum(1.0 / (60 + int(signal["rank"])) for signal in first.results[0].signals.values())
    )


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
    assert metadata["scope_coverage"] == {
        "embedded_scope_ids": ["project:alpha"],
        "intended_scope_ids": ["project:alpha"],
        "source_scope_ids": ["project:alpha"],
    }
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


def test_high_override_cannot_bypass_aggregate_project_cap(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        for index in range(2):
            (root / f"{index}.md").write_text("bounded content", encoding="utf-8")
    policy = SourcePolicy(max_files_inactive=2)
    engine = HybridSearchEngine(tmp_path / "index")

    result = engine.build(
        [
            HybridSource(
                first,
                scope_id="project:inactive",
                project_id="inactive",
                active=False,
                max_files=100,
                policy=policy,
            ),
            HybridSource(
                second,
                scope_id="project:inactive",
                project_id="inactive",
                active=False,
                max_files=100,
                policy=policy,
            ),
        ]
    )

    assert result.indexed_files == 2
    assert result.capped_sources


def test_conflicting_source_policies_use_project_cap_before_crawl(tmp_path: Path) -> None:
    active = tmp_path / "a-active"
    inactive = tmp_path / "z-inactive"
    active.mkdir()
    inactive.mkdir()
    for root in (active, inactive):
        for index in range(3):
            (root / f"{index}.md").write_text("bounded content", encoding="utf-8")
    policy = SourcePolicy(max_files_active=3, max_files_inactive=1)

    result = HybridSearchEngine(tmp_path / "index").build(
        [
            HybridSource(
                active,
                scope_id="project:shared",
                project_id="shared",
                active=True,
                policy=policy,
            ),
            HybridSource(
                inactive,
                scope_id="project:shared",
                project_id="shared",
                active=False,
                policy=policy,
            ),
        ]
    )

    assert result.indexed_files == 1
    assert f"project:shared:{active}" in result.capped_sources


def test_blank_query_is_rejected_before_embedder_invocation(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "guide.md").write_text("guide", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    engine.build([HybridSource(source, scope_id="project:alpha")])
    calls: list[str] = []

    with pytest.raises(ValueError, match="query cannot be empty"):
        engine.search(
            "   ",
            scope_ids=["project:alpha"],
            query_embed_fn=lambda text: calls.append(text) or [1.0],
        )

    assert calls == []


@pytest.mark.parametrize("index_relative", [Path(".afs/search"), Path("generated-index")])
def test_nested_index_is_never_self_indexed(tmp_path: Path, index_relative: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "guide.md").write_text("stable source", encoding="utf-8")
    engine = HybridSearchEngine(source / index_relative)
    registered = HybridSource(
        source,
        scope_id="project:alpha",
        policy=SourcePolicy(allow_hidden=True),
    )

    first = engine.build([registered])
    second = engine.build([registered])

    assert first.indexed_files == 1
    assert second.indexed_files == 1
    response = engine.search(
        "stable", scope_ids=["project:alpha"], include_common=False, mode="text"
    )
    assert [hit.relative_path for hit in response.results] == ["guide.md"]


def test_index_parent_does_not_hide_registered_source(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "guide.md").write_text("visible source", encoding="utf-8")

    result = HybridSearchEngine(tmp_path).build([HybridSource(source, scope_id="project:alpha")])

    assert result.indexed_files == 1


def test_publication_failure_leaves_previous_generation_searchable(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    document = source / "guide.md"
    document.write_text("old marker", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    registered = HybridSource(source, scope_id="project:alpha")
    engine.build([registered])
    old_generation = engine.current_path.read_text(encoding="utf-8")
    document.write_text("new marker", encoding="utf-8")

    import afs.hybrid_search as hybrid_module

    real_atomic_write = hybrid_module.atomic_write_text

    def fail_current(path: Path, text: str) -> None:
        if path.name == "CURRENT":
            raise RuntimeError("injected publication crash")
        real_atomic_write(path, text)

    monkeypatch.setattr(hybrid_module, "atomic_write_text", fail_current)
    with pytest.raises(RuntimeError, match="injected publication crash"):
        engine.build([registered])

    assert engine.current_path.read_text(encoding="utf-8") == old_generation
    response = engine.search("old", scope_ids=["project:alpha"], include_common=False, mode="text")
    assert response.results[0].text_preview == "old marker"


def test_reader_holds_generation_until_search_completes(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    document = source / "guide.md"
    document.write_text("old marker", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    registered = HybridSource(source, scope_id="project:alpha", embed_allowed=True)
    engine.build([registered], embed_fn=lambda _: [1.0, 0.0])

    entered = threading.Event()
    release = threading.Event()
    search_result: list[HybridSearchResponse] = []
    failures: list[BaseException] = []

    def query_embed(_: str) -> list[float]:
        entered.set()
        assert release.wait(5)
        return [1.0, 0.0]

    def run_search() -> None:
        try:
            search_result.append(
                engine.search(
                    "old",
                    scope_ids=["project:alpha"],
                    include_common=False,
                    query_embed_fn=query_embed,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def run_build() -> None:
        try:
            engine.build([registered], embed_fn=lambda _: [1.0, 0.0])
        except BaseException as exc:  # pragma: no cover - asserted below
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
    old_response = search_result[0]
    assert old_response.results[0].text_preview == "old marker"
    new_response = engine.search(
        "new", scope_ids=["project:alpha"], include_common=False, mode="text"
    )
    assert new_response.results[0].text_preview == "new marker"


def test_builders_are_serialized_and_old_generations_are_collected(tmp_path: Path) -> None:
    source = tmp_path / "project"
    source.mkdir()
    (source / "guide.md").write_text("serialized", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    registered = HybridSource(source, scope_id="project:alpha", embed_allowed=True)
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
            engine.build([registered], embed_fn=slow_embed)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [threading.Thread(target=run_build) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert failures == []
    assert maximum_active == 1
    assert all(not thread.is_alive() for thread in threads)
    assert len(list(engine.generations_root.iterdir())) == 2

    previous_generation = engine.current_path.read_text(encoding="utf-8").strip()
    abandoned = engine.generations_root / ("f" * 32)
    abandoned.mkdir()
    engine.build([registered], embed_fn=lambda _: [1.0, 0.0])
    current_generation = engine.current_path.read_text(encoding="utf-8").strip()

    assert {path.name for path in engine.generations_root.iterdir()} == {
        previous_generation,
        current_generation,
    }


def test_required_scope_rejects_generation_rebuilt_for_another_project(
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "note.md").write_text("alpha-only marker", encoding="utf-8")
    (beta / "note.md").write_text("beta-only marker", encoding="utf-8")
    engine = HybridSearchEngine(tmp_path / "index")
    engine.build([HybridSource(alpha, scope_id="project:alpha")])
    engine.build([HybridSource(beta, scope_id="project:beta")])

    with pytest.raises(HybridScopeCoverageError, match="project:alpha"):
        engine.search(
            "marker",
            scope_ids=["project:alpha"],
            include_common=False,
            mode="text",
            required_scope_ids=["project:alpha"],
        )


def test_source_policy_digest_changes_when_foreign_root_is_excluded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    unrestricted = HybridSource(source, scope_id="project:alpha")
    restricted = HybridSource(
        source,
        scope_id="project:alpha",
        excluded_roots=(nested,),
    )

    from afs.hybrid_search import source_policy_digest

    assert source_policy_digest([unrestricted]) != source_policy_digest([restricted])
