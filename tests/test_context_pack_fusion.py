from __future__ import annotations

from afs.context_pack import (
    ContextPackSection,
    _fused_retrieval_section,
    rrf_fuse,
)


def test_rrf_rewards_corroboration_across_signals() -> None:
    # B is #2 by keyword but #1 by semantic; A is #1 keyword only.
    fused = rrf_fuse([["A", "B", "D"], ["B", "C", "A"]])
    keys = [key for key, _ in fused]
    assert keys[0] == "B"  # corroborated by both -> top
    assert keys.index("A") < keys.index("C")  # A (in both) beats C (semantic-only)


def test_rrf_is_deterministic_on_ties() -> None:
    # Single-list inputs with symmetric ranks: ties break by key.
    fused = rrf_fuse([["x", "y"], ["y", "x"]])
    assert [key for key, _ in fused] == ["x", "y"]


def test_rrf_ignores_empty_keys_and_lists() -> None:
    fused = rrf_fuse([["", "a"], []])
    assert [key for key, _ in fused] == ["a"]


def test_rrf_k_damps_rank_contribution() -> None:
    # Larger k flattens the gap between rank 1 and rank 2.
    top_small_k = rrf_fuse([["a", "b"]], k=1)
    top_large_k = rrf_fuse([["a", "b"]], k=1000)
    gap_small = top_small_k[0][1] - top_small_k[1][1]
    gap_large = top_large_k[0][1] - top_large_k[1][1]
    assert gap_large < gap_small


def _section(title: str, path: str) -> ContextPackSection:
    return ContextPackSection(title=title, body=path, priority=1, sources=[path])


def test_fused_section_requires_both_signals() -> None:
    query_sections = [_section("Indexed Hit 1", "/repo/a.py")]
    # Only keyword hits, no embedding section -> no fused section.
    assert _fused_retrieval_section(query_sections, None, pack_mode="focused", max_results=5) is None

    # Only embedding hits, no keyword sections -> no fused section.
    embedding = ContextPackSection(
        title="Semantic Hits", body="hits", priority=4, sources=["/repo/b.py"]
    )
    assert _fused_retrieval_section([], embedding, pack_mode="focused", max_results=5) is None


def test_fused_section_annotates_signal_provenance() -> None:
    query_sections = [
        _section("Indexed Hit 1", "/repo/a.py"),
        _section("Indexed Hit 2", "/repo/shared.py"),
    ]
    embedding = ContextPackSection(
        title="Semantic Hits",
        body="hits",
        priority=4,
        sources=["/repo/shared.py", "/repo/c.py"],
    )
    section = _fused_retrieval_section(
        query_sections, embedding, pack_mode="focused", max_results=5
    )
    assert section is not None
    assert section.title == "Fused Retrieval"
    # shared.py is corroborated by both signals and should rank first with both tags.
    first_line = [ln for ln in section.body.splitlines() if ln.startswith("- ")][0]
    assert "/repo/shared.py" in first_line
    assert "keyword #2" in first_line
    assert "semantic #1" in first_line
    # keyword-only / semantic-only docs are still present with a single signal tag.
    assert "/repo/a.py" in section.body
    assert "/repo/c.py" in section.body


def test_fused_section_rejects_keyword_fallback_mislabeled_as_semantic() -> None:
    query_sections = [_section("Indexed Hit 1", "/repo/a.py")]
    fallback = ContextPackSection(
        title="Indexed Text Hits", body="hits", priority=4, sources=["/repo/a.py"]
    )

    assert (
        _fused_retrieval_section(
            query_sections, fallback, pack_mode="focused", max_results=5
        )
        is None
    )
