from __future__ import annotations

from pathlib import Path

import pytest

from afs.context_layout import scaffold_v2
from afs.project_registry import COMMON_SCOPE_ID, ProjectRegistry
from afs.schema import AFSConfig, SensitivityConfig
from afs.search_service import ScopedSearchRequest, search_scoped


def _central_context(tmp_path: Path) -> tuple[Path, Path, Path, ProjectRegistry]:
    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    registry.register(alpha)
    registry.register(beta)
    return context, alpha, beta, registry


def _fake_embedder_factory(calls: list[str]):
    def factory(provider: str, **kwargs):
        def embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.5, 0.25]

        embed._afs_embedding_provider = provider  # type: ignore[attr-defined]
        default_model = (
            "nomic-embed-text" if provider == "ollama" else "gemini-embedding-2"
        )
        embed._afs_embedding_model = kwargs.get(  # type: ignore[attr-defined]
            "model", default_model
        )
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = kwargs.get(  # type: ignore[attr-defined]
            "task_type", "RETRIEVAL_DOCUMENT"
        )
        return embed

    return factory


def test_search_scoped_limits_v2_results_to_current_project_and_common(
    tmp_path: Path,
) -> None:
    context, alpha, beta, registry = _central_context(tmp_path)
    (alpha / "alpha.md").write_text("boundary-token alpha", encoding="utf-8")
    (beta / "beta.md").write_text("boundary-token beta-private", encoding="utf-8")
    common = context / "knowledge" / "common"
    common.mkdir(parents=True, exist_ok=True)
    (common / "shared.md").write_text("boundary-token common", encoding="utf-8")

    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="boundary-token",
            rebuild=True,
        ),
        config=AFSConfig(),
    )

    alpha_record = registry.resolve(alpha)
    assert alpha_record is not None
    assert result.project_id == alpha_record.project_id
    assert result.scope_id == alpha_record.scope_id
    assert {Path(hit.source_path).name for hit in result.response.results} == {
        "alpha.md",
        "shared.md",
    }
    payload = result.to_dict()
    assert payload["project_id"] == alpha_record.project_id
    assert payload["rebuilt"] is True
    assert payload["semantic_status"] == "not_requested"
    assert payload["semantic_requested"] is False
    assert payload["embedding_provider"] == ""
    assert payload["embedding_model"] == ""


def test_search_scoped_does_not_construct_embedder_without_semantic_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha, _beta, _registry = _central_context(tmp_path)
    (alpha / "keyword.md").write_text("local-keyword", encoding="utf-8")

    def unexpected_factory(_provider: str, **_kwargs):
        raise AssertionError("embedding factory called without semantic opt-in")

    monkeypatch.setattr("afs.embeddings.create_embed_fn", unexpected_factory)
    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="local-keyword",
            rebuild=True,
            semantic=False,
        ),
        config=AFSConfig(),
    )

    assert result.response.semantic_status == "not_requested"
    assert [Path(hit.source_path).name for hit in result.response.results] == ["keyword.md"]


def test_search_request_rejects_invalid_input_before_semantic_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha, _beta, _registry = _central_context(tmp_path)
    (alpha / "secret.md").write_text("must-not-be-embedded", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr("afs.embeddings.create_embed_fn", _fake_embedder_factory(calls))

    with pytest.raises(ValueError, match="query cannot be empty"):
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="  ",
            semantic=True,
        )
    assert calls == []

    with pytest.raises(ValueError, match="between 1 and 100"):
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="valid",
            limit=0,
        )


def test_search_scoped_semantic_coverage_is_current_plus_common(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha, beta, registry = _central_context(tmp_path)
    (alpha / "alpha.md").write_text("alpha-semantic-canary", encoding="utf-8")
    (beta / "beta.md").write_text("beta-semantic-secret", encoding="utf-8")
    common = context / "knowledge" / "common"
    common.mkdir(parents=True, exist_ok=True)
    (common / "shared.md").write_text("common-semantic-canary", encoding="utf-8")
    embed_calls: list[str] = []
    factory = _fake_embedder_factory(embed_calls)
    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)

    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="semantic canary",
            semantic=True,
            embedding_provider="ollama",
        ),
        config=AFSConfig(),
    )

    alpha_record = registry.resolve(alpha)
    assert alpha_record is not None
    assert result.build is not None
    assert set(result.build.intended_scope_ids) == {
        alpha_record.scope_id,
        COMMON_SCOPE_ID,
    }
    assert set(result.build.embedded_scope_ids) == {
        alpha_record.scope_id,
        COMMON_SCOPE_ID,
    }
    assert result.response.semantic_status == "ready"
    assert result.embedding_provider == "ollama"
    assert result.embedding_model == "nomic-embed-text"
    assert result.to_dict()["embedding_model"] == "nomic-embed-text"
    transmitted = "\n".join(embed_calls)
    assert "alpha-semantic-canary" in transmitted
    assert "common-semantic-canary" in transmitted
    assert "beta-semantic-secret" not in transmitted


def test_search_scoped_enforces_sensitivity_for_index_embedding_and_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha, _beta, _registry = _central_context(tmp_path)
    for name, marker in (
        ("visible.md", "policy-token visible-canary"),
        ("blocked-index.md", "policy-token index-secret"),
        ("blocked-embed.md", "policy-token embed-secret"),
        ("blocked-export.md", "policy-token export-secret"),
    ):
        (alpha / name).write_text(marker, encoding="utf-8")
    embed_calls: list[str] = []
    factory = _fake_embedder_factory(embed_calls)
    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)
    config = AFSConfig(
        sensitivity=SensitivityConfig(
            never_index=["blocked-index.md"],
            never_embed=["blocked-embed.md"],
            never_export=["blocked-export.md"],
        )
    )

    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="policy-token",
            rebuild=True,
            semantic=True,
        ),
        config=config,
    )

    result_names = {Path(hit.source_path).name for hit in result.response.results}
    assert "visible.md" in result_names
    assert "blocked-index.md" not in result_names
    assert "blocked-export.md" not in result_names
    transmitted = "\n".join(embed_calls)
    assert "visible-canary" in transmitted
    assert "index-secret" not in transmitted
    assert "embed-secret" not in transmitted
    assert "export-secret" not in transmitted


def test_search_scoped_excludes_exact_paths_before_index_and_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha, _beta, _registry = _central_context(tmp_path)
    visible = alpha / "visible.md"
    excluded = alpha / "generated-report.md"
    visible.write_text("exclusion-token visible-canary", encoding="utf-8")
    excluded.write_text(
        "exclusion-token generated-report-must-not-transmit",
        encoding="utf-8",
    )
    embed_calls: list[str] = []
    factory = _fake_embedder_factory(embed_calls)
    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)

    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=alpha,
            query="exclusion-token",
            rebuild=True,
            semantic=True,
            embedding_provider="ollama",
            excluded_paths=(excluded,),
        ),
        config=AFSConfig(),
    )

    assert {Path(hit.source_path).name for hit in result.response.results} == {
        "visible.md"
    }
    transmitted = "\n".join(embed_calls)
    assert "visible-canary" in transmitted
    assert "generated-report-must-not-transmit" not in transmitted
