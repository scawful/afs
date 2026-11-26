from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from afs.cli.embeddings import embeddings_probe_command, register_parsers
from afs.context_layout import scaffold_v2
from afs.embeddings import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_LMSTUDIO_BASE_URL,
    EMBEDDINGGEMMA_DOCUMENT_INSTRUCTION,
    EMBEDDINGGEMMA_QUERY_INSTRUCTION,
    NOMIC_DOCUMENT_INSTRUCTION,
    NOMIC_QUERY_INSTRUCTION,
    QWEN3_QUERY_INSTRUCTION,
    create_embed_fn,
    default_text_instructions,
    embed_texts,
    probe_embedding_providers,
    select_auto_embedding_provider,
)
from afs.hybrid_search import HybridSearchEngine, HybridSource
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig
from afs.search_service import ScopedSearchRequest, search_scoped


class _FakeHttpResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("HTTP error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


def test_default_text_instructions_modern_models() -> None:
    assert default_text_instructions("text-embedding-nomic-embed-text-v1.5") == (
        NOMIC_DOCUMENT_INSTRUCTION,
        NOMIC_QUERY_INSTRUCTION,
    )
    assert default_text_instructions("embeddinggemma:latest") == (
        EMBEDDINGGEMMA_DOCUMENT_INSTRUCTION,
        EMBEDDINGGEMMA_QUERY_INSTRUCTION,
    )
    assert default_text_instructions("qwen3-embedding-0.6b") == ("", QWEN3_QUERY_INSTRUCTION)
    assert default_text_instructions("BAAI/bge-small-en-v1.5") == ("", BGE_QUERY_INSTRUCTION)
    assert default_text_instructions("mxbai-embed-large") == ("", BGE_QUERY_INSTRUCTION)
    assert default_text_instructions("snowflake-arctic-embed-m") == ("", BGE_QUERY_INSTRUCTION)
    assert default_text_instructions("custom-encoder") == ("", "")


def test_lmstudio_backend_batching_and_sorted_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        if isinstance(json["input"], list):
            return _FakeHttpResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [0.2, 0.8]},
                        {"index": 0, "embedding": [0.8, 0.2]},
                    ]
                }
            )
        return _FakeHttpResponse({"data": [{"embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(httpx, "post", fake_post)

    embed = create_embed_fn("lmstudio", model="text-embedding-nomic-embed-text-v1.5")
    assert embed("doc text") == [1.0, 0.0]
    assert calls[0]["url"] == f"{DEFAULT_LMSTUDIO_BASE_URL}/embeddings"
    assert calls[0]["json"]["input"] == "search_document: doc text"

    calls.clear()
    vectors = embed_texts(embed, ["first", "second"], batch_size=2)
    assert vectors == [[0.8, 0.2], [0.2, 0.8]]
    assert len(calls) == 1
    assert calls[0]["json"]["input"] == [
        "search_document: first",
        "search_document: second",
    ]


def test_ollama_batch_embed_fast_path_and_404_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_post_modern(url, *, json=None, timeout=None):
        calls.append(url)
        return _FakeHttpResponse({"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    monkeypatch.setattr(httpx, "post", fake_post_modern)
    embed = create_embed_fn("ollama", model="nomic-embed-text", host="http://localhost:11434")
    assert embed_texts(embed, ["a", "b"], batch_size=8) == [[1.0, 0.0], [0.0, 1.0]]
    assert calls == ["http://localhost:11434/api/embed"]

    calls.clear()

    def fake_post_legacy(url, *, json=None, timeout=None):
        calls.append(url)
        if url.endswith("/api/embed"):
            return _FakeHttpResponse({}, status_code=404)
        return _FakeHttpResponse({"embedding": [0.5, 0.5]})

    monkeypatch.setattr(httpx, "post", fake_post_legacy)
    assert embed_texts(embed, ["a", "b"], batch_size=8) == [[0.5, 0.5], [0.5, 0.5]]
    assert calls == [
        "http://localhost:11434/api/embed",
        "http://localhost:11434/api/embeddings",
        "http://localhost:11434/api/embeddings",
    ]


def test_gemini_sdk_batch_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_calls: list[dict] = []

    class _FakeModels:
        def embed_content(self, *, model: str, contents, config: dict):
            sdk_calls.append({"model": model, "contents": contents, "config": config})
            if isinstance(contents, list):
                embs = [SimpleNamespace(values=[float(i + 1), 0.5]) for i in range(len(contents))]
            else:
                embs = [SimpleNamespace(values=[1.0, 0.5])]
            return SimpleNamespace(embeddings=embs)

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = _FakeModels()

    fake_genai = SimpleNamespace(Client=_FakeClient)
    import google

    monkeypatch.setattr(google, "genai", fake_genai, raising=False)

    embed = create_embed_fn("gemini", model="gemini-embedding-2", api_key="test-key")
    vectors = embed_texts(embed, ["alpha", "beta"], batch_size=8)
    assert vectors == [[1.0, 0.5], [2.0, 0.5]]
    assert len(sdk_calls) == 1
    assert sdk_calls[0]["contents"] == [
        "title: none | text: alpha",
        "title: none | text: beta",
    ]
    assert "task_type" not in sdk_calls[0]["config"]


def test_optional_fastembed_and_mlx_missing_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"fastembed", "mlx_embeddings"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install fastembed"):
        create_embed_fn("fastembed")
    with pytest.raises(RuntimeError, match="pip install mlx-embeddings"):
        create_embed_fn("mlx")


def test_probe_embedding_providers_ranks_and_prefers_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def fake_get(url: str, timeout: float = 1.5):
        if "1234" in url:
            return _FakeHttpResponse(
                {
                    "data": [
                        {"id": "oracle-tools-7b"},
                        {"id": "text-embedding-embeddings"},
                        {"id": "text-embedding-nomic-embed-text-v1.5"},
                        {"id": "qwen3-embedding-0.6b"},
                    ]
                }
            )
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    probes = probe_embedding_providers(timeout=0.5)
    by_name = {probe.provider: probe for probe in probes}

    assert by_name["lmstudio"].available is True
    assert by_name["lmstudio"].recommended_model == "qwen3-embedding-0.6b"
    assert "oracle-tools-7b" not in by_name["lmstudio"].detected_models
    assert by_name["gemini"].available is True

    selected_local = select_auto_embedding_provider(probes, prefer_local=True)
    assert selected_local is not None
    assert selected_local.provider == "lmstudio"


def test_hybrid_search_build_batches_and_recreates_lmstudio_query_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    post_calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        post_calls.append({"url": url, "json": json})
        inp = json["input"]
        if isinstance(inp, list):
            return _FakeHttpResponse(
                {
                    "data": [
                        {"index": idx, "embedding": [1.0 if "approval" in item else 0.0, 0.5]}
                        for idx, item in enumerate(inp)
                    ]
                }
            )
        return _FakeHttpResponse(
            {"data": [{"embedding": [1.0 if "approval" in inp else 0.0, 0.5]}]}
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    source = tmp_path / "project"
    source.mkdir()
    (source / "a.md").write_text("approval gate verification notes", encoding="utf-8")
    (source / "b.md").write_text("general readme notes", encoding="utf-8")

    engine = HybridSearchEngine(tmp_path / "search")
    embed_fn = create_embed_fn("lmstudio", model="text-embedding-nomic-embed-text-v1.5")
    build_res = engine.build(
        [HybridSource(source, scope_id="project:alpha", embed_allowed=True)],
        embed_fn=embed_fn,
    )

    assert build_res.healthy
    assert len(post_calls) == 1
    assert isinstance(post_calls[0]["json"]["input"], list)
    assert len(post_calls[0]["json"]["input"]) == 2

    metadata = json.loads(engine.metadata_path.read_text(encoding="utf-8"))
    assert metadata["collection"]["provider"] == "lmstudio"
    assert metadata["collection"]["endpoint"] == DEFAULT_LMSTUDIO_BASE_URL
    assert metadata["collection"]["query_instruction"] == NOMIC_QUERY_INSTRUCTION

    post_calls.clear()
    response = engine.search(
        "approval",
        scope_ids=["project:alpha"],
        include_common=False,
        mode="hybrid",
        recreate_query_embedder=True,
    )
    assert response.semantic_status == "ready"
    assert len(post_calls) == 1
    assert post_calls[0]["json"]["input"] == "search_query: approval"


def test_search_scoped_auto_provider_and_cli_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def fake_get(url: str, timeout: float = 1.5):
        if "1234" in url:
            return _FakeHttpResponse(
                {"data": [{"id": "text-embedding-nomic-embed-text-v1.5"}]}
            )
        raise httpx.ConnectError("down")

    def fake_post(url, *, json=None, headers=None, timeout=None):
        inp = json["input"]
        if isinstance(inp, list):
            return _FakeHttpResponse(
                {"data": [{"index": i, "embedding": [1.0, 0.25]} for i in range(len(inp))]}
            )
        return _FakeHttpResponse({"data": [{"embedding": [1.0, 0.25]}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    import argparse

    parser = argparse.ArgumentParser()
    register_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["embeddings", "probe", "--json"])
    assert embeddings_probe_command(args) == 0
    probe_payload = json.loads(capsys.readouterr().out)
    assert probe_payload["recommended"]["provider"] == "lmstudio"
    assert probe_payload["recommended"]["recommended_model"] == "text-embedding-nomic-embed-text-v1.5"

    context = tmp_path / ".context"
    project = tmp_path / "proj"
    project.mkdir()
    (project / "notes.md").write_text("semantic retrieval local endpoint", encoding="utf-8")
    scaffold_v2(context)
    ProjectRegistry(context).register(project)

    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=project,
            query="local endpoint",
            semantic=True,
            embedding_provider="auto",
        ),
        config=AFSConfig(),
    )
    assert result.embedding_provider == "lmstudio"
    assert result.embedding_model == "text-embedding-nomic-embed-text-v1.5"
    assert result.response.semantic_status == "ready"
