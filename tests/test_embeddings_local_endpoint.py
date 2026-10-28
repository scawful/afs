from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pytest

from afs.cli.embeddings import _resolve_embed_fn, embeddings_search_command, register_parsers
from afs.embeddings import (
    NOMIC_DOCUMENT_INSTRUCTION,
    NOMIC_QUERY_INSTRUCTION,
    build_embedding_index,
    create_embed_fn,
    create_openai_embed_fn,
    create_query_embed_fn_from_index,
    load_embedding_collection_metadata,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _capture_httpx(monkeypatch, payload: dict) -> list[dict]:
    calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(payload)

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    register_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_ollama_collection_records_endpoint_and_query_embedder_reuses_it(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _capture_httpx(monkeypatch, {"embedding": [1.0, 0.0, 0.0]})
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("dungeon palette notes", encoding="utf-8")
    embed = create_embed_fn(
        "ollama",
        model="nomic-embed-text",
        host="http://127.0.0.1:9999",
        task_type=NOMIC_DOCUMENT_INSTRUCTION,
        query_task_type=NOMIC_QUERY_INSTRUCTION,
    )

    build_embedding_index([source], tmp_path / "index", embed_fn=embed)

    metadata = load_embedding_collection_metadata(tmp_path / "index")
    assert metadata.endpoint == "http://127.0.0.1:9999"
    assert metadata.document_instruction == NOMIC_DOCUMENT_INSTRUCTION
    assert metadata.query_instruction == NOMIC_QUERY_INSTRUCTION
    assert calls[0]["json"]["prompt"].startswith("search_document: ")

    calls.clear()
    query_embed = create_query_embed_fn_from_index(tmp_path / "index")
    assert query_embed("palette") == [1.0, 0.0, 0.0]
    assert calls[0]["url"] == "http://127.0.0.1:9999/api/embeddings"
    assert calls[0]["json"]["prompt"] == "search_query: palette"


def test_openai_compatible_loopback_server_needs_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    calls = _capture_httpx(monkeypatch, {"data": [{"embedding": [0.5, 0.5]}]})

    embed = create_openai_embed_fn(
        "text-embedding-nomic-embed-text-v1.5",
        base_url="http://localhost:1234/v1",
        task_type=NOMIC_QUERY_INSTRUCTION,
    )

    assert embed("sprite slots") == [0.5, 0.5]
    assert calls == [
        {
            "url": "http://localhost:1234/v1/embeddings",
            "json": {
                "model": "text-embedding-nomic-embed-text-v1.5",
                "input": "search_query: sprite slots",
            },
            "headers": {},
        }
    ]
    with pytest.raises(RuntimeError, match="Missing API key"):
        create_openai_embed_fn("text-embedding-3-small", base_url="https://api.example.com/v1")


def test_cli_defaults_nomic_instructions_and_allows_override(tmp_path: Path) -> None:
    base = [
        "embeddings",
        "index",
        "--knowledge-path",
        str(tmp_path),
        "--provider",
        "openai",
        "--model",
        "text-embedding-nomic-embed-text-v1.5",
        "--openai-base-url",
        "http://localhost:1234/v1",
    ]

    document_embed = _resolve_embed_fn(_parse(base))
    query_embed = _resolve_embed_fn(_parse(base), mode="query")
    plain_embed = _resolve_embed_fn(
        _parse([*base, "--document-instruction", "", "--query-instruction", ""])
    )

    assert document_embed._afs_embedding_instruction == NOMIC_DOCUMENT_INSTRUCTION
    assert document_embed._afs_embedding_query_instruction == NOMIC_QUERY_INSTRUCTION
    assert document_embed._afs_embedding_endpoint == "http://localhost:1234/v1"
    assert query_embed._afs_embedding_instruction == NOMIC_QUERY_INSTRUCTION
    assert plain_embed._afs_embedding_instruction == ""
    assert plain_embed._afs_embedding_query_instruction == ""


def test_cli_search_auto_recreates_query_embedder_from_metadata(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    calls = _capture_httpx(monkeypatch, {"data": [{"embedding": [1.0, 0.0]}]})
    source = tmp_path / "source"
    source.mkdir()
    (source / "rooms.md").write_text("room header layout", encoding="utf-8")
    embed = create_embed_fn(
        "openai",
        model="text-embedding-nomic-embed-text-v1.5",
        base_url="http://localhost:1234/v1",
        task_type=NOMIC_DOCUMENT_INSTRUCTION,
        query_task_type=NOMIC_QUERY_INSTRUCTION,
    )
    build_embedding_index([source], tmp_path / "index", embed_fn=embed)
    calls.clear()

    args = _parse(
        [
            "embeddings",
            "search",
            "--knowledge-path",
            str(tmp_path / "index"),
            "--provider",
            "auto",
            "--json",
            "room layout",
        ]
    )
    assert embeddings_search_command(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["semantic_status"] == "ready"
    assert payload["results"][0]["source_path"].endswith("rooms.md")
    assert calls[0]["json"]["input"] == "search_query: room layout"
