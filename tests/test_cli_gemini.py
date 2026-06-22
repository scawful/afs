from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

from afs.cli import gemini
from afs.embeddings import SearchResult
from afs.schema import AFSConfig


def _base_args(**overrides) -> argparse.Namespace:
    payload = {
        "config": None,
        "context_root": None,
        "knowledge_path": None,
        "project": None,
        "json": False,
        "skip_ping": True,
        "query": None,
        "top_k": 5,
        "min_score": 0.3,
        "include_content": False,
        "settings_path": None,
        "scope": "user",
        "project_path": None,
        "python_module": False,
        "force": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_gemini_setup_writes_afs_mcp_entry(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    exit_code = gemini.gemini_setup_command(_base_args())
    capsys.readouterr()

    assert exit_code == 0
    settings_path = tmp_path / ".gemini" / "settings.json"
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    entry = payload["mcpServers"]["afs"]
    assert entry["args"] == ["mcp", "serve"]
    assert entry["command"].endswith("/scripts/afs")
    assert entry["env"]["AFS_PREFER_REPO_CONFIG"] == "1"
    assert entry["env"]["AFS_ROOT"] == str(Path(entry["command"]).parents[1])


def test_gemini_setup_supports_project_scope(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "repo"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))

    exit_code = gemini.gemini_setup_command(
        _base_args(scope="project", project_path=str(project))
    )
    capsys.readouterr()

    assert exit_code == 0
    settings_path = project / ".gemini" / "settings.json"
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    entry = payload["mcpServers"]["afs"]
    assert entry["cwd"] == str(project)
    assert entry["env"]["AFS_PREFER_REPO_CONFIG"] == "1"


def test_gemini_status_detects_child_indexes_under_context_root(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    context_root = tmp_path / ".context"
    project_root = context_root / "knowledge" / "project-a"
    project_root.mkdir(parents=True)
    (project_root / "embedding_index.json").write_text(
        json.dumps([{"doc_id": "a"}, {"doc_id": "b"}]) + "\n",
        encoding="utf-8",
    )

    config = AFSConfig()
    config.general.context_root = context_root

    monkeypatch.setattr(gemini, "_load_cli_config", lambda _args: config)
    monkeypatch.setattr(
        gemini,
        "find_afs_mcp_registrations",
        lambda: {"gemini": [], "claude": [], "codex": []},
    )
    fake_google = types.ModuleType("google")
    fake_google.genai = object()
    monkeypatch.setitem(sys.modules, "google", fake_google)

    exit_code = gemini.gemini_status_command(_base_args(json=True))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    embedding_check = next(
        check for check in payload["checks"] if check["name"] == "Embeddings indexed"
    )
    assert embedding_check["ok"] is True
    assert "1 index roots, 2 docs" in embedding_check["detail"]


# ---------------------------------------------------------------------------
# Gemini status command
# ---------------------------------------------------------------------------


def test_gemini_status_runs_without_api_key(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Status command should complete gracefully even without API keys."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    config = AFSConfig()
    monkeypatch.setattr(gemini, "_load_cli_config", lambda _args: config)
    monkeypatch.setattr(
        gemini,
        "find_afs_mcp_registrations",
        lambda: {"gemini": [], "claude": [], "codex": []},
    )

    exit_code = gemini.gemini_status_command(_base_args(skip_ping=True))
    captured = capsys.readouterr()

    # The command returns 1 because several checks fail, but it should not crash
    assert exit_code in (0, 1)
    assert "API key" in captured.out or "NOT SET" in captured.out


def test_gemini_status_json_produces_valid_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """--json flag should produce a parseable JSON payload with expected keys."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    config = AFSConfig()
    monkeypatch.setattr(gemini, "_load_cli_config", lambda _args: config)
    monkeypatch.setattr(
        gemini,
        "find_afs_mcp_registrations",
        lambda: {"gemini": [], "claude": [], "codex": []},
    )

    exit_code = gemini.gemini_status_command(_base_args(json=True, skip_ping=True))
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert "checks" in payload
    assert "embedding_index" in payload
    assert "api_latency" in payload
    check_names = {c["name"] for c in payload["checks"]}
    assert "API key" in check_names
    assert "API ping" in check_names


def test_gemini_status_skip_ping_flag_is_respected(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """When --skip-ping is set, API ping should show 'skipped' even if key exists."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-fake")

    config = AFSConfig()
    monkeypatch.setattr(gemini, "_load_cli_config", lambda _args: config)
    monkeypatch.setattr(
        gemini,
        "find_afs_mcp_registrations",
        lambda: {"gemini": [], "claude": [], "codex": []},
    )

    exit_code = gemini.gemini_status_command(_base_args(json=True, skip_ping=True))
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    api_ping = next(c for c in payload["checks"] if c["name"] == "API ping")
    assert api_ping["detail"] == "skipped"
    assert payload["api_latency"]["tested"] is False


def test_gemini_context_search_merges_indexed_children_and_uses_query_task_type(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    context_root = tmp_path / ".context"
    knowledge_root = context_root / "knowledge"
    alpha_root = knowledge_root / "alpha"
    beta_root = knowledge_root / "beta"
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    (alpha_root / "embedding_index.json").write_text("[]\n", encoding="utf-8")
    (beta_root / "embedding_index.json").write_text("[]\n", encoding="utf-8")

    config = AFSConfig()
    config.general.context_root = context_root
    monkeypatch.setattr(gemini, "_load_cli_config", lambda _args: config)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    calls: list[tuple[str, str]] = []

    def fake_create_embed_fn(provider: str, **kwargs):
        calls.append((provider, str(kwargs.get("task_type"))))
        return object()

    def fake_search_embedding_index(root: Path, query: str, **kwargs):
        assert query == "sprite"
        if root.name == "alpha":
            return [
                SearchResult(
                    doc_id="alpha::doc-1",
                    score=0.9,
                    source_path=str(root / "doc-1.md"),
                    text_preview="alpha result",
                )
            ]
        if root.name == "beta":
            return [
                SearchResult(
                    doc_id="beta::doc-2",
                    score=0.8,
                    source_path=str(root / "doc-2.md"),
                    text_preview="beta result",
                )
            ]
        return []

    monkeypatch.setattr(gemini, "create_embed_fn", fake_create_embed_fn)
    monkeypatch.setattr(gemini, "search_embedding_index", fake_search_embedding_index)

    exit_code = gemini.gemini_context_command(
        _base_args(query="sprite", json=True, top_k=2, min_score=0.0)
    )

    assert exit_code == 0
    assert calls == [("gemini", "RETRIEVAL_QUERY")]

    payload = json.loads(capsys.readouterr().out)
    assert [doc["doc_id"] for doc in payload["documents"]] == [
        "alpha::doc-1",
        "beta::doc-2",
    ]
