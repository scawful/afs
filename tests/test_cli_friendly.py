from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from afs.cli import build_parser
from afs.cli.core import hivemind_list_command, hivemind_reap_command
from afs.cli.friendly import (
    handoff_create_command,
    handoff_list_command,
    handoff_revise_command,
    messages_clean_command,
    messages_list_command,
    messages_send_command,
    notes_archive_command,
    notes_create_command,
    notes_draft_command,
    notes_list_command,
    notes_promote_command,
    projects_current_command,
    projects_import_command,
    projects_list_command,
    search_command,
)
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry


def _central(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    root = tmp_path / ".context"
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    scaffold_v2(root)
    registry = ProjectRegistry(root)
    registry.register(project)
    registry.register(other)
    config = tmp_path / "afs.toml"
    config.write_text(f'[general]\ncontext_root = "{root}"\n', encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    return root, project, other


def test_projects_current_and_list_use_central_registry(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, project, _other = _central(tmp_path, monkeypatch)
    current = argparse.Namespace(
        config=None,
        path=str(project),
        context_root=None,
        context_dir=None,
        json=True,
    )
    assert projects_current_command(current) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_root"] == str(root.resolve())
    assert payload["scope_id"].startswith("project:prj_")

    listing = argparse.Namespace(config=None, context_root=None, json=True)
    assert projects_list_command(listing) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_projects_current_orients_an_unregistered_v2_checkout(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, _project, _other = _central(tmp_path, monkeypatch)
    unregistered = tmp_path / "unregistered"
    unregistered.mkdir()
    current = argparse.Namespace(
        config=None,
        path=str(unregistered),
        context_root=None,
        context_dir=None,
        json=True,
    )

    assert projects_current_command(current) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "context_root": str(root.resolve()),
        "layout_version": 2,
        "project_path": str(unregistered.resolve()),
        "registered": False,
        "scope_id": "common",
        "project": None,
        "next_step": f"afs projects register {unregistered.resolve()}",
    }


def test_common_artifacts_do_not_require_a_registered_cwd(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, _project, _other = _central(tmp_path, monkeypatch)
    unregistered = tmp_path / "unregistered"
    unregistered.mkdir()
    base = {
        "config": None,
        "path": str(unregistered),
        "context_root": None,
        "context_dir": None,
        "common": True,
        "limit": 10,
        "json": True,
    }

    assert notes_list_command(argparse.Namespace(**base)) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert handoff_list_command(argparse.Namespace(**base, thread=None)) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_projects_import_is_dry_run_until_apply(tmp_path: Path, monkeypatch, capsys) -> None:
    root, _project, _other = _central(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (tmp_path / "WORKSPACE.toml").write_text(
        '[[item]]\npath = "candidate"\ndescription = "new project"\n',
        encoding="utf-8",
    )

    args = argparse.Namespace(
        workspace_root=str(tmp_path),
        config=None,
        context_root=str(root),
        no_local=False,
        apply=False,
        json=True,
    )
    assert projects_import_command(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["applied"] is False
    assert preview["candidates"] == [str(candidate.resolve())]
    assert ProjectRegistry(root).resolve(candidate) is None

    args.apply = True
    assert projects_import_command(args) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert len(applied["registered"]) == 1
    assert ProjectRegistry(root).resolve(candidate) is not None


def test_messages_commands_enforce_current_scope(tmp_path: Path, monkeypatch, capsys) -> None:
    _root, project, other = _central(tmp_path, monkeypatch)

    def args(path: Path, **values):
        defaults = {
            "config": None,
            "path": str(path),
            "context_root": None,
            "context_dir": None,
            "all_projects": False,
            "include_legacy": False,
            "json": True,
        }
        defaults.update(values)
        return argparse.Namespace(**defaults)

    assert (
        messages_send_command(
            args(
                project,
                from_agent="alpha",
                type="status",
                payload='{"ok": true}',
                to=None,
                topic=None,
                ttl_hours=None,
                scope=None,
            )
        )
        == 0
    )
    capsys.readouterr()

    assert messages_list_command(args(project, agent=None, type=None, topic=None, limit=10)) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

    assert messages_list_command(args(other, agent=None, type=None, topic=None, limit=10)) == 0
    assert json.loads(capsys.readouterr().out) == []

    legacy_args = argparse.Namespace(
        config=None,
        path=str(other),
        context_root=None,
        context_dir=None,
        topic=None,
        limit=10,
    )
    assert hivemind_list_command(legacy_args) == 0
    legacy_output = capsys.readouterr().out
    assert "no messages" in legacy_output
    assert "alpha" not in legacy_output


def test_legacy_message_cleanup_requires_explicit_v2_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, project, _other = _central(tmp_path, monkeypatch)
    args = argparse.Namespace(
        config=None,
        path=str(project),
        context_root=None,
        context_dir=None,
        max_age_hours=None,
        dry_run=True,
        all_projects=False,
        json=False,
    )

    assert hivemind_reap_command(args) == 1
    assert "--all-projects" in capsys.readouterr().out


def test_friendly_top_level_parsers_are_discoverable() -> None:
    parser = build_parser(["start", "--help"])
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert {
        "start",
        "search",
        "repair",
        "projects",
        "notes",
        "handoff",
        "messages",
    }.issubset(choices)


def test_search_rebuilds_for_each_current_plus_common_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, project, other = _central(tmp_path, monkeypatch)
    (project / "alpha.md").write_text("shared-token alpha guide", encoding="utf-8")
    (other / "beta.md").write_text("shared-token beta secret", encoding="utf-8")
    common = root / "knowledge" / "common"
    common.mkdir(parents=True)
    (common / "shared.md").write_text("shared-token common note", encoding="utf-8")

    def args(path: Path, *, rebuild: bool) -> argparse.Namespace:
        return argparse.Namespace(
            config=None,
            path=str(path),
            context_root=None,
            context_dir=None,
            query="shared-token",
            semantic=False,
            all_projects=False,
            rebuild=rebuild,
            mode="text",
            limit=10,
            provider="gemini",
            model=None,
            json=True,
        )

    assert search_command(args(project, rebuild=True)) == 0
    alpha = json.loads(capsys.readouterr().out)
    alpha_paths = {Path(hit["source_path"]).name for hit in alpha["results"]}
    assert alpha_paths == {"alpha.md", "shared.md"}
    assert alpha["rebuilt"] is True
    assert alpha["semantic_status"] == "not_requested"

    assert search_command(args(other, rebuild=False)) == 0
    beta = json.loads(capsys.readouterr().out)
    beta_paths = {Path(hit["source_path"]).name for hit in beta["results"]}
    assert beta_paths == {"beta.md", "shared.md"}
    assert beta["rebuilt"] is True


def test_search_rejects_a_linked_current_project_category_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, project, other = _central(tmp_path, monkeypatch)
    registry = ProjectRegistry(root)
    alpha_record = registry.resolve(project)
    beta_record = registry.resolve(other)
    assert alpha_record is not None
    assert beta_record is not None
    beta_root = root / "knowledge" / "projects" / beta_record.project_id
    beta_root.mkdir(parents=True)
    (beta_root / "secret.md").write_text("linked-search-private", encoding="utf-8")
    alpha_root = root / "knowledge" / "projects" / alpha_record.project_id
    alpha_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        alpha_root.symlink_to(beta_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    args = argparse.Namespace(
        config=None,
        path=str(project),
        context_root=None,
        context_dir=None,
        query="linked-search-private",
        semantic=False,
        all_projects=False,
        rebuild=True,
        mode="text",
        limit=10,
        provider="gemini",
        model=None,
        json=True,
    )
    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        search_command(args)


def test_search_semantic_opt_in_upgrades_keyword_index(tmp_path: Path, monkeypatch, capsys) -> None:
    _root, project, _other = _central(tmp_path, monkeypatch)
    (project / "semantic.md").write_text("semantic upgrade marker", encoding="utf-8")

    def args(*, semantic: bool, rebuild: bool) -> argparse.Namespace:
        return argparse.Namespace(
            config=None,
            path=str(project),
            context_root=None,
            context_dir=None,
            query="semantic marker",
            semantic=semantic,
            all_projects=False,
            rebuild=rebuild,
            mode="text",
            limit=10,
            provider="gemini",
            model=None,
            json=True,
        )

    assert search_command(args(semantic=False, rebuild=True)) == 0
    keyword = json.loads(capsys.readouterr().out)
    assert keyword["build"]["vector_count"] == 0

    def factory(_provider: str, **kwargs):
        def embed(_text: str) -> list[float]:
            return [1.0, 0.0, 0.25]

        embed._afs_embedding_provider = "gemini"  # type: ignore[attr-defined]
        embed._afs_embedding_model = kwargs.get("model", "gemini-embedding-2")  # type: ignore[attr-defined]
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = kwargs.get("task_type", "RETRIEVAL_DOCUMENT")  # type: ignore[attr-defined]
        return embed

    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)
    assert search_command(args(semantic=True, rebuild=False)) == 0
    semantic = json.loads(capsys.readouterr().out)
    assert semantic["rebuilt"] is True
    assert semantic["build"]["vector_count"] > 0
    assert semantic["semantic_status"] == "ready"


def test_search_semantic_rebuilds_only_current_plus_common_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, alpha, beta = _central(tmp_path, monkeypatch)
    (alpha / "alpha.md").write_text("alpha-semantic-marker", encoding="utf-8")
    (beta / "beta.md").write_text("beta-semantic-marker", encoding="utf-8")
    common = root / "knowledge" / "common"
    common.mkdir(parents=True)
    (common / "shared.md").write_text("common-semantic-marker", encoding="utf-8")
    embed_calls: list[str] = []

    def factory(_provider: str, **kwargs):
        def embed(text: str) -> list[float]:
            embed_calls.append(text)
            return [1.0, 0.5, 0.25]

        embed._afs_embedding_provider = "gemini"  # type: ignore[attr-defined]
        embed._afs_embedding_model = kwargs.get(  # type: ignore[attr-defined]
            "model", "gemini-embedding-2"
        )
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = kwargs.get(  # type: ignore[attr-defined]
            "task_type", "RETRIEVAL_DOCUMENT"
        )
        return embed

    def args(path: Path) -> argparse.Namespace:
        return argparse.Namespace(
            config=None,
            path=str(path),
            context_root=None,
            context_dir=None,
            query="semantic marker",
            semantic=True,
            all_projects=False,
            rebuild=False,
            mode="text",
            limit=10,
            provider="gemini",
            model=None,
            json=True,
        )

    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)
    registry = ProjectRegistry(root)
    alpha_scope = registry.resolve(alpha).scope_id  # type: ignore[union-attr]
    beta_scope = registry.resolve(beta).scope_id  # type: ignore[union-attr]

    assert search_command(args(alpha)) == 0
    alpha_payload = json.loads(capsys.readouterr().out)
    assert alpha_payload["rebuilt"] is True
    assert set(alpha_payload["build"]["intended_scope_ids"]) == {
        alpha_scope,
        "common",
    }
    assert all("beta-semantic-marker" not in text for text in embed_calls)

    embed_calls.clear()
    assert search_command(args(beta)) == 0
    beta_payload = json.loads(capsys.readouterr().out)
    assert beta_payload["rebuilt"] is True
    assert set(beta_payload["build"]["intended_scope_ids"]) == {
        beta_scope,
        "common",
    }
    assert set(beta_payload["build"]["embedded_scope_ids"]) == {
        beta_scope,
        "common",
    }
    assert all("alpha-semantic-marker" not in text for text in embed_calls)
    assert any("beta-semantic-marker" in text for text in embed_calls)

    assert search_command(args(alpha)) == 0
    alpha_again = json.loads(capsys.readouterr().out)
    assert alpha_again["rebuilt"] is True
    assert set(alpha_again["build"]["intended_scope_ids"]) == {
        alpha_scope,
        "common",
    }
    alpha_hit = next(hit for hit in alpha_again["results"] if hit["relative_path"] == "alpha.md")
    assert "semantic" in alpha_hit["signals"]


def test_search_prunes_visible_central_context_nested_under_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    alpha = tmp_path / "workspace"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    context = alpha / "central-context"
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    (alpha / "alpha.md").write_text("alpha-visible-marker", encoding="utf-8")
    beta_memory = context / "knowledge" / "projects" / beta_record.project_id
    beta_memory.mkdir(parents=True)
    (beta_memory / "secret.md").write_text(
        "beta-private-canary",
        encoding="utf-8",
    )
    config = tmp_path / "afs.toml"
    config.write_text(
        f'[general]\ncontext_root = "{context}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    embedded: list[str] = []

    def factory(_provider: str, **kwargs):
        def embed(text: str) -> list[float]:
            embedded.append(text)
            return [1.0, 0.5, 0.25]

        embed._afs_embedding_provider = "gemini"  # type: ignore[attr-defined]
        embed._afs_embedding_model = kwargs.get(  # type: ignore[attr-defined]
            "model", "gemini-embedding-2"
        )
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = kwargs.get(  # type: ignore[attr-defined]
            "task_type", "RETRIEVAL_DOCUMENT"
        )
        return embed

    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)
    args = argparse.Namespace(
        config=None,
        path=str(alpha),
        context_root=None,
        context_dir=None,
        query="marker",
        semantic=True,
        all_projects=False,
        rebuild=True,
        mode="text",
        limit=10,
        provider="gemini",
        model=None,
        json=True,
    )

    assert search_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    rendered = json.dumps(payload)
    assert payload["project_id"] == alpha_record.project_id
    assert "alpha-visible-marker" in rendered
    assert "beta-private-canary" not in rendered
    assert all("beta-private-canary" not in text for text in embedded)


def test_search_prunes_nested_registered_project_from_parent_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    alpha = tmp_path / "workspace"
    beta = alpha / "nested-project"
    beta.mkdir(parents=True)
    context = tmp_path / ".context"
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    (alpha / "alpha.md").write_text("alpha-visible-marker", encoding="utf-8")
    (beta / "secret.md").write_text("beta-private-canary", encoding="utf-8")
    config = tmp_path / "afs.toml"
    config.write_text(
        f'[general]\ncontext_root = "{context}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    embedded: list[str] = []

    def factory(_provider: str, **kwargs):
        def embed(text: str) -> list[float]:
            embedded.append(text)
            return [1.0, 0.5, 0.25]

        embed._afs_embedding_provider = "gemini"  # type: ignore[attr-defined]
        embed._afs_embedding_model = kwargs.get(  # type: ignore[attr-defined]
            "model", "gemini-embedding-2"
        )
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = kwargs.get(  # type: ignore[attr-defined]
            "task_type", "RETRIEVAL_DOCUMENT"
        )
        return embed

    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)

    def args(*, all_projects: bool, query: str) -> argparse.Namespace:
        return argparse.Namespace(
            config=None,
            path=str(alpha),
            context_root=None,
            context_dir=None,
            query=query,
            semantic=True,
            all_projects=all_projects,
            rebuild=True,
            mode="text",
            limit=10,
            provider="gemini",
            model=None,
            json=True,
        )

    assert search_command(args(all_projects=False, query="marker")) == 0
    scoped = json.loads(capsys.readouterr().out)
    scoped_rendered = json.dumps(scoped)
    assert scoped["project_id"] == alpha_record.project_id
    assert "alpha-visible-marker" in scoped_rendered
    assert "beta-private-canary" not in scoped_rendered
    assert all("beta-private-canary" not in text for text in embedded)

    embedded.clear()
    assert search_command(
        args(all_projects=True, query="beta-private-canary")
    ) == 0
    all_scopes = json.loads(capsys.readouterr().out)
    beta_hits = [
        hit
        for hit in all_scopes["results"]
        if "beta-private-canary" in hit["text_preview"]
    ]
    assert beta_hits
    assert {hit["scope_id"] for hit in beta_hits} == {beta_record.scope_id}
    assert any("beta-private-canary" in text for text in embedded)


def test_search_never_traverses_inactive_project_without_all_projects(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root, alpha, beta = _central(tmp_path, monkeypatch)
    (alpha / "alpha.md").write_text("scope-traversal alpha", encoding="utf-8")
    (beta / "beta.md").write_text("scope-traversal beta", encoding="utf-8")

    from afs import hybrid_search

    original = hybrid_search._iter_safe_source_files
    traversed: list[Path] = []

    def observed(source, result, *, budget, excluded_root):
        traversed.append(Path(source.path))
        yield from original(
            source,
            result,
            budget=budget,
            excluded_root=excluded_root,
        )

    monkeypatch.setattr(hybrid_search, "_iter_safe_source_files", observed)

    def args(*, all_projects: bool) -> argparse.Namespace:
        return argparse.Namespace(
            config=None,
            path=str(alpha),
            context_root=None,
            context_dir=None,
            query="scope-traversal",
            semantic=False,
            all_projects=all_projects,
            rebuild=True,
            mode="text",
            limit=10,
            provider="gemini",
            model=None,
            json=True,
        )

    assert search_command(args(all_projects=False)) == 0
    capsys.readouterr()
    assert alpha.resolve() in traversed
    assert beta.resolve() not in traversed

    traversed.clear()
    assert search_command(args(all_projects=True)) == 0
    capsys.readouterr()
    assert alpha.resolve() in traversed
    assert beta.resolve() in traversed


def test_search_applies_sensitivity_to_local_index_remote_embedding_and_export(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root, project, _other = _central(tmp_path, monkeypatch)
    config = tmp_path / "afs.toml"
    config.write_text(
        f'[general]\ncontext_root = "{root}"\n'
        "[sensitivity]\n"
        'never_index = ["blocked-index.md"]\n'
        'never_embed = ["blocked-embed.md"]\n'
        'never_export = ["blocked-export.md"]\n',
        encoding="utf-8",
    )
    for name, marker in (
        ("visible.md", "policy-token visible-canary"),
        ("blocked-index.md", "policy-token index-secret"),
        ("blocked-embed.md", "policy-token embed-secret"),
        ("blocked-export.md", "policy-token export-secret"),
    ):
        (project / name).write_text(marker, encoding="utf-8")

    embed_calls: list[str] = []

    def factory(_provider: str, **kwargs):
        def embed(text: str) -> list[float]:
            embed_calls.append(text)
            return [1.0, 0.5, 0.25]

        embed._afs_embedding_provider = "gemini"  # type: ignore[attr-defined]
        embed._afs_embedding_model = kwargs.get("model", "gemini-embedding-2")  # type: ignore[attr-defined]
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = kwargs.get("task_type", "RETRIEVAL_DOCUMENT")  # type: ignore[attr-defined]
        return embed

    monkeypatch.setattr("afs.embeddings.create_embed_fn", factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", factory)
    args = argparse.Namespace(
        config=None,
        path=str(project),
        context_root=None,
        context_dir=None,
        query="policy-token",
        semantic=True,
        all_projects=False,
        rebuild=True,
        mode="text",
        limit=10,
        provider="gemini",
        model=None,
        json=True,
    )

    assert search_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    result_paths = {Path(hit["source_path"]).name for hit in payload["results"]}
    assert "visible.md" in result_paths
    assert "blocked-index.md" not in result_paths
    assert "blocked-export.md" not in result_paths
    transmitted = "\n".join(embed_calls)
    assert "visible-canary" in transmitted
    assert "index-secret" not in transmitted
    assert "embed-secret" not in transmitted
    assert "export-secret" not in transmitted


def test_sensitivity_policy_change_purges_forbidden_content_from_all_generations(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root, project, _other = _central(tmp_path, monkeypatch)
    config = tmp_path / "afs.toml"
    canary = project / "purge.md"
    canary.write_text("purge-policy-canary", encoding="utf-8")
    args = argparse.Namespace(
        config=None,
        path=str(project),
        context_root=None,
        context_dir=None,
        query="purge-policy-canary",
        semantic=False,
        all_projects=False,
        rebuild=True,
        mode="text",
        limit=10,
        provider="gemini",
        model=None,
        json=True,
    )
    assert search_command(args) == 0
    capsys.readouterr()

    config.write_text(
        f'[general]\ncontext_root = "{root}"\n'
        '[sensitivity]\nnever_index = ["purge.md"]\n',
        encoding="utf-8",
    )
    args.rebuild = False
    assert search_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rebuilt"] is True
    assert payload["results"] == []

    databases = list((root / ".afs" / "search" / "generations").glob("*/search.sqlite3"))
    assert len(databases) == 1
    for database in databases:
        with sqlite3.connect(database) as connection:
            bodies = "\n".join(row[0] for row in connection.execute("SELECT body FROM documents"))
        assert "purge-policy-canary" not in bodies


def test_message_cleanup_requires_explicit_all_projects() -> None:
    args = argparse.Namespace(all_projects=False)

    try:
        messages_clean_command(args)
    except PermissionError as exc:
        assert "--all-projects" in str(exc)
    else:  # pragma: no cover - assertion helper without pytest dependency
        raise AssertionError("cleanup unexpectedly crossed the scope boundary")


def _artifact_args(project: Path, **values) -> argparse.Namespace:
    defaults = {
        "config": None,
        "path": str(project),
        "context_root": None,
        "context_dir": None,
        "common": False,
        "json": True,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_notes_drafts_promote_and_archive_within_current_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, project, other = _central(tmp_path, monkeypatch)
    content = {
        "title": "Readable planning note",
        "body": "Keep the decision boundary explicit.",
        "body_file": None,
        "task_id": None,
        "agent_name": "codex",
        "author_kind": "agent",
        "sensitivity": "internal",
    }
    assert notes_create_command(_artifact_args(project, **content)) == 0
    durable = json.loads(capsys.readouterr().out)

    assert notes_list_command(_artifact_args(other, limit=10)) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert notes_draft_command(_artifact_args(project, **content)) == 0
    draft = json.loads(capsys.readouterr().out)
    draft_id = draft["metadata"]["artifact_id"]
    assert "readable-planning-note" in Path(draft["path"]).name

    assert notes_promote_command(_artifact_args(project, identifier=draft_id)) == 0
    promoted = json.loads(capsys.readouterr().out)
    assert promoted["metadata"]["provenance"]["source_artifact_id"] == draft_id
    assert promoted["path"] != durable["path"]

    assert notes_archive_command(_artifact_args(project, identifier=draft_id)) == 0
    archived = json.loads(capsys.readouterr().out)
    assert Path(archived["path"]).parent.name == "archive"


def test_handoff_create_and_revise_require_readable_titles(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, project, _other = _central(tmp_path, monkeypatch)
    base = {
        "title": "Context v2 implementation",
        "agent_name": "codex",
        "accomplished": ["layout complete"],
        "blocked": [],
        "next_steps": ["review search"],
        "target_agent": "reviewer",
        "priority": "high",
    }
    assert handoff_create_command(_artifact_args(project, **base)) == 0
    first = json.loads(capsys.readouterr().out)

    revised = dict(base)
    revised["title"] = "Context v2 review follow-up"
    assert (
        handoff_revise_command(_artifact_args(project, revision_id=first["revision_id"], **revised))
        == 0
    )
    second = json.loads(capsys.readouterr().out)

    assert second["stream_id"] == first["stream_id"]
    assert second["supersedes"] == [first["revision_id"]]
    assert "context-v2-review-follow-up" in Path(second["artifact_path"]).name
