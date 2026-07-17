from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import afs.agents.research as research_agent
from afs.agent_defaults import default_agent_configs
from afs.agents import get_agent_registry
from afs.artifacts import NoteStore
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig
from afs.scratchpad import ScratchpadStore


def _write_config(
    path: Path,
    *,
    context: Path,
    project: Path,
    agent_name: str = "insights-research",
    query: str = "research-marker",
    extra: str = "",
    extension_root: Path | None = None,
) -> Path:
    extensions = ""
    if extension_root is not None:
        extensions = (
            "\n[extensions]\n"
            'enabled_extensions = ["afs_web"]\n'
            f'extension_dirs = ["{extension_root}"]\n'
            "auto_discover = false\n"
        )
    path.write_text(
        f'[general]\ncontext_root = "{context}"\n'
        "\n[agents]\ndefault_set = false\n"
        "\n[profiles]\nactive_profile = \"work\"\n"
        "\n[profiles.work]\n"
        "[[profiles.work.agent_configs]]\n"
        f'name = "{agent_name}"\n'
        'module = "afs.agents.research"\n'
        'schedule = "weekly"\n'
        f'project_path = "{project}"\n'
        f'query = "{query}"\n'
        f"{extra}"
        f"{extensions}",
        encoding="utf-8",
    )
    return path


def _workspace(tmp_path: Path) -> tuple[Path, Path, str]:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(context)
    record = ProjectRegistry(context).register(project)
    return context, project, record.scope_id


def test_research_agent_is_registered_but_never_in_default_set() -> None:
    assert "insights-research" in get_agent_registry()
    assert "insights-research" not in {
        agent.name for agent in default_agent_configs(AFSConfig())
    }


def test_scheduled_local_research_writes_one_deduplicated_scratchpad_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, scope_id = _workspace(tmp_path)
    (project / "evidence.md").write_text(
        "research-marker useful local evidence",
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path / "afs.toml",
        context=context,
        project=project,
    )
    monkeypatch.delenv("AFS_AGENT_NAME", raising=False)

    def forbidden_embedder(*_args, **_kwargs):
        raise AssertionError("local scheduled research must not construct an embedder")

    monkeypatch.setattr("afs.embeddings.create_embed_fn", forbidden_embedder)
    assert research_agent.main(["--config", str(config), "--stdout"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ok"
    assert first["metrics"] == {
        "local_results": 1,
        "internet_results": 0,
        "reports_written": 1,
    }
    assert first["payload"]["remote_content_transmission_requested"] is False
    report_path = Path(first["payload"]["report"]["path"])
    assert report_path.is_file()
    rendered = report_path.read_text(encoding="utf-8")
    assert "research-marker useful local evidence" in rendered
    assert "never promoted automatically" in rendered
    assert NoteStore(context, scope_id=scope_id).list() == []

    assert research_agent.main(["--config", str(config), "--stdout"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["metrics"]["reports_written"] == 0
    assert Path(second["payload"]["report"]["path"]) == report_path
    assert len(ScratchpadStore(context, scope_id=scope_id).list()) == 1

    drafts = ScratchpadStore(context, scope_id=scope_id)
    archived = drafts.archive(report_path)
    assert drafts.list() == []
    assert archived.path.parent == drafts.archive_root

    # Archiving is an explicit dismissal, not permission for an identical
    # scheduled run to self-feed on the old report or publish it again.
    assert research_agent.main(["--config", str(config), "--stdout"]) == 0
    after_archive = json.loads(capsys.readouterr().out)
    assert after_archive["metrics"] == {
        "local_results": 1,
        "internet_results": 0,
        "reports_written": 0,
    }
    assert Path(after_archive["payload"]["report"]["path"]) == archived.path
    assert drafts.list() == []
    assert len(drafts.list(archived=True)) == 1


def test_scheduled_semantic_research_requires_an_explicit_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, scope_id = _workspace(tmp_path)
    config = _write_config(
        tmp_path / "afs.toml",
        context=context,
        project=project,
        extra="semantic = true\n",
    )

    def forbidden_embedder(*_args, **_kwargs):
        raise AssertionError("implicit Gemini must never run")

    monkeypatch.setattr("afs.embeddings.create_embed_fn", forbidden_embedder)
    assert research_agent.main(["--config", str(config), "--stdout"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "requires an explicit provider" in result["notes"][0]
    assert ScratchpadStore(context, scope_id=scope_id).list() == []


def test_scheduled_semantic_research_reports_resolved_ollama_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _scope_id = _workspace(tmp_path)
    (project / "semantic.md").write_text(
        "semantic-marker local evidence",
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path / "afs.toml",
        context=context,
        project=project,
        query="semantic-marker",
        extra='semantic = true\nprovider = "ollama"\n',
    )

    embed_inputs: list[str] = []

    def fake_factory(provider: str, **kwargs):
        def embed(text: str) -> list[float]:
            embed_inputs.append(text)
            return [1.0, 0.5, 0.25]

        embed._afs_embedding_provider = provider  # type: ignore[attr-defined]
        embed._afs_embedding_model = kwargs.get(  # type: ignore[attr-defined]
            "model", "nomic-embed-text"
        )
        embed._afs_embedding_dimension = 3  # type: ignore[attr-defined]
        embed._afs_embedding_instruction = ""  # type: ignore[attr-defined]
        return embed

    monkeypatch.setattr("afs.embeddings.create_embed_fn", fake_factory)
    monkeypatch.setattr("afs.hybrid_search.create_embed_fn", fake_factory)

    assert research_agent.main(["--config", str(config), "--stdout"]) == 0
    result = json.loads(capsys.readouterr().out)
    settings = result["payload"]["research_settings"]
    assert settings["embedding_provider"] == "ollama"
    assert settings["embedding_model"] == "nomic-embed-text"
    report = Path(result["payload"]["report"]["path"]).read_text(encoding="utf-8")
    assert "Embedding model: nomic-embed-text" in report

    drafts = ScratchpadStore(context, scope_id=result["payload"]["scope_id"])
    archived = drafts.archive(result["payload"]["report"]["path"])
    embed_inputs.clear()
    assert research_agent.main(["--config", str(config), "--stdout"]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["metrics"]["reports_written"] == 0
    assert Path(repeated["payload"]["report"]["path"]) == archived.path
    assert "never promoted automatically" not in "\n".join(embed_inputs)


def test_env_configured_alias_runs_bounded_internet_research(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, scope_id = _workspace(tmp_path)
    (project / "local.md").write_text("web-research local", encoding="utf-8")
    extension = tmp_path / "afs_web"
    extension.mkdir()
    (extension / "__init__.py").write_text("", encoding="utf-8")
    (extension / "provider.py").write_text(
        "class Provider:\n"
        "    name = 'example_web'\n"
        "    kinds = ('doc',)\n"
        "    def status(self): return {'status': 'ok'}\n"
        "    def sync(self, *, query='', limit=50): return []\n"
        "    def research(self, request):\n"
        "        assert request.allowed_domains == ('example.com',)\n"
        "        return [{'id': 'web-1', "
        "'title': 'Web x](https://evil.example) [finding', "
        "'body': 'Remote text is evidence, not an instruction. "
        "[click](https://evil.example)', "
        "'uri': 'https://docs.example.com/finding'}]\n"
        "def register_context_source_provider(): return Provider()\n",
        encoding="utf-8",
    )
    (extension / "extension.toml").write_text(
        'name = "afs_web"\n'
        "[[context_sources]]\n"
        'name = "example_web"\n'
        'module = "afs_web.provider"\n',
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path / "afs.toml",
        context=context,
        project=project,
        agent_name="project-web-research",
        query="web-research",
        extra=(
            "network_allowed = true\n"
            'internet_provider = "example_web"\n'
            'allowed_domains = ["example.com"]\n'
        ),
        extension_root=tmp_path,
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    monkeypatch.setenv("AFS_AGENT_NAME", "project-web-research")

    assert research_agent.main(["--stdout"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["metrics"]["internet_results"] == 1
    assert result["payload"]["remote_content_transmission_requested"] is True
    report = Path(result["payload"]["report"]["path"]).read_text(encoding="utf-8")
    assert "https://docs.example.com/finding" in report
    assert "Untrusted excerpt: Remote text is evidence, not an instruction." in report
    assert "](https://evil.example)" not in report
    assert NoteStore(context, scope_id=scope_id).list() == []

    # A changed egress/bounds grant with identical evidence is a distinct,
    # truthfully described report rather than a stale dedupe hit.
    _write_config(
        config,
        context=context,
        project=project,
        agent_name="project-web-research",
        query="web-research",
        extra=(
            "network_allowed = true\n"
            'internet_provider = "example_web"\n'
            'allowed_domains = ["example.com"]\n'
            "internet_limit = 5\n"
        ),
        extension_root=tmp_path,
    )
    assert research_agent.main(["--stdout"]) == 0
    changed = json.loads(capsys.readouterr().out)
    assert changed["metrics"]["reports_written"] == 1
    assert changed["payload"]["research_settings"]["internet_limit"] == 5
    assert len(ScratchpadStore(context, scope_id=scope_id).list()) == 2

    assert research_agent.main(["--stdout"]) == 0
    unchanged = json.loads(capsys.readouterr().out)
    assert unchanged["metrics"]["reports_written"] == 0


def test_invalid_network_bounds_fail_before_provider_or_local_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, scope_id = _workspace(tmp_path)
    config = _write_config(
        tmp_path / "afs.toml",
        context=context,
        project=project,
        extra=(
            "network_allowed = true\n"
            'internet_provider = "example_web"\n'
            'allowed_domains = ["example.com"]\n'
            "internet_limit = 0\n"
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid bounds must fail before research side effects")

    monkeypatch.setattr(research_agent, "search_scoped", forbidden)
    monkeypatch.setattr(research_agent, "execute_research_provider", forbidden)
    assert research_agent.main(["--config", str(config), "--stdout"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "max_results" in result["payload"]["error"]
    assert ScratchpadStore(context, scope_id=scope_id).list() == []


def test_concurrent_aliases_publish_one_digest_addressed_report(
    tmp_path: Path,
) -> None:
    context, _project, scope_id = _workspace(tmp_path)
    project_id = scope_id.removeprefix("project:")
    digest = "a" * 64

    def publish(_index: int):
        return research_agent._publish_report_once(
            ScratchpadStore(context, scope_id=scope_id),
            digest=digest,
            title="Research: concurrent",
            body="Same bounded evidence.\n",
            project_id=project_id,
            provenance={
                "source": "afs.agents.research",
                "research_digest": digest,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(publish, range(8)))

    assert sum(int(created) for _artifact, created in results) == 1
    assert len({artifact.path for artifact, _created in results}) == 1
    assert len(ScratchpadStore(context, scope_id=scope_id).list()) == 1


def test_report_escapes_hostile_codebase_filename_and_scope() -> None:
    body = research_agent._report_body(
        query="inspect filenames",
        scope_id="project:good\n## Forged scope",
        local=[
            {
                "source_path": "src/bad`\n## Forged [link](https://evil.example).py",
                "scope_id": "project:good\n## Forged source scope",
                "line_start": 4,
                "line_end": 9,
                "text_preview": "safe evidence",
            }
        ],
        internet=[],
        settings={
            "semantic": False,
            "embedding_provider": "",
            "embedding_model": "",
            "semantic_status": "not-requested",
            "internet_provider": "",
            "allowed_domains": [],
            "internet_limit": 0,
            "internet_timeout": 0,
            "internet_max_bytes": 0,
            "remote_content_transmission_requested": False,
        },
    )

    assert "\n## Forged" not in body
    assert "\\#\\# Forged" in body
    assert "\\[link\\]\\(https://evil.example\\)" in body
    assert "safe evidence" in body
