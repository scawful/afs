from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import afs.cli.core as cli_core_module
from afs.cli.core import session_pack_command
from afs.context_index import ContextSQLiteIndex
from afs.context_pack import (
    _mount_fingerprint,
    _scoped_health_for_pack,
    build_context_pack,
    estimate_tokens,
    render_context_pack,
    write_context_pack_artifacts,
)
from afs.embeddings import build_embedding_index
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, ContextIndexConfig, GeneralConfig, SensitivityConfig


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        ),
        sensitivity=SensitivityConfig(never_export=["knowledge/private/*"]),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager


def _section_bodies(pack: dict) -> str:
    return "\n".join(section["body"] for section in pack["sections"])


def test_v2_pack_isolates_project_content_artifacts_and_caches(tmp_path: Path) -> None:
    from afs.context_layout import scaffold_v2
    from afs.handoff import HandoffStore
    from afs.hybrid_search import HybridSearchEngine, HybridSource
    from afs.messages import MessageBus
    from afs.project_registry import ProjectRegistry
    from afs.schema import SessionPackCacheConfig
    from afs.scratchpad import ScratchpadStore

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        session_pack_cache=SessionPackCacheConfig(cache_dir=tmp_path / "pack-cache"),
    )
    manager = AFSManager(config=config)

    for scope_id, marker in (
        ("common", "common-visible-marker"),
        (alpha_record.scope_id, "alpha-visible-marker"),
        (beta_record.scope_id, "beta-private-marker"),
    ):
        scope_dir = (
            Path("common")
            if scope_id == "common"
            else Path("projects") / scope_id.removeprefix("project:")
        )
        knowledge = context_root / "knowledge" / scope_dir
        knowledge.mkdir(parents=True)
        (knowledge / "scope.md").write_text(
            f"shared-query {marker}",
            encoding="utf-8",
        )

    ScratchpadStore(
        context_root,
        scope_id=alpha_record.scope_id,
        config=config,
    ).create(title="Alpha draft", body="alpha-visible-marker")
    ScratchpadStore(
        context_root,
        scope_id=beta_record.scope_id,
        config=config,
    ).create(title="Beta draft", body="beta-private-marker")
    HandoffStore(
        context_root,
        scope_id=alpha_record.scope_id,
        config=config,
    ).create_revision(title="Alpha handoff", agent_name="alpha")
    HandoffStore(
        context_root,
        scope_id=beta_record.scope_id,
        config=config,
    ).create_revision(title="Beta handoff", agent_name="beta")
    MessageBus(
        context_root,
        scope_id=alpha_record.scope_id,
        config=config,
    ).send("alpha", "status", {"marker": "alpha-visible-marker"})
    MessageBus(
        context_root,
        scope_id=beta_record.scope_id,
        config=config,
    ).send("beta", "status", {"marker": "beta-private-marker"})
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )
    HybridSearchEngine(context_root / ".afs" / "search").build(
        [
            HybridSource(
                context_root / "knowledge" / "common",
                scope_id="common",
            ),
            HybridSource(
                context_root / "knowledge" / "projects" / alpha_record.project_id,
                scope_id=alpha_record.scope_id,
                project_id=alpha_record.project_id,
            ),
            HybridSource(
                context_root / "knowledge" / "projects" / beta_record.project_id,
                scope_id=beta_record.scope_id,
                project_id=beta_record.project_id,
            ),
        ]
    )

    kwargs = {
        "query": "shared-query",
        "task": "Review the scoped context.",
        "model": "codex",
        "pack_mode": "full_slice",
        "token_budget": 4000,
        "include_content": True,
        "semantic": True,
    }
    alpha_pack = build_context_pack(
        manager,
        context_root,
        project_path=alpha,
        **kwargs,
    )
    alpha_text = json.dumps(alpha_pack)
    assert alpha_pack["scope_id"] == alpha_record.scope_id
    assert "alpha-visible-marker" in alpha_text
    assert "common-visible-marker" in alpha_text
    assert "beta-private-marker" not in alpha_text
    assert "Beta draft" not in alpha_text
    assert "Beta handoff" not in alpha_text
    assert "Recent Messages" in {section["title"] for section in alpha_pack["sections"]}
    hybrid_sources = {
        source
        for section in alpha_pack["sections"]
        if section["title"] == "Indexed Text Hits"
        for source in section["sources"]
    }
    assert hybrid_sources
    assert all(beta_record.project_id not in source for source in hybrid_sources)

    alpha_artifacts = write_context_pack_artifacts(manager, context_root, alpha_pack)
    beta_pack = build_context_pack(
        manager,
        context_root,
        project_path=beta,
        **kwargs,
    )
    beta_artifacts = write_context_pack_artifacts(manager, context_root, beta_pack)
    assert alpha_pack["cache"]["key"] != beta_pack["cache"]["key"]
    assert alpha_artifacts["json"] != beta_artifacts["json"]
    assert alpha_record.project_id in alpha_artifacts["json"]
    assert beta_record.project_id in beta_artifacts["json"]
    (
        context_root
        / "knowledge"
        / "projects"
        / beta_record.project_id
        / "scope.md"
    ).write_text("shared-query beta-changed-marker", encoding="utf-8")

    cached_alpha = build_context_pack(
        manager,
        context_root,
        project_path=alpha,
        **kwargs,
    )
    assert cached_alpha["cache"]["hit"] is True
    assert cached_alpha["scope_id"] == alpha_record.scope_id
    assert "beta-private-marker" not in json.dumps(cached_alpha)


def test_v2_pack_rejects_hybrid_generation_for_another_project(
    tmp_path: Path,
) -> None:
    from afs.context_layout import scaffold_v2
    from afs.hybrid_search import (
        HybridScopeCoverageError,
        HybridSearchEngine,
        HybridSource,
    )
    from afs.project_registry import ProjectRegistry

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    registry.register(alpha)
    beta_record = registry.register(beta)
    beta_knowledge = (
        context_root / "knowledge" / "projects" / beta_record.project_id
    )
    beta_knowledge.mkdir(parents=True)
    (beta_knowledge / "note.md").write_text("beta-only marker", encoding="utf-8")
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    HybridSearchEngine(context_root / ".afs" / "search").build(
        [
            HybridSource(
                beta_knowledge,
                scope_id=beta_record.scope_id,
                project_id=beta_record.project_id,
            )
        ]
    )

    with pytest.raises(HybridScopeCoverageError, match="rebuild the index"):
        build_context_pack(
            manager,
            context_root,
            project_path=alpha,
            query="marker",
            semantic=True,
        )


def test_build_context_pack_respects_sensitivity_and_budget(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root

    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text("fix service wiring", encoding="utf-8")

    memory_root = manager.resolve_mount_root(context_root, MountType.MEMORY)
    summary_root = memory_root / "history_consolidation"
    summary_root.mkdir(parents=True, exist_ok=True)
    (memory_root / "entries.jsonl").write_text(json.dumps({"id": "mem-1"}) + "\n", encoding="utf-8")
    (summary_root / "mem-1.md").write_text("previous durable memory summary", encoding="utf-8")

    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "public").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public" / "guide.md").write_text(
        "service wiring guide for agents",
        encoding="utf-8",
    )
    (knowledge_root / "private" / "secret.md").write_text(
        "private service details",
        encoding="utf-8",
    )

    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.KNOWLEDGE, MountType.SCRATCHPAD], include_content=True)

    pack = build_context_pack(
        manager,
        context_root,
        query="service wiring",
        task="Fix the service wiring issue with minimal edits.",
        model="gemini",
        workflow="edit_fast",
        token_budget=900,
        include_content=True,
    )

    assert pack["model"] == "gemini"
    assert pack["task"] == "Fix the service wiring issue with minimal edits."
    assert pack["execution_profile"]["workflow"] == "edit_fast"
    assert pack["execution_profile"]["loop_policy"].startswith("Prompt-only rail.")
    assert "Retry on Flash for a narrower edit" in pack["execution_profile"]["retry_hint"]
    assert len(pack["execution_profile"]["retry_contract"]) == 2
    assert pack["execution_profile"]["tool_profile"]["name"] == "edit_and_verify"
    assert pack["estimated_tokens"] <= 900
    assert any(section["title"] == "Scratchpad State" for section in pack["sections"])
    assert any("guide.md" in source for source in pack["sources"])
    assert all("secret.md" not in source for source in pack["sources"])


def test_full_slice_knowledge_inventory_respects_mount_prefixed_never_export(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "public").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public" / "guide.md").write_text("public guide", encoding="utf-8")
    (knowledge_root / "private" / "secret.md").write_text("private guide", encoding="utf-8")

    pack = build_context_pack(
        manager,
        context_root,
        task="Build a full slice.",
        model="codex",
        pack_mode="full_slice",
        token_budget=1200,
    )

    rendered = render_context_pack(pack)
    assert "public/guide.md" in rendered
    assert "private/secret.md" not in rendered


def test_query_hits_overfetch_after_sensitive_filtering(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "public").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public" / "guide.md").write_text(
        "shared-token public guide",
        encoding="utf-8",
    )
    (knowledge_root / "private" / "secret.md").write_text(
        "shared-token private guide",
        encoding="utf-8",
    )
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    pack = build_context_pack(
        manager,
        context_root,
        query="shared-token",
        task="Find the visible guide.",
        model="codex",
        token_budget=1200,
        include_content=True,
        max_query_results=1,
    )

    rendered = render_context_pack(pack)
    assert "public/guide.md" in rendered
    assert "private/secret.md" not in rendered


def test_embedding_hits_respect_never_embed_rules(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            context_index=ContextIndexConfig(enabled=False),
            sensitivity=SensitivityConfig(never_embed=["private/*"]),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "public").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public" / "guide.md").write_text(
        "embed-only public marker",
        encoding="utf-8",
    )
    (knowledge_root / "private" / "secret.md").write_text(
        "embed-only secret marker",
        encoding="utf-8",
    )
    build_embedding_index([knowledge_root], knowledge_root)

    pack = build_context_pack(
        manager,
        context_root,
        query="embed-only secret",
        task="Check embedding sensitivity.",
        model="codex",
        token_budget=1400,
        semantic=True,
        max_embedding_results=1,
    )

    rendered = render_context_pack(pack)
    assert "Indexed Text Hits" in rendered
    assert "embed-only public marker" in rendered
    assert "embed-only secret marker" not in rendered
    assert "private/secret.md" not in rendered


def test_embedding_hits_trim_previews_for_tight_packs(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            context_index=ContextIndexConfig(enabled=False),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "long.md").write_text(
        "embed-long " + ("x " * 450) + "TAIL_MARKER",
        encoding="utf-8",
    )
    build_embedding_index([knowledge_root], knowledge_root)

    pack = build_context_pack(
        manager,
        context_root,
        query="embed-long",
        model="codex",
        pack_mode="retrieval",
        token_budget=1200,
        semantic=True,
        max_embedding_results=1,
    )

    rendered = render_context_pack(pack)
    assert "Indexed Text Hits" in rendered
    assert "TAIL_MARKER" not in rendered
    assert pack["estimated_tokens"] <= 1200


def test_context_pack_never_enters_semantic_path_without_explicit_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _make_manager(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic retrieval requires explicit consent")

    monkeypatch.setattr("afs.context_pack._embedding_section", forbidden)
    pack = build_context_pack(
        manager,
        manager.config.general.context_root,
        query="local-only query",
        model="codex",
    )

    assert pack["semantic"] is False


def test_session_pack_command_outputs_json_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "guide.md").write_text("codex pack guide", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    monkeypatch.setattr(cli_core_module, "load_manager", lambda _config_path=None: manager)

    exit_code = session_pack_command(
        Namespace(
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
            query="codex pack",
            task="Implement the codex pack update.",
            model="codex",
            workflow="edit_fast",
            tool_profile="edit_and_verify",
            pack_mode="focused",
            token_budget=400,
            include_content=False,
            max_query_results=4,
            max_embedding_results=2,
            no_write_artifacts=False,
            json=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "codex"
    assert payload["task"] == "Implement the codex pack update."
    assert payload["execution_profile"]["workflow"] == "edit_fast"
    assert payload["pack_mode"] == "focused"
    assert payload["cache"]["prefix_hash"]
    assert payload["artifact_paths"]["json"].endswith("session_pack_codex.json")
    assert Path(payload["artifact_paths"]["json"]).exists()
    assert Path(payload["artifact_paths"]["markdown"]).exists()


def test_session_pack_command_skips_artifact_rewrite_on_cache_hit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text("cached command state", encoding="utf-8")

    monkeypatch.setattr(cli_core_module, "load_manager", lambda _config_path=None: manager)

    args = Namespace(
        config=None,
        path=None,
        context_root=context_root,
        context_dir=None,
        query="cached command",
        task="Keep the cached command pack stable.",
        model="codex",
        workflow="general",
        tool_profile="default",
        pack_mode="focused",
        token_budget=400,
        include_content=False,
        max_query_results=4,
        max_embedding_results=2,
        no_write_artifacts=False,
        json=True,
    )

    exit_code = session_pack_command(args)
    assert exit_code == 0
    first = json.loads(capsys.readouterr().out)
    assert first["cache"]["hit"] is False

    marker = Path(first["artifact_paths"]["json"])
    assert marker.exists()

    monkeypatch.setattr(
        "afs.context_pack.write_context_pack_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit should not rewrite artifacts")),
    )

    exit_code = session_pack_command(args)
    assert exit_code == 0
    second = json.loads(capsys.readouterr().out)
    assert second["cache"]["hit"] is True
    assert Path(second["artifact_paths"]["json"]) == marker


def test_build_context_pack_reuses_cached_artifact_when_inputs_match(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text("cached pack state", encoding="utf-8")

    first_pack = build_context_pack(
        manager,
        context_root,
        query="cached pack",
        task="Keep the cached pack stable.",
        model="codex",
        workflow="general",
        token_budget=400,
    )
    write_context_pack_artifacts(manager, context_root, first_pack)

    monkeypatch.setattr(
        "afs.context_pack._build_sections",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache should avoid rebuild")),
    )

    cached_pack = build_context_pack(
        manager,
        context_root,
        query="cached pack",
        task="Keep the cached pack stable.",
        model="codex",
        workflow="general",
        token_budget=400,
    )

    assert cached_pack["cache"]["hit"] is True
    assert cached_pack["artifact_paths"]["json"].endswith("session_pack_codex.json")


def test_build_context_pack_invalidates_cache_when_bootstrap_changes(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    state_path = scratchpad_root / "state.md"
    state_path.write_text("initial state", encoding="utf-8")

    first_pack = build_context_pack(
        manager,
        context_root,
        query="state drift",
        task="Investigate state drift.",
        model="codex",
    )
    write_context_pack_artifacts(manager, context_root, first_pack)

    state_path.write_text("changed state", encoding="utf-8")

    second_pack = build_context_pack(
        manager,
        context_root,
        query="state drift",
        task="Investigate state drift.",
        model="codex",
    )

    assert second_pack["cache"]["hit"] is False


def test_write_context_pack_artifacts_writes_files(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    pack = {
        "context_path": str(context_root),
        "project": "project",
        "profile": "default",
        "model": "generic",
        "query": "",
        "task": "Summarize the pack",
        "token_budget": 100,
        "estimated_tokens": 10,
        "guidance": "use this pack",
        "execution_profile": {
            "workflow": "general",
            "summary": "Balanced workflow when the task is not classified yet.",
            "intent": "Read the cited context, keep the plan flat, and move to action without over-scaffolding.",
            "model_hint": "Use the model's default reasoning level and escalate only when the evidence stays ambiguous.",
            "loop_policy": "Prompt-only rail.",
            "retry_hint": "Retry with a narrower query or smaller pack before escalating the workflow.",
            "prompt_contract": [],
            "verification_contract": [],
            "retry_contract": [],
            "tool_profile": {
                "name": "default",
                "summary": "Balanced AFS surface for normal repo work.",
                "preferred_surfaces": [],
                "notes": [],
            },
        },
        "sections": [],
        "sources": [],
        "omitted_sections": [],
    }

    artifact_paths = write_context_pack_artifacts(manager, context_root, pack)

    assert Path(artifact_paths["json"]).exists()
    assert Path(artifact_paths["markdown"]).exists()


def test_rendered_context_pack_places_task_at_end(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    pack = build_context_pack(
        manager,
        context_root,
        query="service guide",
        task="Review the service guide and propose the smallest fix.",
        model="gemini",
        workflow="review_deep",
        tool_profile="context_readonly",
        token_budget=400,
    )

    rendered = render_context_pack(pack)
    assert "## Task" in rendered
    assert "Loop policy:" in rendered
    assert "Retry contract:" in rendered
    assert "afs query <text> --path <workspace>" in rendered
    assert "afs index rebuild --path <workspace>" in rendered
    assert rendered.rstrip().endswith("Review the service guide and propose the smallest fix.")


def test_context_pack_renders_guidance_once_and_counts_it(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root

    pack = build_context_pack(
        manager,
        context_root,
        query="service guide",
        task="Review the service guide.",
        model="codex",
        token_budget=500,
    )

    rendered = render_context_pack(pack)
    guidance_tokens = estimate_tokens("Guidance") + estimate_tokens(pack["guidance"])
    section_tokens = sum(section["estimated_tokens"] for section in pack["sections"])

    assert rendered.count("Treat all pack sections") == 1
    assert all(section["title"] != "Model Usage Notes" for section in pack["sections"])
    assert pack["estimated_tokens"] >= section_tokens + guidance_tokens


def test_context_pack_prefix_hash_stays_stable_when_only_task_changes(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "guide.md").write_text("stable prefix guide", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    first = build_context_pack(
        manager,
        context_root,
        query="stable prefix",
        task="First task wording.",
        model="gemini",
        workflow="edit_fast",
        token_budget=700,
    )
    second = build_context_pack(
        manager,
        context_root,
        query="stable prefix",
        task="Second task wording.",
        model="gemini",
        workflow="edit_fast",
        token_budget=700,
    )

    assert first["cache"]["prefix_hash"] == second["cache"]["prefix_hash"]
    assert first["cache"]["stable_prefix_hash"] == second["cache"]["stable_prefix_hash"]


def test_stable_prefix_hash_ignores_scratchpad_and_task_drift(tmp_path: Path) -> None:
    """stable_prefix_hash stays the same even when scratchpad content changes."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "ref.md").write_text("stable reference doc", encoding="utf-8")
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text("state version A", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE, MountType.SCRATCHPAD],
        include_content=True,
    )

    first = build_context_pack(
        manager,
        context_root,
        query="stable reference",
        task="Task A",
        model="gemini",
        workflow="edit_fast",
        token_budget=4000,
    )

    # Change scratchpad content
    (scratchpad_root / "state.md").write_text("state version B", encoding="utf-8")

    second = build_context_pack(
        manager,
        context_root,
        query="stable reference",
        task="Task B",
        model="gemini",
        workflow="edit_fast",
        token_budget=4000,
    )

    # Section cost includes absolute paths, so a tight budget under a deep
    # tmp_path (e.g. pytest-xdist basetemp) can evict the knowledge section
    # and make this equality vacuous. Assert the premise explicitly.
    assert "stable reference doc" in _section_bodies(first)
    assert "stable reference doc" in _section_bodies(second)

    # stable_prefix_hash should match (scratchpad is volatile, excluded)
    assert first["cache"]["stable_prefix_hash"] == second["cache"]["stable_prefix_hash"]
    # But full prefix_hash may differ because it includes scratchpad
    # (we don't assert inequality since both *might* match if scratchpad
    # happens to render identically, but the stable hash is the guarantee)


def test_stable_prefix_hash_changes_when_knowledge_changes(tmp_path: Path) -> None:
    """stable_prefix_hash must change when knowledge docs change."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "ref.md").write_text("knowledge version A", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    first = build_context_pack(
        manager,
        context_root,
        query="knowledge",
        model="gemini",
        token_budget=4000,
    )

    (knowledge_root / "ref.md").write_text("knowledge version B", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    second = build_context_pack(
        manager,
        context_root,
        query="knowledge",
        model="gemini",
        token_budget=4000,
    )

    # Guard the premise: if the budget ever evicts the knowledge section
    # (its cost includes the absolute file path, which grows under deep
    # tmp_path roots), both packs render identically and the hash
    # comparison below fails without naming the real cause.
    assert "knowledge version A" in _section_bodies(first)
    assert "knowledge version B" in _section_bodies(second)

    assert first["cache"]["stable_prefix_hash"] != second["cache"]["stable_prefix_hash"]


def test_retrieval_pack_mode_prioritizes_indexed_hits(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text("scratch context", encoding="utf-8")
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "guide.md").write_text("sprite service guide", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE, MountType.SCRATCHPAD],
        include_content=True,
    )

    pack = build_context_pack(
        manager,
        context_root,
        query="service guide",
        model="gemini",
        pack_mode="retrieval",
        token_budget=1400,
    )

    titles = [section["title"] for section in pack["sections"]]
    assert pack["pack_mode"] == "retrieval"
    assert "Indexed Hit 1" in titles
    assert "Scratchpad State" in titles
    assert titles.index("Indexed Hit 1") < titles.index("Scratchpad State")


def test_full_slice_pack_mode_adds_knowledge_slice_without_query(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "guide.md").write_text("# Guide\n\nKnowledge slice content", encoding="utf-8")

    pack = build_context_pack(
        manager,
        context_root,
        model="gemini",
        pack_mode="full_slice",
        token_budget=900,
    )

    titles = [section["title"] for section in pack["sections"]]
    assert pack["pack_mode"] == "full_slice"
    assert "Knowledge Slice" in titles
    assert "Broader long-context slice" in pack["pack_mode_summary"]


def test_v2_full_slice_skips_linked_cross_project_index(tmp_path: Path) -> None:
    from afs.context_layout import scaffold_v2
    from afs.project_registry import ProjectRegistry

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    alpha_root = context_root / "knowledge" / "projects" / alpha_record.project_id
    beta_root = context_root / "knowledge" / "projects" / beta_record.project_id
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    (alpha_root / "safe.md").write_text("alpha-safe-slice", encoding="utf-8")
    beta_index = beta_root / "INDEX.md"
    beta_index.write_text("beta-private-slice", encoding="utf-8")
    try:
        (alpha_root / "INDEX.md").symlink_to(beta_index)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    pack = build_context_pack(
        manager,
        context_root,
        project_path=alpha,
        pack_mode="full_slice",
        token_budget=1600,
    )

    rendered = render_context_pack(pack)
    assert "alpha-safe-slice" not in rendered  # non-INDEX files remain metadata-only
    assert "safe.md" in rendered
    assert "beta-private-slice" not in rendered


def test_v2_pack_fingerprint_and_health_skip_linked_descendants(
    tmp_path: Path,
) -> None:
    import os

    from afs.context_layout import scaffold_v2
    from afs.project_registry import ProjectRegistry
    from afs.scopes import resolve_scope

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    outside = tmp_path / "outside"
    alpha.mkdir()
    beta.mkdir()
    outside.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    alpha_root = context_root / "knowledge" / "projects" / alpha_record.project_id
    beta_root = context_root / "knowledge" / "projects" / beta_record.project_id
    common_root = context_root / "knowledge" / "common"
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    common_root.mkdir(parents=True, exist_ok=True)
    (alpha_root / "safe.md").write_text("alpha safe", encoding="utf-8")
    (common_root / "shared.md").write_text("common safe", encoding="utf-8")
    beta_secret = beta_root / "secret.md"
    beta_secret.write_text("beta private", encoding="utf-8")
    outside_secret = outside / "secret.md"
    outside_secret.write_text("outside private", encoding="utf-8")
    try:
        (alpha_root / "linked-file.md").symlink_to(beta_secret)
        (alpha_root / "linked-dir").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    scope = resolve_scope(context_root, requester_path=alpha)
    before = _mount_fingerprint(context_root, config=manager.config, scoped=scope)
    status, _diff = _scoped_health_for_pack(
        manager,
        context_root,
        status={"mount_counts": {"knowledge": 99}, "total_files": 99},
        diff={"added": [], "modified": [], "deleted": []},
        scoped=scope,
    )
    assert status["mount_counts"]["knowledge"] == 2
    assert status["total_files"] == 2

    future = beta_secret.stat().st_mtime + 10
    beta_secret.write_text("beta private changed", encoding="utf-8")
    outside_secret.write_text("outside private changed", encoding="utf-8")
    os.utime(beta_secret, (future, future))
    os.utime(outside_secret, (future, future))
    after = _mount_fingerprint(context_root, config=manager.config, scoped=scope)
    assert after == before


def test_v2_pack_health_counts_complete_scoped_index_over_500_rows(
    tmp_path: Path,
) -> None:
    from afs.context_layout import scaffold_v2
    from afs.project_registry import ProjectRegistry
    from afs.scopes import resolve_scope

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    alpha_root = context_root / "knowledge" / "projects" / alpha_record.project_id
    beta_root = context_root / "knowledge" / "projects" / beta_record.project_id
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    for index in range(501):
        (alpha_root / f"note-{index:03d}.md").write_text("alpha", encoding="utf-8")
    (beta_root / "private.md").write_text("beta", encoding="utf-8")

    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    context_index = ContextSQLiteIndex(manager, context_root)
    context_index.rebuild(mount_types=[MountType.KNOWLEDGE])
    scope = resolve_scope(context_root, requester_path=alpha)
    expected = context_index.count_entries_scoped(scope)
    assert expected > 500
    assert expected < context_index.total_entries

    status, _diff = _scoped_health_for_pack(
        manager,
        context_root,
        status={"index": {"enabled": True, "total_entries": 9999}},
        diff={"added": [], "modified": [], "deleted": []},
        scoped=scope,
    )

    assert status["index"]["total_entries"] == expected


def test_v2_pack_auto_index_never_traverses_another_project(
    monkeypatch, tmp_path: Path
) -> None:
    from afs.context_layout import scaffold_v2
    from afs.project_registry import ProjectRegistry

    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    knowledge = context_root / "knowledge"
    alpha_root = knowledge / "projects" / alpha_record.project_id
    beta_root = knowledge / "projects" / beta_record.project_id
    common_root = knowledge / "common"
    for root, marker in (
        (alpha_root, "pack-auto-index alpha-visible"),
        (beta_root, "pack-auto-index beta-private"),
        (common_root, "pack-auto-index common-visible"),
    ):
        root.mkdir(parents=True)
        (root / "note.md").write_text(marker, encoding="utf-8")

    real_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == beta_root:
            raise AssertionError("beta scope was traversed")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    pack = build_context_pack(
        manager,
        context_root,
        project_path=alpha,
        query="pack-auto-index",
        task="Build a scoped pack",
        pack_mode="full_slice",
        token_budget=2000,
        include_content=True,
    )

    rendered = json.dumps(pack)
    assert "alpha-visible" in rendered
    assert "common-visible" in rendered
    assert "beta-private" not in rendered
