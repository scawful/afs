from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.cli.core as cli_core_module
from afs.cli.core import session_pack_command
from afs.context_index import ContextSQLiteIndex
from afs.context_pack import (
    build_context_pack,
    render_context_pack,
    write_context_pack_artifacts,
)
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig, SensitivityConfig


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
        token_budget=900,
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
        token_budget=900,
    )

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
        token_budget=700,
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
        token_budget=700,
    )

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
        token_budget=700,
    )

    titles = [section["title"] for section in pack["sections"]]
    assert pack["pack_mode"] == "retrieval"
    assert "Indexed Hit 1" in titles
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
