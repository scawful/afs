from __future__ import annotations

import asyncio
from pathlib import Path

from afs.agent.tools import read_context_handler, write_scratchpad_handler
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry


def test_write_scratchpad_creates_unique_readable_v2_drafts(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(context_root)
    record = ProjectRegistry(context_root).register(project)
    arguments = {
        "context_root": str(context_root),
        "_requester_path": str(project),
        "filename": "reactor-design.md",
        "content": "first draft",
        "agent_name": "reviewer",
    }

    first = asyncio.run(write_scratchpad_handler(dict(arguments)))
    second = asyncio.run(write_scratchpad_handler(dict(arguments)))

    assert first.success is True
    assert second.success is True
    first_path = Path(first.metadata["path"])
    second_path = Path(second.metadata["path"])
    assert first_path != second_path
    assert "reactor-design" in first_path.name
    assert "reactor-design" in second_path.name
    assert first_path.parent == (
        context_root / "scratchpad" / "projects" / record.project_id / "notes"
    )
    assert first_path.read_text(encoding="utf-8").endswith("first draft\n")
    assert not (context_root / "scratchpad" / "reactor-design.md").exists()


def test_write_scratchpad_rejects_nested_filename(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)

    result = asyncio.run(
        write_scratchpad_handler(
            {
                "context_root": str(context_root),
                "filename": "../outside.md",
                "content": "escape",
            }
        )
    )

    assert result.success is False
    assert "contained basename" in str(result.error)
    assert not (tmp_path / "outside.md").exists()


def test_read_context_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    evil = tmp_path / ".context-evil"
    evil.mkdir()
    (evil / "secret.md").write_text("outside-secret", encoding="utf-8")

    result = asyncio.run(
        read_context_handler(
            {
                "context_root": str(context_root),
                "path": "../.context-evil/secret.md",
            }
        )
    )

    assert result.success is False
    assert "escapes context root" in str(result.error)
    assert "outside-secret" not in result.content


def test_read_context_v2_allows_current_and_common_but_not_other_project(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    paths = {
        "alpha": context_root
        / "knowledge"
        / "projects"
        / alpha_record.project_id
        / "note.md",
        "beta": context_root
        / "knowledge"
        / "projects"
        / beta_record.project_id
        / "secret.md",
        "common": context_root / "knowledge" / "common" / "shared.md",
        "system": context_root / ".afs" / "private.md",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}-marker", encoding="utf-8")

    def read(path: Path):
        return asyncio.run(
            read_context_handler(
                {
                    "context_root": str(context_root),
                    "_requester_path": str(alpha),
                    "path": path.relative_to(context_root).as_posix(),
                }
            )
        )

    assert read(paths["alpha"]).content == "alpha-marker"
    assert read(paths["common"]).content == "common-marker"
    beta_result = read(paths["beta"])
    system_result = read(paths["system"])
    assert beta_result.success is False
    assert system_result.success is False
    assert "beta-marker" not in beta_result.content
    assert "system-marker" not in system_result.content


def test_v2_agent_tools_fail_closed_without_registered_requester(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    common = context_root / "knowledge" / "common" / "shared.md"
    common.parent.mkdir(parents=True, exist_ok=True)
    common.write_text("common-marker", encoding="utf-8")

    read_result = asyncio.run(
        read_context_handler(
            {
                "context_root": str(context_root),
                "_requester_path": str(tmp_path / "unregistered"),
                "path": "knowledge/common/shared.md",
            }
        )
    )
    write_result = asyncio.run(
        write_scratchpad_handler(
            {
                "context_root": str(context_root),
                "_requester_path": str(tmp_path / "unregistered"),
                "filename": "draft.md",
                "content": "must not leak to common",
            }
        )
    )

    assert read_result.success is False
    assert write_result.success is False
    assert not list((context_root / "scratchpad" / "common" / "notes").glob("*.md"))
