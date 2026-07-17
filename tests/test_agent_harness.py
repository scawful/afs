from __future__ import annotations

import asyncio
from pathlib import Path

from afs.agent.harness import AgentHarness, HarnessConfig
from afs.agent.tools import Tool, ToolResult
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry


def test_default_harness_tools_bind_configured_v2_root_and_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configured_context = tmp_path / "configured-context"
    other_context = tmp_path / "other-context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(configured_context)
    scaffold_v2(other_context)
    configured_registry = ProjectRegistry(configured_context)
    alpha_record = configured_registry.register(alpha)
    beta_record = configured_registry.register(beta)
    ProjectRegistry(other_context).register(beta)
    beta_secret = (
        configured_context
        / "knowledge"
        / "projects"
        / beta_record.project_id
        / "secret.md"
    )
    beta_secret.parent.mkdir(parents=True)
    beta_secret.write_text("beta-only", encoding="utf-8")
    monkeypatch.chdir(beta)

    harness = AgentHarness(
        "ollama:test",
        config=HarnessConfig(
            context_root=configured_context,
            project_path=alpha,
        ),
    )
    written = asyncio.run(
        harness.tools["write_scratchpad"].execute(
            {"filename": "configured-root.md", "content": "alpha-only"}
        )
    )
    denied = asyncio.run(
        harness.tools["read_context"].execute(
            {"path": beta_secret.relative_to(configured_context).as_posix()}
        )
    )

    assert written.success is True
    assert Path(written.metadata["path"]).parent == (
        configured_context
        / "scratchpad"
        / "projects"
        / alpha_record.project_id
        / "notes"
    )
    assert denied.success is False
    assert "authorized project scope" in str(denied.error)
    assert not list((other_context / "scratchpad").rglob("*.md"))


def test_default_harness_project_path_is_bound_when_config_is_created(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = tmp_path / "context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    registry.register(beta)
    monkeypatch.chdir(alpha)
    config = HarnessConfig(context_root=context)
    monkeypatch.chdir(beta)

    harness = AgentHarness("ollama:test", config=config)
    result = asyncio.run(
        harness.tools["write_scratchpad"].execute(
            {"filename": "bound-project.md", "content": "alpha"}
        )
    )

    assert result.success is True
    assert Path(result.metadata["path"]).parent == (
        context / "scratchpad" / "projects" / alpha_record.project_id / "notes"
    )


def test_harness_preserves_empty_and_explicit_tool_lists() -> None:
    async def handler(_arguments) -> ToolResult:  # noqa: ANN001
        return ToolResult(success=True, content="ok")

    explicit = Tool(
        name="explicit",
        description="Explicit test tool",
        parameters={"type": "object"},
        handler=handler,
    )

    empty_harness = AgentHarness("ollama:test", tools=[])
    explicit_harness = AgentHarness("ollama:test", tools=[explicit])

    assert empty_harness.tools == {}
    assert explicit_harness.tools == {"explicit": explicit}
