from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from afs.agents import gemini_workspace_brief as agent
from afs.agents import list_agents
from afs.models import ContextRoot, ProjectMetadata
from afs.schema import AFSConfig, GeneralConfig, WorkspaceDirectory


class _FakeBackend:
    async def generate(self, messages, tools=None):  # noqa: ANN001
        self.messages = messages
        return SimpleNamespace(
            content="## Snapshot\n- workspace summary\n\n## Risks\n- none\n\n## Next Actions\n- review",
            usage={"prompt_tokens": 12, "completion_tokens": 8},
        )

    async def close(self) -> None:
        return None


def test_gemini_workspace_brief_is_registered() -> None:
    names = [spec.name for spec in list_agents()]
    assert "gemini-workspace-brief" in names


def test_gemini_workspace_brief_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "google"
    workspace_root.mkdir()
    config = AFSConfig(
        general=GeneralConfig(
            context_root=tmp_path / "context",
            agent_workspaces_dir=(tmp_path / "context" / "workspaces"),
            workspace_directories=[WorkspaceDirectory(path=workspace_root)],
        )
    )
    context = ContextRoot(
        path=workspace_root / "repo" / ".context",
        project_name="repo",
        metadata=ProjectMetadata(description="workspace repo"),
        mounts={},
    )
    json_output = tmp_path / "brief.json"
    markdown_output = tmp_path / "brief.md"
    backend = _FakeBackend()
    seen: dict[str, object] = {}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(agent, "load_agent_config", lambda _path: config)
    def _fake_resolve_contexts(config_arg, **kwargs):  # noqa: ANN001
        seen["config"] = config_arg
        seen["kwargs"] = kwargs
        return [context]

    monkeypatch.setattr(agent, "resolve_contexts", _fake_resolve_contexts)
    monkeypatch.setattr(agent, "create_backend", lambda _config: backend)

    exit_code = agent.main(
        [
            "--output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["payload"]["workspace_roots"] == [str(workspace_root)]
    assert "workspace summary" in payload["payload"]["brief_markdown"]
    assert "workspace summary" in markdown_output.read_text(encoding="utf-8")
    assert seen["config"] is config
    assert seen["kwargs"]["search_paths"] == [str(workspace_root)]
