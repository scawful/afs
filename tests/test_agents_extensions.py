from __future__ import annotations

from pathlib import Path

from afs.agents import get_agent, list_agents


def test_core_agents_include_claude_orchestrator() -> None:
    names = {spec.name for spec in list_agents()}
    assert "claude-orchestrator" in names


def test_extension_agent_modules_register_agents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extension_root = tmp_path / "extensions" / "agent_ext"
    package_root = extension_root / "agent_ext"
    package_root.mkdir(parents=True)

    (extension_root / "extension.toml").write_text(
        "name = \"agent_ext\"\n"
        "agent_modules = [\"agent_ext.agents\"]\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "agents.py").write_text(
        "def _run(_args=None):\n"
        "    return 0\n"
        "\n"
        "def register_agents():\n"
        "    return [\n"
        "        {\n"
        "            'name': 'demo-agent',\n"
        "            'description': 'extension-loaded agent',\n"
        "            'entrypoint': _run,\n"
        "        }\n"
        "    ]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_EXTENSION_DIRS", str(tmp_path / "extensions"))
    monkeypatch.setenv("AFS_ENABLED_EXTENSIONS", "agent_ext")

    names = {spec.name for spec in list_agents()}
    assert "demo-agent" in names
    assert get_agent("demo-agent") is not None
