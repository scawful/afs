from __future__ import annotations

from pathlib import Path

from afs.agent_manifest import export_for_harness, load_manifest, validate_manifest
from afs.diagnostics import check_agent_manifest


def test_default_agent_manifest_validates() -> None:
    data = load_manifest(Path("configs/agent_manifest.toml"))
    issues = validate_manifest(data)
    assert not [issue for issue in issues if issue.level == "error"]


def test_agent_manifest_exports_harness_slice() -> None:
    data = load_manifest(Path("configs/agent_manifest.toml"))
    payload = export_for_harness(data, "codex")
    assert payload["harness"]["name"] == "codex"
    assert "paths" in payload
    assert any(skill["name"] == "agentic-context" for skill in payload["skills"])
    assert payload["slash_command_packs"] == []
    assert any(server["name"] == "afs" for server in payload["mcp_servers"])


def test_agent_manifest_exports_hcode_slash_commands() -> None:
    data = load_manifest(Path("configs/agent_manifest.toml"))
    payload = export_for_harness(data, "hcode")
    assert payload["harness"]["name"] == "hcode"
    assert any(pack["name"] == "afs-opencode" for pack in payload["slash_command_packs"])
    assert payload["harness"]["command_roots"] == []


def test_agent_manifest_resolves_repo_paths_from_manifest_location(tmp_path: Path) -> None:
    config_dir = tmp_path / "portable" / "configs"
    command_dir = tmp_path / "portable" / "commands"
    config_dir.mkdir(parents=True)
    command_dir.mkdir(parents=True)
    manifest = config_dir / "agent_manifest.toml"
    manifest.write_text(
        """
version = 1
[paths]
afs_root = ".."
[[harnesses]]
name = "hcode"
kind = "cli"
instructions = []
skill_roots = []
command_roots = []
mcp_servers = []
startup = ["hcode"]
manifest_exports = []
[[slash_command_packs]]
name = "portable"
canonical_path = "../commands"
targets = ["hcode"]
""",
        encoding="utf-8",
    )

    data = load_manifest(manifest)

    assert data["paths"]["afs_root"] == str((tmp_path / "portable").resolve())
    assert data["slash_command_packs"][0]["canonical_path"] == str(command_dir.resolve())


def test_doctor_agent_manifest_check_accepts_synced_skill(tmp_path: Path, monkeypatch) -> None:
    canonical = tmp_path / "codex" / "sample-skill"
    copied_root = tmp_path / "claude-skills"
    copied = copied_root / "sample-skill"
    canonical.mkdir(parents=True)
    copied.mkdir(parents=True)
    skill_text = "---\nname: sample-skill\ndescription: sample\n---\n# Sample\n"
    (canonical / "SKILL.md").write_text(skill_text, encoding="utf-8")
    (copied / "SKILL.md").write_text(skill_text, encoding="utf-8")

    manifest = tmp_path / "agent_manifest.toml"
    manifest.write_text(
        f"""
version = 1

[paths]
workspace_root = "{tmp_path}"

[[harnesses]]
name = "claude"
kind = "cli"
skill_roots = ["{copied_root}"]
instructions = []
mcp_servers = []
startup = []

[[skills]]
name = "sample-skill"
canonical_path = "{canonical}"
targets = ["claude"]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_AGENT_MANIFEST", str(manifest))

    result = check_agent_manifest()

    assert result.status == "ok"


def test_default_agent_manifest_stays_domain_neutral() -> None:
    data = load_manifest(Path("configs/agent_manifest.toml"))
    harness_names = {harness.get("name") for harness in data.get("harnesses", [])}
    server_names = {server.get("name") for server in data.get("mcp_servers", [])}
    harness_server_names = {
        name for harness in data.get("harnesses", []) for name in harness.get("mcp_servers", [])
    }

    assert "z3cli" not in harness_names
    forbidden = {
        "hyrule-historian",
        "book-of-mudora",
        "yaze-mcp",
        "yaze-debugger",
        "yaze-editor",
    }
    assert server_names.isdisjoint(forbidden)
    assert harness_server_names.isdisjoint(forbidden)
