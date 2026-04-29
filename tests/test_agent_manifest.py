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
    assert any(skill["name"] == "focused-verification" for skill in payload["skills"])
    assert any(server["name"] == "afs" for server in payload["mcp_servers"])


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
        name
        for harness in data.get("harnesses", [])
        for name in harness.get("mcp_servers", [])
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
