from __future__ import annotations

import json
from pathlib import Path

from afs.agent_manifest_sync import sync_manifest


def test_agent_manifest_sync_copies_skills_and_writes_exports(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical" / "focused-verification"
    canonical.mkdir(parents=True)
    (canonical / "SKILL.md").write_text(
        "---\nname: focused-verification\ndescription: verify\n---\n# Verify\n",
        encoding="utf-8",
    )
    skill_root = tmp_path / "claude-skills"
    export_path = tmp_path / "claude-export.json"
    data = {
        "version": 1,
        "paths": {},
        "harnesses": [
            {
                "name": "claude",
                "skill_roots": [str(skill_root)],
                "mcp_servers": ["afs"],
                "manifest_exports": [str(export_path)],
            }
        ],
        "skills": [
            {
                "name": "focused-verification",
                "canonical_path": str(canonical),
                "targets": ["claude"],
            }
        ],
        "mcp_servers": [{"name": "afs"}],
    }

    dry_run = sync_manifest(data)
    assert {action.status for action in dry_run} == {"would_create"}
    assert not (skill_root / "focused-verification" / "SKILL.md").exists()

    applied = sync_manifest(data, apply=True)
    assert {action.status for action in applied} == {"synced"}
    assert (skill_root / "focused-verification" / "SKILL.md").read_text(encoding="utf-8").startswith("---")
    export_payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert export_payload["harness"]["name"] == "claude"
    assert export_payload["generated_by"] == "afs agent-manifest sync"

    current = sync_manifest(data)
    assert {action.status for action in current} == {"up_to_date"}


def test_agent_manifest_sync_replaces_skill_symlink_with_copy(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical" / "handoff-writer"
    canonical.mkdir(parents=True)
    (canonical / "SKILL.md").write_text("---\nname: handoff-writer\n---\n# Handoff\n", encoding="utf-8")
    linked_target = tmp_path / "linked"
    linked_target.mkdir()
    skill_root = tmp_path / "gemini-skills"
    skill_root.mkdir()
    (skill_root / "handoff-writer").symlink_to(linked_target, target_is_directory=True)
    data = {
        "version": 1,
        "paths": {},
        "harnesses": [{"name": "gemini", "skill_roots": [str(skill_root)], "mcp_servers": []}],
        "skills": [
            {
                "name": "handoff-writer",
                "canonical_path": str(canonical),
                "targets": ["gemini"],
            }
        ],
        "mcp_servers": [],
    }

    actions = sync_manifest(data)
    assert actions[0].status == "would_replace_symlink"

    sync_manifest(data, apply=True)
    copied = skill_root / "handoff-writer"
    assert not copied.is_symlink()
    assert (copied / "SKILL.md").exists()


def test_agent_manifest_sync_copies_slash_command_pack(tmp_path: Path) -> None:
    canonical = tmp_path / "commands"
    canonical.mkdir()
    (canonical / "afs-status.md").write_text(
        "---\ndescription: status\n---\n\nUse context.status.\n",
        encoding="utf-8",
    )
    root = tmp_path / "opencode" / "command"
    data = {
        "version": 1,
        "paths": {},
        "harnesses": [{"name": "hcode", "command_roots": [str(root)], "mcp_servers": []}],
        "slash_command_packs": [
            {
                "name": "afs-opencode",
                "canonical_path": str(canonical),
                "targets": ["hcode"],
            }
        ],
        "mcp_servers": [],
    }

    dry = sync_manifest(data)
    assert dry[0].action == "copy_slash_command_pack"
    assert dry[0].status == "would_create"
    assert not (root / "afs-status.md").exists()

    applied = sync_manifest(data, apply=True)
    assert applied[0].status == "synced"
    assert (root / "afs-status.md").read_text(encoding="utf-8").endswith("Use context.status.\n")

    current = sync_manifest(data)
    assert current[0].status == "up_to_date"


def test_agent_manifest_sync_preserves_custom_slash_commands(tmp_path: Path) -> None:
    canonical = tmp_path / "commands"
    canonical.mkdir()
    (canonical / "afs-status.md").write_text("canonical\n", encoding="utf-8")
    root = tmp_path / "opencode" / "command"
    root.mkdir(parents=True)
    existing = root / "afs-status.md"
    existing.write_text("custom\n", encoding="utf-8")
    data = {
        "version": 1,
        "paths": {},
        "harnesses": [{"name": "hcode", "command_roots": [str(root)], "mcp_servers": []}],
        "slash_command_packs": [
            {
                "name": "afs-opencode",
                "canonical_path": str(canonical),
                "targets": ["hcode"],
                "overwrite": False,
            }
        ],
        "mcp_servers": [],
    }

    dry = sync_manifest(data)
    assert dry[0].status == "customized"

    applied = sync_manifest(data, apply=True)
    assert applied[0].status == "customized"
    assert existing.read_text(encoding="utf-8") == "custom\n"
