from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_setup_scripts_do_not_assume_a_home_workspace_layout() -> None:
    upgrade = (REPO_ROOT / "scripts" / "afs-upgrade-agent-setup").read_text(encoding="utf-8")
    hcode = (REPO_ROOT / "scripts" / "afs-hcode").read_text(encoding="utf-8")

    assert "$HOME/src" not in upgrade
    assert "${HOME}/src" not in upgrade
    assert "--setup-hcode requires --halext-code PATH" in upgrade
    assert "${WORKSPACE}/.agents/skills" in upgrade
    assert "--skill-root NAME=PATH" in upgrade
    assert "--command-root NAME=PATH" in upgrade
    assert "--export-path NAME=PATH" in upgrade
    assert 'HCODE_CMD="${HCODE_CMD:-hcode}"' in hcode


def test_opencode_command_pack_is_small_namespaced_and_path_neutral() -> None:
    root = REPO_ROOT / "slash-commands" / "opencode"
    commands = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.md"))

    assert commands == [
        "afs.md",
        "afs/handoff.md",
        "afs/repair.md",
        "afs/status.md",
        "afs/verify.md",
    ]
    for relative in commands:
        content = (root / relative).read_text(encoding="utf-8")
        assert "/Users/" not in content
        assert "~/src" not in content
        assert "scripts/afs" not in content


def test_bundled_gemini_antigravity_skill_is_provider_and_path_neutral() -> None:
    content = (REPO_ROOT / "src" / "afs" / "bundled_skills" / "afs" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "name: afs" in content
    assert "description:" in content
    assert "/Users/" not in content
    assert "~/src" not in content
    assert "gemini-" not in content.lower()
