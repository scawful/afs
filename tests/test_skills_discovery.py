"""Tests for skill discovery and matching with bundled skills."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from afs.cli.skills import skills_list_command
from afs.skills import discover_skills, parse_skill_metadata, score_skill_relevance


def test_parse_skill_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: my-skill\n"
        "triggers:\n"
        "  - test\n"
        "  - debug\n"
        "profiles:\n"
        "  - general\n"
        "requires:\n"
        "  - afs\n"
        "---\n"
        "\n# My Skill\n\nDoes things.\n",
        encoding="utf-8",
    )
    meta = parse_skill_metadata(skill_md)
    assert meta.name == "my-skill"
    assert meta.triggers == ["test", "debug"]
    assert meta.profiles == ["general"]
    assert meta.requires == ["afs"]
    assert meta.enforcement == []
    assert meta.verification == []


def test_discover_skills_from_roots(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    for name in ("alpha", "beta"):
        d = root / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
    skills = discover_skills([root])
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"alpha", "beta"}


def test_discover_skills_profile_filter(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    for name, prof in [("general-skill", "general"), ("dev-skill", "dev")]:
        d = root / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\nprofiles:\n  - {prof}\n---\n",
            encoding="utf-8",
        )
    dev_skills = discover_skills([root], profile="dev")
    names = {s.name for s in dev_skills}
    # dev profile sees dev-skill + general-skill (general is always included)
    assert "dev-skill" in names
    assert "general-skill" in names


def test_score_skill_relevance_trigger_match() -> None:
    from afs.skills import SkillMetadata

    skill = SkillMetadata(
        name="agent-ops",
        path=Path("/fake"),
        triggers=["agent", "spawn", "supervisor"],
    )
    assert score_skill_relevance("how do I spawn an agent", skill) == 2
    assert score_skill_relevance("unrelated query", skill) == 0


def test_discover_bundled_skills() -> None:
    """Verify that the skills/ directory in the AFS repo is discoverable."""
    skills_dir = Path(__file__).parent.parent / "skills"
    if not skills_dir.exists():
        return  # Skip if running from installed package
    skills = discover_skills([skills_dir])
    names = {s.name for s in skills}
    assert "context-setup" in names
    assert "agent-ops" in names
    assert "task-queue" in names
    assert "code-review" in names
    assert "cpp-quality" in names
    assert "python-quality" in names
    assert "typescript-quality" in names
    assert "software-design" in names
    assert "implementation-planning" in names
    assert len(skills) >= 12


def test_quality_skills_match_engineering_prompts() -> None:
    skills_dir = Path(__file__).parent.parent / "skills"
    if not skills_dir.exists():
        return

    skills = {skill.name: skill for skill in discover_skills([skills_dir])}
    prompt = (
        "Review a python refactor for anti-patterns, plan a TypeScript migration, "
        "check the C++ ownership design, and call out architecture risks."
    )

    assert score_skill_relevance(prompt, skills["code-review"]) >= 2
    assert score_skill_relevance(prompt, skills["python-quality"]) >= 1
    assert score_skill_relevance(prompt, skills["typescript-quality"]) >= 1
    assert score_skill_relevance(prompt, skills["cpp-quality"]) >= 1
    assert score_skill_relevance(prompt, skills["software-design"]) >= 1
    assert score_skill_relevance(prompt, skills["implementation-planning"]) >= 1


def test_skills_list_includes_bundled_afs_root_skills(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    repo_root = tmp_path / "afs-root"
    bundled_skill = repo_root / "skills" / "bundled-demo"
    bundled_skill.mkdir(parents=True)
    (bundled_skill / "SKILL.md").write_text(
        "---\nname: bundled-demo\nprofiles:\n  - general\n---\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")

    monkeypatch.setenv("AFS_ROOT", str(repo_root))
    args = Namespace(config=str(config_path), profile=None, root=None, json=True)

    exit_code = skills_list_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in payload["skills"]}
    assert "bundled-demo" in names
    assert str((repo_root / "skills").resolve()) in payload["roots"]
