from __future__ import annotations

import os
from pathlib import Path

import pytest

from afs.skills import (
    MAX_SKILL_BODIES_CHARS,
    MAX_SKILL_BODY_CHARS,
    build_skill_matches,
    bundled_skill_roots,
    discover_skills,
    parse_skill_metadata,
    read_skill_body,
    score_skill_relevance,
    truncate_skill_body,
)


def test_parse_skill_frontmatter() -> None:
    path = Path(__file__).parent / "fixtures" / "skill_frontmatter" / "SKILL.md"
    metadata = parse_skill_metadata(path)

    assert metadata.name == "gemini-work"
    assert metadata.triggers == ["antigravity-cli", "agent studio"]
    assert metadata.requires == ["knowledge/work", "gemini mcp"]
    assert metadata.profiles == ["work", "general"]
    assert metadata.enforcement == [
        "Keep the Gemini workspace flow deterministic.",
        "Prefer MCP-backed context over ad hoc summaries.",
    ]
    assert metadata.verification == [
        "Run the workspace health checks before handoff.",
    ]
    assert read_skill_body(path) == ("# Skill Fixture", False)


def test_discover_skills_profile_filter(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    work_skill = root / "work" / "SKILL.md"
    domain_skill = root / "specialized" / "SKILL.md"
    work_skill.parent.mkdir(parents=True)
    domain_skill.parent.mkdir(parents=True)

    work_skill.write_text(
        "---\n"
        "name: work-skill\n"
        "triggers: [gemini]\n"
        "profiles: [work]\n"
        "---\n",
        encoding="utf-8",
    )
    domain_skill.write_text(
        "---\n"
        "name: specialized-skill\n"
        "triggers: [embedded]\n"
        "profiles: [specialized]\n"
        "---\n",
        encoding="utf-8",
    )

    work_only = discover_skills([root], profile="work")
    assert [skill.name for skill in work_only] == ["work-skill"]


def test_score_skill_relevance() -> None:
    path = Path(__file__).parent / "fixtures" / "skill_frontmatter" / "SKILL.md"
    metadata = parse_skill_metadata(path)
    score = score_skill_relevance("Need antigravity setup for agent studio", metadata)
    assert score >= 1


def test_read_skill_body_handles_plain_and_malformed_documents(tmp_path: Path) -> None:
    plain = tmp_path / "plain.md"
    plain.write_text("# Plain Skill\n\nDo the thing.\n", encoding="utf-8")
    assert read_skill_body(plain) == ("# Plain Skill\n\nDo the thing.", False)

    malformed = tmp_path / "malformed.md"
    malformed.write_text("---\nname: unsafe\n# Missing close\n", encoding="utf-8")
    assert read_skill_body(malformed) == ("", False)


def test_read_skill_body_enforces_exact_character_boundary(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("x" * MAX_SKILL_BODY_CHARS, encoding="utf-8")
    body, truncated = read_skill_body(skill)
    assert body == "x" * MAX_SKILL_BODY_CHARS
    assert truncated is False

    skill.write_text("x" * (MAX_SKILL_BODY_CHARS + 1), encoding="utf-8")
    first = read_skill_body(skill)
    second = read_skill_body(skill)
    assert first == second
    assert len(first[0]) == MAX_SKILL_BODY_CHARS
    assert first[0].endswith("...")
    assert first[1] is True
    assert read_skill_body(skill, max_chars=0) == ("", True)


def test_truncated_skill_body_closes_markdown_fence() -> None:
    body = "```python\n" + ("print('bounded')\n" * 200) + "```"
    excerpt, truncated = truncate_skill_body(body, max_chars=200)

    assert truncated is True
    assert len(excerpt) <= 200
    assert excerpt.endswith("\n```")
    assert excerpt.count("```") % 2 == 0


def test_truncated_skill_body_preserves_long_fence_length() -> None:
    body = "````python\nshort marker follows\n```\n" + ("x\n" * 200) + "````"
    excerpt, truncated = truncate_skill_body(body, max_chars=100)

    assert truncated is True
    assert len(excerpt) <= 100
    assert excerpt.endswith("\n````")


def test_discover_skills_rejects_symlinks_outside_trusted_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    linked_skill = root / "linked" / "SKILL.md"
    linked_skill.parent.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nname: outside\ntriggers: [outside]\n---\n\nDo not inject me.\n",
        encoding="utf-8",
    )
    try:
        os.symlink(outside, linked_skill)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert discover_skills([root]) == []


def test_build_skill_matches_is_stable_and_bounds_aggregate_bodies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    for name in ("zeta", "alpha", "delta", "beta"):
        skill = root / name / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            f"---\nname: {name}\ntriggers: [focus]\n---\n\n" + "x" * 2500,
            encoding="utf-8",
        )

    first = build_skill_matches("focus", [root], top_k=4)
    second = build_skill_matches("focus", [root], top_k=4)
    assert first == second
    assert [match["name"] for match in first] == ["alpha", "beta", "delta", "zeta"]
    assert sum(len(str(match["body"])) for match in first) == MAX_SKILL_BODIES_CHARS
    assert first[-1]["body"] == ""
    assert first[-1]["body_truncated"] is False
    assert first[-1]["body_omitted"] == "match_limit"
    assert all(match["body_chars"] == len(match["body"]) for match in first)


def test_build_skill_matches_prefers_first_root_for_duplicate_names(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "profile", tmp_path / "bundled"]
    for root, body in zip(
        roots,
        ("Profile instructions.", "Bundled instructions."),
        strict=True,
    ):
        skill = root / "shared" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\nname: shared\ntriggers: [focus]\n---\n\n" + body,
            encoding="utf-8",
        )

    matches = build_skill_matches("focus", roots)
    assert len(matches) == 1
    assert matches[0]["path"] == str((roots[0] / "shared" / "SKILL.md").resolve())
    assert matches[0]["body"] == "Profile instructions."


def test_build_skill_matches_uses_root_precedence_for_score_ties(tmp_path: Path) -> None:
    roots = [tmp_path / "profile", tmp_path / "bundled"]
    fixtures = [
        (roots[0], "z-profile", "Profile instructions."),
        (roots[1], "a-bundled", "Bundled instructions."),
    ]
    for root, name, body in fixtures:
        skill = root / name / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            f"---\nname: {name}\ntriggers: [focus]\n---\n\n{body}",
            encoding="utf-8",
        )

    matches = build_skill_matches("focus", roots)
    assert [match["name"] for match in matches] == ["z-profile", "a-bundled"]


def test_build_skill_matches_resolves_same_root_name_collisions_by_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    for directory, body in (("z-last", "Last body."), ("a-first", "First body.")):
        skill = root / directory / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            f"---\nname: shared\ntriggers: [focus]\n---\n\n{body}",
            encoding="utf-8",
        )

    matches = build_skill_matches("focus", [root])
    assert len(matches) == 1
    assert matches[0]["path"] == str((root / "a-first" / "SKILL.md").resolve())
    assert matches[0]["body"] == "First body."


def test_bundled_skill_root_resolves_from_source_checkout() -> None:
    expected = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    assert expected in bundled_skill_roots()
