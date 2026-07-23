"""Tests for skill discovery and matching with bundled skills."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from afs.cli.skills import skills_list_command, skills_match_command
from afs.skills import (
    discover_skills,
    escape_skill_diagnostic_text,
    parse_skill_metadata,
    score_skill_relevance,
)


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
    """Verify that packaged AFS skills are discoverable."""
    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    skills = discover_skills([skills_dir])
    names = {s.name for s in skills}
    assert names == {
        "adversarial-verification",
        "afs-cli-map",
        "agent-ops",
        "approvals-and-gates",
        "code-review",
        "context-search",
        "context-setup",
        "cpp-quality",
        "event-log",
        "extension-authoring",
        "fs-operations",
        "health-repair",
        "hivemind-comms",
        "implementation-planning",
        "issue-tracking",
        "mcp-server",
        "memory-maintenance",
        "mission-tracking",
        "profile-management",
        "python-quality",
        "research",
        "review-handoff",
        "session-workflows",
        "skill-authoring",
        "software-design",
        "structured-schemas",
        "task-queue",
        "triage",
        "typescript-quality",
        "verification-plans",
    }


def test_bundled_skills_are_matchable_and_have_bodies() -> None:
    """Every packaged skill has auto-match metadata and injectable instructions."""
    from afs.skills import read_skill_body

    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    for skill in discover_skills([skills_dir]):
        assert skill.triggers, f"bundled skill {skill.name!r} has no triggers"
        body, _truncated = read_skill_body(skill.path)
        assert body.strip(), f"bundled skill {skill.name!r} has an empty body"


def test_afs_feature_skills_rank_first_for_domain_prompts() -> None:
    """Distinct feature prompts rank their dedicated skill ahead of neighbors."""
    from afs.skills import build_skill_matches

    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    expectations = {
        "create a session handoff before ending": "session-workflows",
        "track this durable mission across sessions": "mission-tracking",
        "the AFS context index is stale; run index rebuild": "context-search",
        "afs doctor says there is a broken context": "health-repair",
        "show AFS event analytics for MCP tool errors": "event-log",
        "consolidate session history into memory": "memory-maintenance",
        "there is a pending approval request": "approvals-and-gates",
        "validate the structured response against the schema": "structured-schemas",
        "run the verification plan for changed files": "verification-plans",
        "write a new SKILL.md with distinctive triggers": "skill-authoring",
        "add an AFS extension manifest with a slim catalog": "extension-authoring",
        "which afs command should route this workflow": "afs-cli-map",
    }

    for prompt, expected in expectations.items():
        matches = build_skill_matches(prompt, [skills_dir], top_k=10)
        assert matches, f"no skill matched {prompt!r}"
        assert matches[0]["name"] == expected, (
            f"{prompt!r} ranked {matches[0]['name']!r} first, expected {expected!r}"
        )
        if expected != "afs-cli-map":
            assert "afs-cli-map" not in {match["name"] for match in matches}


def test_afs_cli_map_ignores_generic_afs_and_cli_mentions() -> None:
    """The catch-all map must not steal prompts from feature or code skills."""
    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    skills = {skill.name: skill for skill in discover_skills([skills_dir])}
    cli_map = skills["afs-cli-map"]

    for prompt in (
        "review the AFS Python CLI implementation",
        "run the afs context repair subcommand after showing its dry run",
        "prepare an afs session handoff",
    ):
        assert score_skill_relevance(prompt, cli_map) == 0


def test_feature_skills_ignore_unrelated_engineering_terms() -> None:
    """Generic substrings must not consume the three instruction-body slots."""
    from afs.skills import build_skill_matches

    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    for prompt, forbidden in (
        (
            "prevent a regression in the Python package builder",
            {"event-log", "session-workflows"},
        ),
        (
            "write tests for a durable C++ interface schema",
            {"mission-tracking", "structured-schemas"},
        ),
        (
            "search the browser extension manifest for a trigger",
            {"context-search", "extension-authoring", "skill-authoring"},
        ),
        (
            "review the C++ compiler optimization architecture and test plan",
            {"approvals-and-gates"},
        ),
        ("debug the service health check endpoint", {"health-repair"}),
        ("open the browser session timeline", {"event-log"}),
    ):
        names = {
            match["name"]
            for match in build_skill_matches(prompt, [skills_dir], top_k=10)
        }
        assert names.isdisjoint(forbidden), (prompt, names & forbidden)


def test_bundled_skill_budgets_and_human_gates_are_explicit() -> None:
    """Operational skill text must preserve current limits and user gates."""
    from afs.skills import read_skill_body

    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"
    skills = {skill.name: skill for skill in discover_skills([skills_dir])}

    authoring_body, _truncated = read_skill_body(skills["skill-authoring"].path)
    assert "at most 3" in authoring_body.lower()
    assert "2,000 characters" in authoring_body
    assert "6,000-character" in authoring_body
    assert "src/afs/bundled_skills/<name>/SKILL.md" in authoring_body

    for name in ("agent-ops", "health-repair"):
        enforcement = " ".join(skills[name].enforcement).lower()
        assert "explicit user direction" in enforcement

    agent_body, agent_truncated = read_skill_body(skills["agent-ops"].path)
    assert agent_truncated is False
    assert "--allow-destructive" in agent_body

    approval_rules = " ".join(skills["approvals-and-gates"].enforcement).lower()
    assert "never approve" in approval_rules
    assert "never promote" in approval_rules

    approvals_body, _ = read_skill_body(skills["approvals-and-gates"].path)
    assert "approvals approve <agent> <action>" in approvals_body
    assert "work approvals approve <approval_id>" in approvals_body
    assert "not the global supervisor store" in approvals_body

    cli_body, _ = read_skill_body(skills["afs-cli-map"].path)
    for surface in ("personal", "briefing", "execution inspect"):
        assert surface in cli_body


def test_quality_skills_match_engineering_prompts() -> None:
    skills_dir = Path(__file__).parent.parent / "src" / "afs" / "bundled_skills"

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


def test_skills_list_surfaces_partial_discovery_warnings(
    tmp_path: Path,
    capsys,
) -> None:
    skill_root = tmp_path / "skills"
    good = skill_root / "good" / "SKILL.md"
    invalid = skill_root / "invalid\nforged\x1b\x85`[entry]" / "SKILL.md"
    good.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    good.write_text(
        "---\nname: good\ntriggers: [focus]\n---\n\n# Good\n",
        encoding="utf-8",
    )
    invalid.write_text(
        "---\nname: invalid\nenforcement:\n"
        + "".join(f"  - rule {index}\n" for index in range(17))
        + "---\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    args = Namespace(
        config=str(config_path),
        profile=None,
        root=[str(skill_root)],
        json=True,
    )

    assert skills_list_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [skill["name"] for skill in payload["skills"]] == ["good"]
    assert payload["diagnostic_count"] == 1
    assert payload["diagnostics"][0]["code"] == "skill_invalid"
    assert payload["diagnostics"][0]["path"] == str(invalid)

    args.json = False
    assert skills_list_command(args) == 0
    human_output = capsys.readouterr().out
    assert "warnings: 1" in human_output
    sanitized_invalid = escape_skill_diagnostic_text(
        invalid,
        max_chars=512,
    )
    assert f"[skill_invalid] {sanitized_invalid}" in human_output
    assert "\nforged" not in human_output
    assert "\x1b" not in human_output
    assert "\x85" not in human_output
    assert "\\u001b" in human_output
    assert "\\u0085" in human_output
    assert "good" in human_output

    match_args = Namespace(
        config=str(config_path),
        profile=None,
        root=[str(skill_root)],
        prompt="focus",
        top_k=10,
        json=True,
    )
    assert skills_match_command(match_args) == 0
    match_payload = json.loads(capsys.readouterr().out)
    assert match_payload["matches"][0]["name"] == "good"
    assert match_payload["diagnostic_count"] == 1
    assert match_payload["diagnostics"][0]["path"] == str(invalid)


def test_skills_list_keeps_good_explicit_root_when_another_cannot_resolve(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    healthy_root = tmp_path / "healthy"
    healthy = healthy_root / "healthy" / "SKILL.md"
    healthy.parent.mkdir(parents=True)
    healthy.write_text("---\nname: healthy\n---\n", encoding="utf-8")
    broken_root = tmp_path / "broken-loop"
    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    original_resolve = Path.resolve

    def simulated_resolve(path: Path, *args, **kwargs) -> Path:
        if path == broken_root:
            raise RuntimeError("simulated symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulated_resolve)
    args = Namespace(
        config=str(config_path),
        profile=None,
        root=[str(healthy_root), str(broken_root)],
        json=True,
    )

    assert skills_list_command(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert [skill["name"] for skill in payload["skills"]] == ["healthy"]
    assert payload["diagnostic_count"] == 1
    assert payload["diagnostics"][0]["code"] == "root_unreadable"
    assert payload["diagnostics"][0]["root"] == str(broken_root)


def test_skills_list_and_match_use_repo_runtime_config_precedence(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    user_config = home / ".config" / "afs" / "config.toml"
    user_skill = tmp_path / "user-skills" / "user" / "SKILL.md"
    repo_skill = repo / "repo-skills" / "repo" / "SKILL.md"
    user_config.parent.mkdir(parents=True)
    repo_skill.parent.mkdir(parents=True)
    user_skill.parent.mkdir(parents=True)
    repo_skill.write_text(
        "---\nname: repo-skill\ntriggers: [repo-focus]\n---\n",
        encoding="utf-8",
    )
    user_skill.write_text(
        "---\nname: user-skill\ntriggers: [user-focus]\n---\n",
        encoding="utf-8",
    )
    user_config.write_text(
        "[profiles]\n"
        'active_profile = "user"\n'
        "[profiles.user]\n"
        f'skill_roots = ["{user_skill.parent.parent}"]\n'
        "[extensions]\n"
        "auto_discover = false\n",
        encoding="utf-8",
    )
    (repo / "afs.toml").write_text(
        "[profiles]\n"
        'active_profile = "repo"\n'
        "[profiles.repo]\n"
        f'skill_roots = ["{repo_skill.parent.parent}"]\n'
        "[extensions]\n"
        "auto_discover = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    for name in (
        "AFS_CONFIG_PATH",
        "AFS_PREFER_REPO_CONFIG",
        "AFS_PREFER_USER_CONFIG",
        "AFS_ROOT",
        "AFS_SKILL_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(repo)

    list_args = Namespace(config=None, profile=None, root=None, json=True)
    assert skills_list_command(list_args) == 0
    list_payload = json.loads(capsys.readouterr().out)

    assert list_payload["profile"] == "repo"
    list_names = {entry["name"] for entry in list_payload["skills"]}
    assert "repo-skill" in list_names
    assert "user-skill" not in list_names

    match_args = Namespace(
        config=None,
        profile=None,
        root=None,
        prompt="repo-focus",
        top_k=10,
        json=True,
    )
    assert skills_match_command(match_args) == 0
    match_payload = json.loads(capsys.readouterr().out)

    assert match_payload["profile"] == "repo"
    assert [entry["name"] for entry in match_payload["matches"]] == ["repo-skill"]
