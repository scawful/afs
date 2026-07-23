from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from afs.skills import (
    MAX_SKILL_BODIES_CHARS,
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_DIAGNOSTIC_CODE_CHARS,
    MAX_SKILL_DIAGNOSTIC_MESSAGE_CHARS,
    MAX_SKILL_DIAGNOSTICS,
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_FILE_CHARS,
    MAX_SKILL_METADATA_ITEM_CHARS,
    MAX_SKILL_PATH_CHARS,
    SkillDiscoveryDiagnostic,
    build_skill_matches,
    bundled_skill_roots,
    discover_skills,
    discover_skills_with_diagnostics,
    escape_skill_diagnostic_text,
    parse_skill_metadata,
    read_skill_body,
    resolve_skill_roots,
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


def test_discovery_reports_bad_entries_without_dropping_good_skills(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    good = root / "good" / "SKILL.md"
    invalid = root / "too-many-rules" / "SKILL.md"
    unreadable = root / "not-a-file" / "SKILL.md"
    good.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    unreadable.mkdir(parents=True)
    good.write_text(
        "---\nname: good\ntriggers: [focus]\n---\n\n# Good\n",
        encoding="utf-8",
    )
    invalid.write_text(
        "---\nname: too-many-rules\nenforcement:\n"
        + "".join(f"  - rule {index}\n" for index in range(17))
        + "---\n\n# Invalid\n",
        encoding="utf-8",
    )

    result = discover_skills_with_diagnostics([root])

    assert [skill.name for skill in result.skills] == ["good"]
    diagnostics = [diagnostic.to_dict() for diagnostic in result.diagnostics]
    assert {diagnostic["code"] for diagnostic in diagnostics} == {
        "skill_invalid",
        "skill_unreadable",
    }
    invalid_diagnostic = next(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic["path"] == str(invalid)
    )
    assert "enforcement exceeds 16 items" in invalid_diagnostic["message"]
    assert invalid_diagnostic["root"] == str(root.resolve())
    assert invalid_diagnostic["severity"] == "warning"


def test_discovery_bounds_diagnostics_and_reports_overflow(tmp_path: Path) -> None:
    missing_roots = [
        tmp_path / f"missing-{index}"
        for index in range(MAX_SKILL_DIAGNOSTICS + 7)
    ]

    result = discover_skills_with_diagnostics(missing_roots)

    assert result.diagnostic_count == MAX_SKILL_DIAGNOSTICS + 7
    assert len(result.diagnostics) == MAX_SKILL_DIAGNOSTICS
    assert result.diagnostics_omitted == 7


def test_root_resolution_failure_does_not_hide_healthy_siblings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    healthy_root = tmp_path / "healthy"
    healthy_skill = healthy_root / "healthy" / "SKILL.md"
    healthy_skill.parent.mkdir(parents=True)
    healthy_skill.write_text("---\nname: healthy\n---\n", encoding="utf-8")
    broken_root = tmp_path / "broken-loop"
    afs_root = tmp_path / "empty-afs"
    (afs_root / "skills").mkdir(parents=True)
    original_resolve = Path.resolve

    def simulated_resolve(path: Path, *args, **kwargs) -> Path:
        if path == broken_root:
            raise RuntimeError("simulated symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulated_resolve)

    roots = resolve_skill_roots(
        [healthy_root, broken_root],
        afs_root=afs_root,
    )
    result = discover_skills_with_diagnostics(roots)

    assert "healthy" in {skill.name for skill in result.skills}
    broken = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "root_unreadable"
    )
    assert broken.root == broken_root
    assert "simulated symlink loop" in broken.message


def test_root_stat_failure_does_not_hide_healthy_siblings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    healthy_root = tmp_path / "healthy"
    healthy_skill = healthy_root / "healthy" / "SKILL.md"
    healthy_skill.parent.mkdir(parents=True)
    healthy_skill.write_text("---\nname: healthy\n---\n", encoding="utf-8")
    broken_root = tmp_path / "too-long"
    original_exists = Path.exists

    def simulated_exists(path: Path) -> bool:
        if path == broken_root:
            raise OSError("simulated stat failure")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", simulated_exists)

    result = discover_skills_with_diagnostics([healthy_root, broken_root])

    assert "healthy" in {skill.name for skill in result.skills}
    broken = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.root == broken_root
    )
    assert broken.code == "root_unreadable"
    assert "simulated stat failure" in broken.message


def test_unknown_tilde_afs_root_reaches_discovery_as_a_diagnostic(
    tmp_path: Path,
) -> None:
    healthy_root = tmp_path / "healthy"
    healthy = healthy_root / "healthy" / "SKILL.md"
    healthy.parent.mkdir(parents=True)
    healthy.write_text("---\nname: healthy\n---\n", encoding="utf-8")
    unknown_home = Path("~afs-user-that-cannot-exist-48ba775")

    roots = resolve_skill_roots(
        [healthy_root],
        afs_root=unknown_home,
    )
    result = discover_skills_with_diagnostics(roots)

    assert "healthy" in {skill.name for skill in result.skills}
    unknown_diagnostic = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.root == unknown_home / "skills"
    )
    assert unknown_diagnostic.code == "root_unreadable"
    assert "expand or resolve" in unknown_diagnostic.message


def test_scan_errors_are_reported_per_root_and_subtree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    readable_root = tmp_path / "readable"
    good = readable_root / "good" / "SKILL.md"
    blocked_subtree = readable_root / "blocked"
    blocked_skill = blocked_subtree / "SKILL.md"
    failed_root = tmp_path / "failed-root"
    failed_skill = failed_root / "hidden" / "SKILL.md"
    good.parent.mkdir(parents=True)
    blocked_subtree.mkdir(parents=True)
    failed_skill.parent.mkdir(parents=True)
    good.write_text("---\nname: good\n---\n", encoding="utf-8")
    blocked_skill.write_text("---\nname: blocked\n---\n", encoding="utf-8")
    failed_skill.write_text("---\nname: hidden\n---\n", encoding="utf-8")
    blocked_resolved = blocked_subtree.resolve()
    failed_resolved = failed_root.resolve()
    original_scandir = os.scandir

    def simulated_scandir(path):
        candidate = Path(path)
        if candidate in {blocked_resolved, failed_resolved}:
            raise PermissionError(f"simulated scan denial: {candidate}")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", simulated_scandir)

    result = discover_skills_with_diagnostics([readable_root, failed_root])

    assert [skill.name for skill in result.skills] == ["good"]
    diagnostics = {diagnostic.code: diagnostic for diagnostic in result.diagnostics}
    assert diagnostics["root_scan_failed"].root == failed_resolved
    assert diagnostics["root_scan_failed"].path is None
    assert diagnostics["subtree_scan_failed"].path == blocked_resolved
    assert result.diagnostic_count == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits required")
def test_unreadable_subtree_is_not_silently_skipped(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    good = root / "good" / "SKILL.md"
    blocked = root / "blocked"
    blocked_skill = blocked / "SKILL.md"
    good.parent.mkdir(parents=True)
    blocked.mkdir(parents=True)
    good.write_text("---\nname: good\n---\n", encoding="utf-8")
    blocked_skill.write_text("---\nname: blocked\n---\n", encoding="utf-8")
    blocked.chmod(0)
    try:
        try:
            with os.scandir(blocked):
                pass
        except PermissionError:
            pass
        else:
            pytest.skip("current execution identity can still scan chmod-0 directories")

        result = discover_skills_with_diagnostics([root])
    finally:
        blocked.chmod(0o700)

    assert [skill.name for skill in result.skills] == ["good"]
    assert result.diagnostic_count == 1
    assert result.diagnostics[0].code == "subtree_scan_failed"
    assert result.diagnostics[0].path == blocked.resolve()


def test_unclosed_frontmatter_is_invalid_but_plain_body_remains_valid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    malformed = root / "malformed" / "SKILL.md"
    plain = root / "plain" / "SKILL.md"
    malformed.parent.mkdir(parents=True)
    plain.parent.mkdir(parents=True)
    malformed.write_text(
        "---\nname: claimed-name\ntriggers: [focus]\n",
        encoding="utf-8",
    )
    plain.write_text("# Plain body-only skill\n", encoding="utf-8")

    result = discover_skills_with_diagnostics([root])

    assert [skill.name for skill in result.skills] == ["plain"]
    assert result.diagnostic_count == 1
    assert result.diagnostics[0].code == "skill_invalid"
    assert result.diagnostics[0].path == malformed
    assert "missing a closing delimiter" in result.diagnostics[0].message


def test_structured_diagnostics_bound_every_field_and_report_truncation(
    tmp_path: Path,
) -> None:
    oversized_root = tmp_path.joinpath(
        *(f"segment-{index:02d}-" + ("x" * 140) for index in range(30))
    )

    discovery = discover_skills_with_diagnostics([oversized_root])
    root_payload = discovery.diagnostics[0].to_dict()

    assert len(root_payload["root"]) <= MAX_SKILL_PATH_CHARS
    assert "root" in root_payload["truncated_fields"]

    direct_payload = SkillDiscoveryDiagnostic(
        code="c" * (MAX_SKILL_DIAGNOSTIC_CODE_CHARS + 20),
        message="m" * (MAX_SKILL_DIAGNOSTIC_MESSAGE_CHARS + 20),
        root=tmp_path,
        path=Path("/tmp") / ("p" * (MAX_SKILL_PATH_CHARS + 20)),
    ).to_dict()
    assert len(direct_payload["code"]) <= MAX_SKILL_DIAGNOSTIC_CODE_CHARS
    assert len(direct_payload["message"]) <= MAX_SKILL_DIAGNOSTIC_MESSAGE_CHARS
    assert len(direct_payload["path"]) <= MAX_SKILL_PATH_CHARS
    assert direct_payload["truncated_fields"] == ["code", "message", "path"]


def test_skill_diagnostic_display_escapes_controls_and_markdown() -> None:
    rendered = escape_skill_diagnostic_text(
        "bad\n\x1b\x85\u202e\u2066\u2028\ud800`[path]*",
        max_chars=200,
        markdown=True,
    )

    assert rendered == (
        r"bad\u000a\u001b\u0085\u202e\u2066\u2028\ud800\`\[path\]\*"
    )
    assert not any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for char in rendered
    )


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


def test_skill_read_rejects_oversized_files_before_full_read(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("x" * (MAX_SKILL_FILE_CHARS + 1), encoding="utf-8")

    with pytest.raises(ValueError, match="Skill file exceeds"):
        read_skill_body(skill)
    assert discover_skills([tmp_path]) == []


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


@pytest.mark.timeout(2)
def test_discover_skills_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs unavailable")
    root = tmp_path / "skills"
    fifo = root / "blocked" / "SKILL.md"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)

    assert discover_skills([root]) == []


def test_skill_reads_reject_oversized_files(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill = root / "oversized" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"x" * (MAX_SKILL_FILE_BYTES + 1))

    assert discover_skills([root]) == []
    with pytest.raises(OSError, match="exceeds"):
        read_skill_body(skill)


def test_build_skill_matches_rejects_file_symlink_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    skill = root / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: demo\ntriggers: [demo]\n---\n\nSafe body.\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("SECRET_OUTSIDE_ROOT\n", encoding="utf-8")

    import afs.skills as skills_module

    def swap_before_body_read(prompt: str, metadata: object) -> int:
        del prompt, metadata
        skill.unlink()
        try:
            skill.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        return 1

    monkeypatch.setattr(skills_module, "score_skill_relevance", swap_before_body_read)
    matches = build_skill_matches("demo", [root])

    assert matches[0]["body"] == ""
    assert "SECRET_OUTSIDE_ROOT" not in str(matches)


def test_build_skill_matches_rejects_parent_symlink_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "demo"
    skill = skill_dir / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: demo\ntriggers: [demo]\n---\n\nSafe body.\n",
        encoding="utf-8",
    )
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "SKILL.md").write_text(
        "SECRET_OUTSIDE_PARENT\n",
        encoding="utf-8",
    )

    import afs.skills as skills_module

    def swap_before_body_read(prompt: str, metadata: object) -> int:
        del prompt, metadata
        skill.unlink()
        skill_dir.rmdir()
        try:
            skill_dir.symlink_to(outside_dir, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")
        return 1

    monkeypatch.setattr(skills_module, "score_skill_relevance", swap_before_body_read)
    matches = build_skill_matches("demo", [root])

    assert matches[0]["body"] == ""
    assert "SECRET_OUTSIDE_PARENT" not in str(matches)


def test_skill_reader_portable_fallback_preserves_regular_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    skill = root / "portable" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: portable\ntriggers: [portable]\n---\n\nPortable body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "supports_dir_fd", set())
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    matches = build_skill_matches("portable", [root])

    assert matches[0]["body"] == "Portable body."


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


def test_discovery_rejects_metadata_that_cannot_be_delivered(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill = root / "large-metadata" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    long_item = "x" * (MAX_SKILL_METADATA_ITEM_CHARS + 100)
    enforcement = "\n".join(f"  - {long_item}{index}" for index in range(12))
    skill.write_text(
        "---\nname: large-metadata\ntriggers: [focus]\nenforcement:\n"
        f"{enforcement}\n---\n\nBound the metadata.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enforcement item exceeds"):
        parse_skill_metadata(skill)
    assert discover_skills([root]) == []
    assert build_skill_matches("focus", [root], top_k=1) == []


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
