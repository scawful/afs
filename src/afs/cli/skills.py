"""Skill metadata CLI commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..cli._utils import load_runtime_config_from_args, resolve_context_paths
from ..config import load_config_model
from ..manager import AFSManager
from ..profiles import resolve_active_profile
from ..skill_mining import (
    mine_skill_candidates,
    normalize_promoted_skill_name,
    record_skill_candidate_review_state,
    render_promoted_skill_markdown,
    render_skill_candidate_review,
    review_skill_candidates,
    write_promoted_skill,
    write_skill_candidate_artifacts,
)
from ..skills import (
    discover_skills,
    resolve_skill_roots,
    score_skill_relevance,
)


def _print_hint(text: str) -> None:
    """Print a dimmed hint line."""
    if sys.stdout.isatty() and os.getenv("NO_COLOR") is None:
        print(f"  \033[2m{text}\033[0m")
    else:
        print(f"  {text}")


def _resolve_skill_roots(args: argparse.Namespace, profile_roots: list[Path]) -> list[Path]:
    explicit_roots = (
        [Path(path).expanduser().resolve() for path in args.root]
        if args.root
        else None
    )
    return resolve_skill_roots(
        profile_roots,
        explicit_roots=explicit_roots,
        afs_root=os.getenv("AFS_ROOT", "").strip() or None,
    )


def skills_list_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    profile = resolve_active_profile(config, profile_name=args.profile)

    roots = _resolve_skill_roots(args, list(profile.skill_roots))

    skills = discover_skills(roots, profile=profile.name)
    if args.json:
        payload = {
            "profile": profile.name,
            "roots": [str(path) for path in roots],
            "skills": [
                {
                    "name": skill.name,
                    "path": str(skill.path),
                    "triggers": skill.triggers,
                    "requires": skill.requires,
                    "profiles": skill.profiles,
                    "enforcement": skill.enforcement,
                    "verification": skill.verification,
                }
                for skill in skills
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"profile: {profile.name}")
    if not skills:
        print("(no skills)")
        _print_hint("add SKILL.md files to profile skill_roots, extensions, or AFS_ROOT/skills")
        _print_hint("example: skills/<name>/SKILL.md with frontmatter (name, triggers, profiles)")
        return 0
    for skill in skills:
        triggers = ",".join(skill.triggers) if skill.triggers else "-"
        requires = ",".join(skill.requires) if skill.requires else "-"
        print(f"{skill.name}\t{skill.path}\ttriggers={triggers}\trequires={requires}")
    _print_hint(f"{len(skills)} skills  |  afs skills match '<prompt>'  |  afs skills list --json")
    return 0


def skills_match_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    profile = resolve_active_profile(config, profile_name=args.profile)

    roots = _resolve_skill_roots(args, list(profile.skill_roots))

    skills = discover_skills(roots, profile=profile.name)
    ranked = []
    for skill in skills:
        score = score_skill_relevance(args.prompt, skill)
        if score > 0:
            ranked.append((score, skill))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if args.json:
        payload = {
            "profile": profile.name,
            "prompt": args.prompt,
            "matches": [
                {
                    "score": score,
                    "name": skill.name,
                    "path": str(skill.path),
                    "triggers": skill.triggers,
                    "requires": skill.requires,
                    "enforcement": skill.enforcement,
                    "verification": skill.verification,
                }
                for score, skill in ranked[: args.top_k]
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    for score, skill in ranked[: args.top_k]:
        print(f"{score}\t{skill.name}\t{skill.path}")
    if not ranked:
        print("(no matches)")
    return 0


def skills_mine_command(args: argparse.Namespace) -> int:
    start_dir = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    config, _config_path = load_runtime_config_from_args(args, start_dir=start_dir)
    manager = AFSManager(config=config)

    try:
        _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args,
            manager,
            prefer_existing=True,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = mine_skill_candidates(
        context_path,
        lookback_hours=args.lookback_hours,
        max_sessions=args.max_sessions,
        min_occurrences=args.min_occurrences,
        max_candidates=args.max_candidates,
        replay_limit=args.replay_limit,
        config=manager.config,
    )

    if not args.no_write_artifacts:
        payload["artifact_paths"] = write_skill_candidate_artifacts(
            manager,
            context_path,
            payload,
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{payload['candidate_count']} candidates from "
        f"{payload['successful_sessions']} successful sessions"
    )
    for candidate in payload["candidates"]:
        tool_sequence = " -> ".join(candidate.get("tool_sequence", []))
        print(
            f"{candidate['confidence']}\t{candidate['occurrences']}x\t"
            f"{candidate['name']}\t{tool_sequence}"
        )
    if not payload["candidates"]:
        print("(no candidates)")

    artifact_paths = payload.get("artifact_paths")
    if isinstance(artifact_paths, dict) and artifact_paths:
        _print_hint(f"json: {artifact_paths.get('json', '')}")
        _print_hint(f"markdown: {artifact_paths.get('markdown', '')}")
    return 0


def skills_review_command(args: argparse.Namespace) -> int:
    start_dir = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    config, _config_path = load_runtime_config_from_args(args, start_dir=start_dir)
    manager = AFSManager(config=config)

    try:
        _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args,
            manager,
            prefer_existing=True,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = review_skill_candidates(
        manager,
        context_path,
        artifact_path=args.artifact,
        candidate_id=args.candidate,
        status_filter=getattr(args, "status", ""),
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(render_skill_candidate_review(payload))
    artifact_count = int(payload.get("artifact_count", 0) or 0)
    available = payload.get("available_artifacts")
    if artifact_count > 1 and isinstance(available, list) and available:
        _print_hint(f"latest artifact selected from {artifact_count} available files")
    if not payload.get("artifact_path"):
        _print_hint("run `afs skills mine --path <project>` to generate candidate artifacts")
    return 0


def _record_candidate_decision_command(
    args: argparse.Namespace,
    *,
    status: str,
    missing_message: str,
) -> int:
    start_dir = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    config, _config_path = load_runtime_config_from_args(args, start_dir=start_dir)
    manager = AFSManager(config=config)

    try:
        _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args,
            manager,
            prefer_existing=True,
        )
        review = review_skill_candidates(
            manager,
            context_path,
            artifact_path=args.artifact,
            candidate_id=args.candidate,
            status_filter="",
            limit=max(args.limit, 1),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_candidate_count = int(review.get("total_candidate_count", 0) or 0)
    if total_candidate_count == 0:
        print(missing_message, file=sys.stderr)
        return 1
    if total_candidate_count > 1:
        print(
            "Multiple candidates matched. Use `afs skills review` or pass `--candidate <id>`.",
            file=sys.stderr,
        )
        return 1

    candidate = review["candidates"][0]
    candidate_id = str(candidate.get("id", "")).strip() or str(candidate.get("name", "")).strip()

    try:
        payload = {
            "artifact_path": str(review.get("artifact_path", "")).strip(),
            "candidate_id": candidate_id,
            "candidate_name": str(candidate.get("name", "")).strip(),
            "status": status,
        }
        payload["review_state"] = record_skill_candidate_review_state(
            manager,
            context_path,
            candidate_id=candidate_id,
            status=status,
            artifact_path=payload["artifact_path"],
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{status}: {candidate_id}")
    print(f"artifact: {payload['artifact_path']}")
    return 0


def _resolve_promotion_root(
    args: argparse.Namespace,
    *,
    profile_name: str,
    config,
) -> Path:
    explicit_root = str(getattr(args, "root", "") or "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    profile = resolve_active_profile(config, profile_name=profile_name)
    if profile.skill_roots:
        return profile.skill_roots[0].expanduser().resolve()

    raise ValueError(
        "No writable skill root configured for the active profile. "
        "Pass `--root <path>` to choose a promotion target."
    )


def skills_promote_command(args: argparse.Namespace) -> int:
    start_dir = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    config, _config_path = load_runtime_config_from_args(args, start_dir=start_dir)
    manager = AFSManager(config=config)

    try:
        _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args,
            manager,
            prefer_existing=True,
        )
        review = review_skill_candidates(
            manager,
            context_path,
            artifact_path=args.artifact,
            candidate_id=args.candidate,
            status_filter="",
            limit=max(args.limit, 1),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_candidate_count = int(review.get("total_candidate_count", 0) or 0)
    if total_candidate_count == 0:
        print("No matching skill candidate found to promote.", file=sys.stderr)
        return 1
    if total_candidate_count > 1:
        print(
            "Multiple candidates matched. Use `afs skills review` or pass `--candidate <id>`.",
            file=sys.stderr,
        )
        return 1

    candidate = review["candidates"][0]
    profile_name = str(args.profile or resolve_active_profile(config).name).strip()
    skill_name = normalize_promoted_skill_name(
        str(args.name or candidate.get("name") or candidate.get("id") or "").strip()
    )

    try:
        root_path = _resolve_promotion_root(args, profile_name=profile_name, config=config)
        content = render_promoted_skill_markdown(
            candidate,
            skill_name=skill_name,
            profile_name=profile_name,
            artifact_path=str(review.get("artifact_path", "")).strip(),
        )
        payload = {
            "artifact_path": str(review.get("artifact_path", "")).strip(),
            "candidate_id": str(candidate.get("id", "")).strip(),
            "skill_name": skill_name,
            "root": str(root_path),
            "dry_run": bool(args.dry_run),
            "content": content,
        }
        if not args.dry_run:
            payload.update(
                write_promoted_skill(
                    root_path,
                    skill_name=skill_name,
                    content=content,
                    force=bool(args.force),
                )
            )
            payload["review_state"] = record_skill_candidate_review_state(
                manager,
                context_path,
                candidate_id=payload["candidate_id"],
                status="promoted",
                artifact_path=payload["artifact_path"],
                skill_name=payload["skill_name"],
                skill_path=payload["skill_path"],
                root=payload["root"],
            )
        else:
            payload.update(
                {
                    "skill_dir": str(root_path / skill_name),
                    "skill_path": str(root_path / skill_name / "SKILL.md"),
                    "existed": (root_path / skill_name / "SKILL.md").exists(),
                    "overwritten": False,
                }
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    action = "preview" if args.dry_run else "wrote"
    print(f"{action}: {payload['skill_path']}")
    print(f"candidate: {payload['candidate_id']}")
    print(f"artifact: {payload['artifact_path']}")
    if args.dry_run:
        print("")
        print(content)
    return 0


def skills_reject_command(args: argparse.Namespace) -> int:
    return _record_candidate_decision_command(
        args,
        status="rejected",
        missing_message="No matching skill candidate found to reject.",
    )


def skills_archive_command(args: argparse.Namespace) -> int:
    return _record_candidate_decision_command(
        args,
        status="archived",
        missing_message="No matching skill candidate found to archive.",
    )


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", help="Project path used to resolve the active context.")
    parser.add_argument("--context-root", help="Explicit AFS context root path.")
    parser.add_argument("--context-dir", help="Context directory name override.")


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("skills", help="Skill metadata and trigger utilities.")
    sub = parser.add_subparsers(dest="skills_command")

    list_parser = sub.add_parser("list", help="List discovered skills.")
    list_parser.add_argument("--config", help="Config path.")
    list_parser.add_argument("--profile", help="Profile name override.")
    list_parser.add_argument("--root", action="append", help="Skill root path override.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=skills_list_command)

    match_parser = sub.add_parser("match", help="Rank skill matches for a prompt.")
    match_parser.add_argument("prompt", help="Prompt to score against skill triggers.")
    match_parser.add_argument("--config", help="Config path.")
    match_parser.add_argument("--profile", help="Profile name override.")
    match_parser.add_argument("--root", action="append", help="Skill root path override.")
    match_parser.add_argument("--top-k", type=int, default=10, help="Maximum matches.")
    match_parser.add_argument("--json", action="store_true", help="Output JSON.")
    match_parser.set_defaults(func=skills_match_command)

    mine_parser = sub.add_parser(
        "mine",
        help="Mine repeated successful session traces into reviewable skill candidates.",
    )
    _add_context_args(mine_parser)
    mine_parser.add_argument("--config", help="Config path.")
    mine_parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24 * 7,
        help="Only analyze sessions ending within this many hours.",
    )
    mine_parser.add_argument(
        "--max-sessions",
        type=int,
        default=50,
        help="Maximum recorded sessions to analyze.",
    )
    mine_parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum repeated successful traces required to emit a candidate.",
    )
    mine_parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum candidates to return.",
    )
    mine_parser.add_argument(
        "--replay-limit",
        type=int,
        default=200,
        help="Maximum events to replay per session.",
    )
    mine_parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Do not write scratchpad skill-candidate artifacts.",
    )
    mine_parser.add_argument("--json", action="store_true", help="Output JSON.")
    mine_parser.set_defaults(func=skills_mine_command)

    review_parser = sub.add_parser(
        "review",
        help="Review mined skill-candidate artifacts from scratchpad.",
    )
    _add_context_args(review_parser)
    review_parser.add_argument("--config", help="Config path.")
    review_parser.add_argument(
        "--artifact",
        help="Explicit skill-candidate JSON artifact path. Defaults to the latest scratchpad artifact.",
    )
    review_parser.add_argument(
        "--candidate",
        help="Filter review output to a specific candidate id/name.",
    )
    review_parser.add_argument(
        "--status",
        help="Optional candidate status filter, e.g. pending or promoted.",
    )
    review_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum candidates to show from the selected artifact.",
    )
    review_parser.add_argument("--json", action="store_true", help="Output JSON.")
    review_parser.set_defaults(func=skills_review_command)

    reject_parser = sub.add_parser(
        "reject",
        help="Mark one reviewed skill candidate as rejected.",
    )
    _add_context_args(reject_parser)
    reject_parser.add_argument("--config", help="Config path.")
    reject_parser.add_argument(
        "--artifact",
        help="Explicit skill-candidate JSON artifact path. Defaults to the latest scratchpad artifact.",
    )
    reject_parser.add_argument(
        "--candidate",
        help="Candidate id/name to reject. Required when the artifact contains multiple candidates.",
    )
    reject_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum candidates to inspect when resolving the requested candidate.",
    )
    reject_parser.add_argument("--json", action="store_true", help="Output JSON.")
    reject_parser.set_defaults(func=skills_reject_command)

    archive_parser = sub.add_parser(
        "archive",
        help="Mark one reviewed skill candidate as archived.",
    )
    _add_context_args(archive_parser)
    archive_parser.add_argument("--config", help="Config path.")
    archive_parser.add_argument(
        "--artifact",
        help="Explicit skill-candidate JSON artifact path. Defaults to the latest scratchpad artifact.",
    )
    archive_parser.add_argument(
        "--candidate",
        help="Candidate id/name to archive. Required when the artifact contains multiple candidates.",
    )
    archive_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum candidates to inspect when resolving the requested candidate.",
    )
    archive_parser.add_argument("--json", action="store_true", help="Output JSON.")
    archive_parser.set_defaults(func=skills_archive_command)

    promote_parser = sub.add_parser(
        "promote",
        help="Promote one reviewed skill candidate into a starter SKILL.md.",
    )
    _add_context_args(promote_parser)
    promote_parser.add_argument("--config", help="Config path.")
    promote_parser.add_argument(
        "--artifact",
        help="Explicit skill-candidate JSON artifact path. Defaults to the latest scratchpad artifact.",
    )
    promote_parser.add_argument(
        "--candidate",
        help="Candidate id/name to promote. Required when the artifact contains multiple candidates.",
    )
    promote_parser.add_argument(
        "--profile",
        help="Profile name override used for default root resolution and frontmatter profiles.",
    )
    promote_parser.add_argument(
        "--root",
        help="Explicit writable skill root target. Defaults to the first configured skill root for the active profile.",
    )
    promote_parser.add_argument(
        "--name",
        help="Override the promoted skill name/directory slug.",
    )
    promote_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum candidates to inspect when resolving the requested candidate.",
    )
    promote_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the promoted skill without writing it.",
    )
    promote_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing target SKILL.md.",
    )
    promote_parser.add_argument("--json", action="store_true", help="Output JSON.")
    promote_parser.set_defaults(func=skills_promote_command)
