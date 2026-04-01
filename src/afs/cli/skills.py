"""Skill metadata CLI commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..config import load_config_model
from ..profiles import resolve_active_profile
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
