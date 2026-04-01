"""Skill discovery and metadata parsing."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillMetadata:
    name: str
    path: Path
    triggers: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)


def _clean_token(value: str) -> str:
    return value.strip().strip('"').strip("'")


def merge_unique_paths(*groups: list[Path]) -> list[Path]:
    """Merge path groups while preserving order and uniqueness."""
    merged: list[Path] = []
    seen: set[str] = set()
    for group in groups:
        for path in group:
            resolved = path.expanduser().resolve()
            marker = str(resolved)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(resolved)
    return merged


def bundled_skill_roots(*, afs_root: str | Path | None = None) -> list[Path]:
    """Return bundled core skill roots available to the current runtime."""
    candidates: list[Path] = []
    root_value = afs_root or os.getenv("AFS_ROOT", "").strip()
    if root_value:
        candidates.append(Path(root_value).expanduser().resolve() / "skills")
    candidates.append(Path(__file__).resolve().parents[3] / "skills")
    return merge_unique_paths(
        [candidate for candidate in candidates if candidate.exists()]
    )


def resolve_skill_roots(
    profile_roots: list[Path],
    *,
    explicit_roots: list[Path] | None = None,
    afs_root: str | Path | None = None,
) -> list[Path]:
    """Resolve skill roots from explicit overrides, profile config, and bundled skills."""
    if explicit_roots:
        return merge_unique_paths(explicit_roots)
    return merge_unique_paths(profile_roots, bundled_skill_roots(afs_root=afs_root))


def _normalize_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean_token(item) for item in value if isinstance(item, str) and _clean_token(item)]
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [_clean_token(part) for part in text.split(",")]
    return [part for part in parts if part]

def _parse_frontmatter_block(lines: list[str]) -> dict[str, str | list[str]]:
    parsed: dict[str, str | list[str]] = {}
    current_list_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue

        if line.lstrip().startswith("- ") and current_list_key:
            parsed.setdefault(current_list_key, [])
            value = line.split("-", 1)[1].strip()
            if isinstance(parsed[current_list_key], list):
                parsed[current_list_key].append(value)
            continue

        current_list_key = None
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key:
            continue

        if not value:
            parsed[key] = []
            current_list_key = key
            continue

        parsed[key] = value.strip('"').strip("'")

    return parsed


def parse_skill_metadata(path: Path) -> SkillMetadata:
    """Parse SKILL.md metadata frontmatter."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    frontmatter: dict[str, str | list[str]] = {}
    if lines and lines[0].strip() == "---":
        closing = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                closing = idx
                break
        if closing is not None:
            frontmatter = _parse_frontmatter_block(lines[1:closing])

    name_value = frontmatter.get("name")
    if isinstance(name_value, str) and name_value.strip():
        name = name_value.strip()
    else:
        name = path.parent.name

    profile_value = frontmatter.get("profiles")
    if profile_value is None:
        profile_value = frontmatter.get("profile")

    return SkillMetadata(
        name=name,
        path=path.resolve(),
        triggers=_normalize_list(frontmatter.get("triggers")),
        requires=_normalize_list(frontmatter.get("requires")),
        profiles=_normalize_list(profile_value),
    )


def discover_skills(skill_roots: list[Path], profile: str | None = None) -> list[SkillMetadata]:
    """Discover SKILL.md files and filter by profile when requested."""
    discovered: list[SkillMetadata] = []
    for root in skill_roots:
        if not root.exists():
            continue
        try:
            candidates = list(root.rglob("SKILL.md"))
        except OSError:
            continue
        for candidate in candidates:
            try:
                metadata = parse_skill_metadata(candidate)
            except OSError:
                continue
            if profile and metadata.profiles:
                if profile not in metadata.profiles and "general" not in metadata.profiles:
                    continue
            discovered.append(metadata)

    discovered.sort(key=lambda item: item.name.lower())
    return discovered


def infer_trigger_matches(prompt: str, triggers: list[str]) -> bool:
    """Return true when any trigger token appears in prompt."""
    if not prompt or not triggers:
        return False
    lowered = prompt.lower()
    for trigger in triggers:
        if trigger.lower() in lowered:
            return True
    return False


def score_skill_relevance(prompt: str, skill: SkillMetadata) -> int:
    """Simple trigger-count relevance score for auto-loading skills."""
    if not prompt:
        return 0
    score = 0
    lowered = prompt.lower()
    for trigger in skill.triggers:
        token = trigger.lower().strip()
        if not token:
            continue
        if re.search(r"\b" + re.escape(token) + r"\b", lowered):
            score += 1
        elif token in lowered:
            score += 1
    return score
