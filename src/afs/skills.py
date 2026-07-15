"""Skill discovery and metadata parsing."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_SKILL_FILE_BYTES = 256 * 1024
MAX_SKILL_BODY_CHARS = 2_000
MAX_SKILL_BODIES_CHARS = 6_000
MAX_SKILL_BODY_MATCHES = 3
MAX_SKILL_MATCHES = 10
MAX_SKILL_FILE_CHARS = 64_000
MAX_SKILL_NAME_CHARS = 256
MAX_SKILL_PATH_CHARS = 4_096
MAX_SKILL_METADATA_ITEMS = 16
MAX_SKILL_METADATA_ITEM_CHARS = 256


@dataclass
class SkillMetadata:
    name: str
    path: Path
    triggers: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)
    enforcement: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)


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
    candidates.append(Path(__file__).resolve().parent / "bundled_skills")
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


def _skill_open_flags(*, directory: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    return flags


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_open_skill_file(descriptor: int, *, path: Path, max_bytes: int) -> str:
    """Read one already-opened regular skill file with a hard byte bound."""
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"skill path is not a regular file: {path}")
        if metadata.st_size > max_bytes:
            raise OSError(f"skill file exceeds the {max_bytes}-byte limit: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise OSError(f"skill file exceeds the {max_bytes}-byte limit: {path}")
    content = payload.decode("utf-8", errors="replace")
    if len(content) > MAX_SKILL_FILE_CHARS:
        raise ValueError(
            f"Skill file exceeds {MAX_SKILL_FILE_CHARS} characters: {path}"
        )
    return content


def _open_skill_beneath(root: Path, relative_path: Path) -> int:
    """Open a file beneath *root* without following path-component symlinks."""
    root_descriptor = os.open(root, _skill_open_flags(directory=True))
    descriptor = root_descriptor
    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise OSError(f"skill root is not a directory: {root}")
        expected_root = os.stat(root, follow_symlinks=False)
        if not _same_file_identity(root_metadata, expected_root):
            raise OSError(f"skill root changed while opening: {root}")

        parts = relative_path.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise OSError(f"invalid skill path beneath {root}: {relative_path}")
        for index, part in enumerate(parts):
            next_descriptor = os.open(
                part,
                _skill_open_flags(directory=index < len(parts) - 1),
                dir_fd=descriptor,
            )
            if descriptor != root_descriptor:
                os.close(descriptor)
            descriptor = next_descriptor
            if index < len(parts) - 1:
                metadata = os.fstat(descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise OSError(
                        f"skill path parent is not a directory: {root / Path(*parts[: index + 1])}"
                    )
        return descriptor
    except BaseException:
        if descriptor != root_descriptor:
            os.close(descriptor)
        raise
    finally:
        os.close(root_descriptor)


def _path_identity_chain(root: Path, path: Path) -> list[tuple[Path, os.stat_result]]:
    """Snapshot a no-symlink path chain for platforms without descriptor traversal."""
    relative = path.relative_to(root)
    chain: list[tuple[Path, os.stat_result]] = []
    current = root
    for part in ("", *relative.parts):
        if part:
            current = current / part
        metadata = os.stat(current, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError(f"skill path must not contain symlinks: {current}")
        chain.append((current, metadata))
    return chain


def _read_skill_text(
    path: Path,
    *,
    trusted_root: Path | None = None,
    max_bytes: int = MAX_SKILL_FILE_BYTES,
) -> str:
    """Read one bounded skill file while preserving a configured-root boundary."""
    limit = max(0, max_bytes)
    resolved_path = path.expanduser().resolve(strict=True)
    if trusted_root is None:
        descriptor = os.open(resolved_path, _skill_open_flags())
        return _read_open_skill_file(descriptor, path=resolved_path, max_bytes=limit)

    resolved_root = trusted_root.expanduser().resolve(strict=True)
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise OSError(
            f"skill path escapes configured root {resolved_root}: {resolved_path}"
        ) from exc
    if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
        descriptor = _open_skill_beneath(resolved_root, relative_path)
        return _read_open_skill_file(descriptor, path=resolved_path, max_bytes=limit)

    # Windows and other platforms without openat/O_NOFOLLOW support get a
    # before/after identity check. The opened descriptor must still identify
    # the same regular file reached through the same non-symlink parent chain.
    before = _path_identity_chain(resolved_root, resolved_path)
    descriptor = os.open(resolved_path, _skill_open_flags())
    try:
        opened_metadata = os.fstat(descriptor)
        after = _path_identity_chain(resolved_root, resolved_path)
        if len(before) != len(after) or any(
            before_path != after_path
            or not _same_file_identity(before_metadata, after_metadata)
            for (before_path, before_metadata), (after_path, after_metadata) in zip(
                before, after, strict=True
            )
        ):
            raise OSError(f"skill path changed while opening: {resolved_path}")
        if not _same_file_identity(opened_metadata, after[-1][1]):
            raise OSError(f"skill file changed while opening: {resolved_path}")
    except BaseException:
        os.close(descriptor)
        raise
    return _read_open_skill_file(descriptor, path=resolved_path, max_bytes=limit)


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


def _parse_skill_metadata_text(content: str, *, path: Path) -> SkillMetadata:
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

    metadata = SkillMetadata(
        name=name,
        path=path.resolve(),
        triggers=_normalize_list(frontmatter.get("triggers")),
        requires=_normalize_list(frontmatter.get("requires")),
        profiles=_normalize_list(profile_value),
        enforcement=_normalize_list(frontmatter.get("enforcement")),
        verification=_normalize_list(
            frontmatter.get("verification")
            or frontmatter.get("checks")
            or frontmatter.get("quality_gates")
        ),
    )
    validate_skill_metadata(metadata)
    return metadata


def validate_skill_metadata(skill: SkillMetadata) -> None:
    """Reject metadata that cannot be delivered completely within public limits."""
    if len(skill.name) > MAX_SKILL_NAME_CHARS:
        raise ValueError(
            f"Skill name exceeds {MAX_SKILL_NAME_CHARS} characters: {skill.path}"
        )
    for field_name in (
        "triggers",
        "requires",
        "profiles",
        "enforcement",
        "verification",
    ):
        values = getattr(skill, field_name)
        if len(values) > MAX_SKILL_METADATA_ITEMS:
            raise ValueError(
                f"Skill {field_name} exceeds {MAX_SKILL_METADATA_ITEMS} items: "
                f"{skill.path}"
            )
        if any(len(value) > MAX_SKILL_METADATA_ITEM_CHARS for value in values):
            raise ValueError(
                f"Skill {field_name} item exceeds "
                f"{MAX_SKILL_METADATA_ITEM_CHARS} characters: {skill.path}"
            )


def parse_skill_metadata(path: Path) -> SkillMetadata:
    """Parse SKILL.md metadata frontmatter."""
    resolved_path = path.expanduser().resolve(strict=True)
    return _parse_skill_metadata_text(
        _read_skill_text(resolved_path),
        path=resolved_path,
    )


def read_skill_body(
    path: Path,
    *,
    max_chars: int = MAX_SKILL_BODY_CHARS,
    trusted_root: Path | None = None,
) -> tuple[str, bool]:
    """Read a bounded instruction body, excluding valid leading frontmatter."""
    content = _read_skill_text(path, trusted_root=trusted_root)
    lines = content.splitlines()
    body_start = 0
    if lines and lines[0].strip() == "---":
        body_start = -1
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                body_start = index + 1
                break
        if body_start < 0:
            return "", False

    body = "\n".join(lines[body_start:]).strip()
    return truncate_skill_body(body, max_chars=max_chars)


def truncate_skill_body(body: str, *, max_chars: int) -> tuple[str, bool]:
    """Bound a body and close Markdown fences opened by truncation."""
    limit = max(0, max_chars)
    if len(body) <= limit:
        return body, False
    if limit <= 3:
        return body[:limit], True

    excerpt = body[: limit - 3].rstrip() + "..."
    for _ in range(4):
        open_fence: tuple[str, int] | None = None
        for line in excerpt.splitlines():
            match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if not match:
                continue
            marker = match.group(1)
            trailing = match.group(2)
            marker_char = marker[0]
            if open_fence is not None:
                open_char, open_length = open_fence
                if (
                    marker_char == open_char
                    and len(marker) >= open_length
                    and not trailing.strip()
                ):
                    open_fence = None
                continue
            if marker_char == "`" and "`" in trailing:
                continue
            open_fence = (marker_char, len(marker))
        if open_fence is None:
            return excerpt, True

        closure = "\n" + (open_fence[0] * open_fence[1])
        if len(excerpt) + len(closure) <= limit:
            return excerpt + closure, True
        content_limit = limit - len(closure) - 3
        if content_limit <= 0:
            return excerpt[:limit], True
        excerpt = body[:content_limit].rstrip() + "..."
    return excerpt[:limit], True


def discover_skills(skill_roots: list[Path], profile: str | None = None) -> list[SkillMetadata]:
    """Discover SKILL.md files and filter by profile when requested."""
    discovered: list[SkillMetadata] = []
    for root in skill_roots:
        try:
            resolved_root = root.expanduser().resolve()
        except OSError:
            continue
        if not resolved_root.exists():
            continue
        try:
            candidates = sorted(
                resolved_root.rglob("SKILL.md"),
                key=lambda candidate: str(candidate),
            )
        except OSError:
            continue
        for candidate in candidates:
            try:
                resolved_candidate = candidate.resolve(strict=True)
                resolved_candidate.relative_to(resolved_root)
                content = _read_skill_text(
                    resolved_candidate,
                    trusted_root=resolved_root,
                )
                metadata = _parse_skill_metadata_text(
                    content,
                    path=resolved_candidate,
                )
            except ValueError:
                # A symlink must not turn a configured instruction root into
                # an implicit trust grant for an unrelated filesystem path.
                continue
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


def _bounded_metadata_text(value: str, *, max_chars: int) -> tuple[str, bool]:
    normalized = str(value).strip()
    if len(normalized) <= max_chars:
        return normalized, False
    if max_chars <= 3:
        return normalized[:max_chars], True
    return normalized[: max_chars - 3].rstrip() + "...", True


def _bounded_metadata_list(values: list[str]) -> tuple[list[str], bool]:
    bounded: list[str] = []
    truncated = len(values) > MAX_SKILL_METADATA_ITEMS
    for value in values[:MAX_SKILL_METADATA_ITEMS]:
        item, item_truncated = _bounded_metadata_text(
            value,
            max_chars=MAX_SKILL_METADATA_ITEM_CHARS,
        )
        if item:
            bounded.append(item)
        truncated = truncated or item_truncated
    return bounded, truncated


def bounded_skill_metadata(skill: SkillMetadata) -> dict[str, Any]:
    """Render bounded metadata suitable for artifacts, prompts, and tool output."""
    name, name_truncated = _bounded_metadata_text(
        skill.name,
        max_chars=MAX_SKILL_NAME_CHARS,
    )
    path, path_truncated = _bounded_metadata_text(
        str(skill.path),
        max_chars=MAX_SKILL_PATH_CHARS,
    )
    payload: dict[str, Any] = {"name": name, "path": path}
    truncated_fields: list[str] = []
    if name_truncated:
        truncated_fields.append("name")
    if path_truncated:
        truncated_fields.append("path")
    for field_name in ("triggers", "requires", "enforcement", "verification"):
        values, truncated = _bounded_metadata_list(getattr(skill, field_name))
        payload[field_name] = values
        if truncated:
            truncated_fields.append(field_name)
    if truncated_fields:
        payload["metadata_truncated"] = truncated_fields
    return payload


def build_skill_matches(
    prompt: str,
    skill_roots: list[Path],
    *,
    profile: str | None = None,
    top_k: int = 5,
    max_body_chars: int = MAX_SKILL_BODY_CHARS,
    max_total_body_chars: int = MAX_SKILL_BODIES_CHARS,
    max_body_matches: int = MAX_SKILL_BODY_MATCHES,
) -> list[dict[str, Any]]:
    """Return deterministic, bounded match records for a task prompt."""
    resolved_roots = [root.expanduser().resolve() for root in skill_roots]
    ranked: list[tuple[int, int, SkillMetadata]] = []
    seen_names: set[str] = set()
    for skill in discover_skills(skill_roots, profile=profile):
        name_key = skill.name.casefold()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        score = score_skill_relevance(prompt, skill)
        if score > 0:
            root_priority = len(resolved_roots)
            for index, root in enumerate(resolved_roots):
                try:
                    skill.path.relative_to(root)
                except ValueError:
                    continue
                root_priority = index
                break
            ranked.append((score, root_priority, skill))
    ranked.sort(
        key=lambda item: (-item[0], item[1], item[2].name.lower(), str(item[2].path))
    )

    remaining = max(0, max_total_body_chars)
    matches: list[dict[str, Any]] = []
    match_limit = min(max(top_k, 0), MAX_SKILL_MATCHES)
    for index, (score, _root_priority, skill) in enumerate(ranked[:match_limit]):
        body_omitted = ""
        if index >= max(0, max_body_matches):
            body_limit = 0
            body_omitted = "match_limit"
        elif remaining <= 0:
            body_limit = 0
            body_omitted = "aggregate_limit"
        elif max_body_chars <= 0:
            body_limit = 0
            body_omitted = "body_limit"
        else:
            body_limit = min(max_body_chars, remaining)

        if body_omitted:
            body, body_truncated = "", False
        else:
            try:
                trusted_root = (
                    resolved_roots[_root_priority]
                    if _root_priority < len(resolved_roots)
                    else None
                )
                body, body_truncated = read_skill_body(
                    skill.path,
                    max_chars=body_limit,
                    trusted_root=trusted_root,
                )
            except OSError:
                body, body_truncated = "", False
        remaining = max(0, remaining - len(body))
        match = {
            "score": score,
            **bounded_skill_metadata(skill),
            "body": body,
            "body_truncated": body_truncated,
            "body_chars": len(body),
        }
        if body_omitted:
            match["body_omitted"] = body_omitted
        matches.append(match)
    return matches
