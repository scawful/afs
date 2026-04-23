"""Repo-local software quality policy loading and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import tomllib


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_rel_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _pattern_variants(pattern: str) -> list[str]:
    normalized = _normalize_rel_path(pattern)
    if not normalized:
        return []
    variants = {normalized}
    pending = [normalized]
    while pending:
        current = pending.pop()
        next_variants: list[str] = []
        if "/**/" in current:
            next_variants.append(current.replace("/**/", "/", 1))
        if current.startswith("**/"):
            next_variants.append(current[3:])
        for candidate in next_variants:
            if candidate and candidate not in variants:
                variants.add(candidate)
                pending.append(candidate)
    return list(variants)


def _path_matches(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    posix_path = PurePosixPath(_normalize_rel_path(path))
    for raw_pattern in patterns:
        for pattern in _pattern_variants(raw_pattern):
            if posix_path.match(pattern) or posix_path.match(f"**/{pattern}"):
                return True
    return False


@dataclass(frozen=True)
class RepoRiskRule:
    name: str
    paths: list[str] = field(default_factory=list)
    message: str = ""
    risk: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoRiskRule:
        return cls(
            name=str(data.get("name", "")).strip(),
            paths=_as_str_list(data.get("paths")),
            message=str(data.get("message", "")).strip(),
            risk=str(data.get("risk", "")).strip(),
        )


@dataclass(frozen=True)
class RepoAntiPatternRule:
    name: str
    pattern: str
    paths: list[str] = field(default_factory=list)
    message: str = ""
    regex: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoAntiPatternRule:
        return cls(
            name=str(data.get("name", "")).strip(),
            pattern=str(data.get("pattern", "")).strip(),
            paths=_as_str_list(data.get("paths")),
            message=str(data.get("message", "")).strip(),
            regex=bool(data.get("regex", False)),
        )


@dataclass(frozen=True)
class RepoPolicy:
    path: Path | None = None
    review_focus: list[str] = field(default_factory=list)
    risk_categories: list[str] = field(default_factory=list)
    design_constraints: list[str] = field(default_factory=list)
    planning_principles: list[str] = field(default_factory=list)
    path_risks: list[RepoRiskRule] = field(default_factory=list)
    anti_patterns: list[RepoAntiPatternRule] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.path is not None

    @classmethod
    def empty(cls) -> RepoPolicy:
        return cls()

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        path: Path | None = None,
    ) -> RepoPolicy:
        review = data.get("review", {}) if isinstance(data.get("review"), dict) else {}
        design = data.get("design", {}) if isinstance(data.get("design"), dict) else {}
        planning = data.get("planning", {}) if isinstance(data.get("planning"), dict) else {}

        risk_rules_raw = review.get("path_risks", [])
        risk_rules = [
            RepoRiskRule.from_dict(item)
            for item in risk_rules_raw
            if isinstance(item, dict)
        ]
        anti_patterns_raw = data.get("anti_patterns", [])
        anti_patterns = [
            RepoAntiPatternRule.from_dict(item)
            for item in anti_patterns_raw
            if isinstance(item, dict)
        ]
        return cls(
            path=path,
            review_focus=_as_str_list(review.get("focus")),
            risk_categories=_as_str_list(review.get("risk_categories")),
            design_constraints=_as_str_list(design.get("constraints")),
            planning_principles=_as_str_list(planning.get("principles")),
            path_risks=risk_rules,
            anti_patterns=anti_patterns,
        )


def find_repo_policy(start_dir: Path | None = None) -> Path | None:
    """Walk upward from *start_dir* looking for ``.afs/policy.toml``."""
    current = (start_dir or Path.cwd()).expanduser().resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".afs" / "policy.toml"
        if candidate.exists():
            return candidate.resolve()
    return None


def load_repo_policy(
    policy_path: str | Path | None = None,
    *,
    start_dir: Path | None = None,
) -> RepoPolicy:
    """Load a repo-local policy file when present."""
    resolved_path: Path | None
    if policy_path is not None:
        resolved_path = Path(policy_path).expanduser().resolve()
    else:
        resolved_path = find_repo_policy(start_dir)
    if resolved_path is None or not resolved_path.exists():
        return RepoPolicy.empty()
    with open(resolved_path, "rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        return RepoPolicy.empty()
    return RepoPolicy.from_dict(data, path=resolved_path)


def evaluate_repo_policy(
    policy: RepoPolicy,
    *,
    repo_root: Path,
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate repo policy against the current changed file scope."""
    normalized_paths = [
        _normalize_rel_path(path)
        for path in (changed_paths or [])
        if _normalize_rel_path(path)
    ]

    matched_risks: list[dict[str, Any]] = []
    for rule in policy.path_risks:
        if not rule.name:
            continue
        matching_paths = [path for path in normalized_paths if _path_matches(path, rule.paths)]
        if not matching_paths:
            continue
        matched_risks.append(
            {
                "name": rule.name,
                "risk": rule.risk or rule.name,
                "message": rule.message,
                "paths": matching_paths,
            }
        )

    anti_pattern_hits: list[dict[str, Any]] = []
    for rule in policy.anti_patterns:
        if not rule.pattern:
            continue
        for rel_path in normalized_paths:
            if not _path_matches(rel_path, rule.paths):
                continue
            file_path = (repo_root / rel_path).resolve()
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError:
                continue
            except UnicodeDecodeError:
                continue
            if rule.regex:
                if not re.search(rule.pattern, content, re.MULTILINE):
                    continue
            else:
                if rule.pattern not in content:
                    continue
            anti_pattern_hits.append(
                {
                    "name": rule.name or rule.pattern,
                    "path": rel_path,
                    "message": rule.message,
                    "pattern": rule.pattern,
                }
            )

    return {
        "available": policy.available,
        "path": str(policy.path) if policy.path else "",
        "review_focus": list(policy.review_focus),
        "risk_categories": list(policy.risk_categories),
        "design_constraints": list(policy.design_constraints),
        "planning_principles": list(policy.planning_principles),
        "matched_risks": matched_risks,
        "anti_pattern_hits": anti_pattern_hits,
    }
