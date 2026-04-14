"""Cheap codebase summaries for agent bootstrap and CLI exploration."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

_DEFAULT_MAX_SCAN_FILES = 400
_DEFAULT_MAX_SCAN_DEPTH = 3
_DEFAULT_MAX_SAMPLE_PATHS = 12
_DEFAULT_MAX_TOP_LEVEL = 12

_IGNORED_DIRS = {
    ".context",
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}

_ALLOWED_HIDDEN_DIRS = {".github"}

_ROOT_KEY_FILES = [
    "AGENTS.md",
    "README.md",
    "README.rst",
    "README.txt",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "justfile",
    "WORKSPACE",
    "WORKSPACE.toml",
    "afs.toml",
    "tsconfig.json",
]

_MANIFEST_ECOSYSTEMS = {
    "Cargo.toml": "rust",
    "Gemfile": "ruby",
    "Package.swift": "swift",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "go.mod": "go",
    "mix.exs": "elixir",
    "package.json": "node",
    "pom.xml": "java",
    "pyproject.toml": "python",
}

_LANGUAGE_BY_EXTENSION = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".json": "json",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".swift": "swift",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "shell",
}

_SOURCE_ROOT_NAMES = {"app", "apps", "cmd", "crates", "extensions", "lib", "libs", "packages", "src"}
_TEST_ROOT_NAMES = {"__tests__", "spec", "specs", "test", "tests"}
_DOC_ROOT_NAMES = {"doc", "docs", "documentation"}
_SCRIPT_ROOT_NAMES = {"bin", "script", "scripts", "tools"}
_WORKFLOW_ROOT_NAMES = {
    "checkpoints",
    "config",
    "data",
    "datasets",
    "eval",
    "evaluation",
    "experiments",
    "models",
    "prompts",
    "training",
}
_SOURCE_ROOT_PRIORITY = {
    "src": 0,
    "app": 1,
    "apps": 1,
    "lib": 2,
    "libs": 2,
    "packages": 3,
    "extensions": 4,
    "cmd": 5,
    "crates": 6,
}


def infer_project_root(context_path: Path) -> Path:
    """Infer the project/workspace root that owns a context path."""
    resolved = context_path.expanduser().resolve()
    if resolved.name == ".context":
        return resolved.parent
    if resolved.is_dir() and (resolved / ".context").is_dir():
        return resolved
    if resolved.is_file():
        return resolved.parent
    return resolved


def build_codebase_summary(
    context_path: Path,
    *,
    max_scan_files: int = _DEFAULT_MAX_SCAN_FILES,
    max_scan_depth: int = _DEFAULT_MAX_SCAN_DEPTH,
    max_sample_paths: int = _DEFAULT_MAX_SAMPLE_PATHS,
    max_top_level: int = _DEFAULT_MAX_TOP_LEVEL,
) -> dict[str, Any]:
    """Build a lightweight codebase structure summary from a context path."""
    project_root = infer_project_root(context_path)
    summary: dict[str, Any] = {
        "project_root": str(project_root),
        "has_git": False,
        "ecosystems": [],
        "manifests": [],
        "nested_manifests": [],
        "top_level_dirs": [],
        "top_level_files": [],
        "source_roots": [],
        "test_roots": [],
        "docs_roots": [],
        "script_roots": [],
        "workflow_roots": [],
        "language_hints": {},
        "sample_paths": [],
        "scan": {
            "files_scanned": 0,
            "max_scan_files": max_scan_files,
            "max_scan_depth": max_scan_depth,
            "truncated": False,
        },
    }
    if not project_root.is_dir():
        return summary

    summary["has_git"] = (project_root / ".git").exists()

    try:
        children = sorted(project_root.iterdir(), key=lambda entry: entry.name.lower())
    except OSError:
        return summary

    ecosystems: set[str] = set()
    top_level_dirs: list[str] = []
    top_level_files: list[str] = []
    manifests: list[str] = []

    for child in children:
        name = child.name
        if child.is_dir():
            if _should_skip_dir(name):
                continue
            top_level_dirs.append(name)
            continue
        if not child.is_file():
            continue
        top_level_files.append(name)
        ecosystem = _MANIFEST_ECOSYSTEMS.get(name)
        if ecosystem:
            manifests.append(name)
            ecosystems.add(ecosystem)

    top_level_dirs.sort(key=_dir_rank)
    summary["top_level_dirs"] = top_level_dirs[:max_top_level]
    preferred_top_level_files = [name for name in _ROOT_KEY_FILES if name in top_level_files]
    summary["top_level_files"] = preferred_top_level_files[:max_top_level]
    if not summary["top_level_files"]:
        summary["top_level_files"] = top_level_files[:max_top_level]
    summary["manifests"] = manifests[:max_top_level]
    summary["source_roots"] = [name for name in top_level_dirs if name in _SOURCE_ROOT_NAMES][:max_top_level]
    summary["test_roots"] = [name for name in top_level_dirs if name in _TEST_ROOT_NAMES][:max_top_level]
    summary["docs_roots"] = [name for name in top_level_dirs if name in _DOC_ROOT_NAMES][:max_top_level]
    summary["script_roots"] = [name for name in top_level_dirs if name in _SCRIPT_ROOT_NAMES][:max_top_level]
    summary["workflow_roots"] = [name for name in top_level_dirs if name in _WORKFLOW_ROOT_NAMES][:max_top_level]

    language_counts: Counter[str] = Counter()
    nested_manifests: list[str] = []
    sample_candidates: list[str] = []
    sample_seen: set[str] = set()
    files_scanned = 0
    truncated = False

    for current_root, dirs, files in os.walk(project_root):
        rel_root = Path(current_root).resolve().relative_to(project_root)
        depth = len(rel_root.parts)
        dirs[:] = [
            name
            for name in sorted(dirs, key=_dir_rank)
            if not _should_skip_dir(name) and depth < max_scan_depth
        ]

        for filename in sorted(files, key=str.lower):
            if files_scanned >= max_scan_files:
                truncated = True
                dirs[:] = []
                break

            rel_path = rel_root / filename if rel_root.parts else Path(filename)
            rel_text = rel_path.as_posix()
            files_scanned += 1

            ecosystem = _MANIFEST_ECOSYSTEMS.get(filename)
            if ecosystem:
                ecosystems.add(ecosystem)
                if rel_text not in manifests and rel_text not in nested_manifests:
                    nested_manifests.append(rel_text)

            language = _LANGUAGE_BY_EXTENSION.get(Path(filename).suffix.lower())
            if language:
                language_counts[language] += 1

            if _is_interesting_path(rel_path) and rel_text not in sample_seen:
                sample_seen.add(rel_text)
                sample_candidates.append(rel_text)

    summary["ecosystems"] = sorted(ecosystems)
    summary["nested_manifests"] = nested_manifests[:max_top_level]
    summary["language_hints"] = {
        language: count
        for language, count in sorted(
            language_counts.items(),
            key=_language_rank,
        )
    }
    summary["sample_paths"] = sorted(sample_candidates, key=_sample_rank)[:max_sample_paths]
    summary["scan"] = {
        "files_scanned": files_scanned,
        "max_scan_files": max_scan_files,
        "max_scan_depth": max_scan_depth,
        "truncated": truncated,
    }
    return summary


def render_codebase_summary(summary: dict[str, Any]) -> str:
    """Render a compact human-readable codebase summary."""
    lines = [f"project_root: {summary.get('project_root', '') or '(unknown)'}"]
    lines.append(f"git: {str(bool(summary.get('has_git', False))).lower()}")

    for key in ("ecosystems", "manifests", "nested_manifests", "top_level_dirs", "top_level_files"):
        values = summary.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{key}: {', '.join(str(value) for value in values)}")

    for key in ("source_roots", "test_roots", "docs_roots", "script_roots", "workflow_roots"):
        values = summary.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{key}: {', '.join(str(value) for value in values)}")

    language_hints = summary.get("language_hints")
    if isinstance(language_hints, dict) and language_hints:
        lines.append(
            "language_hints: "
            + ", ".join(f"{name}={count}" for name, count in language_hints.items())
        )

    sample_paths = summary.get("sample_paths")
    if isinstance(sample_paths, list) and sample_paths:
        lines.append("sample_paths:")
        lines.extend(f"- {path}" for path in sample_paths)

    scan = summary.get("scan")
    if isinstance(scan, dict):
        lines.append(
            "scan: "
            + f"files_scanned={scan.get('files_scanned', 0)} "
            + f"depth<={scan.get('max_scan_depth', 0)} "
            + f"truncated={str(bool(scan.get('truncated', False))).lower()}"
        )

    return "\n".join(lines)


def _should_skip_dir(name: str) -> bool:
    return name in _IGNORED_DIRS or (name.startswith(".") and name not in _ALLOWED_HIDDEN_DIRS)


def _is_interesting_path(rel_path: Path) -> bool:
    name = rel_path.name
    if name in _ROOT_KEY_FILES or name in _MANIFEST_ECOSYSTEMS:
        return True
    if name.lower().startswith("readme"):
        return True
    extension = rel_path.suffix.lower()
    if extension not in _LANGUAGE_BY_EXTENSION:
        return False
    if len(rel_path.parts) <= 3:
        return True
    return rel_path.parts[0] in (
        *_SOURCE_ROOT_NAMES,
        *_TEST_ROOT_NAMES,
        *_DOC_ROOT_NAMES,
        *_SCRIPT_ROOT_NAMES,
        *_WORKFLOW_ROOT_NAMES,
    )


def _dir_rank(name: str) -> tuple[int, int, str]:
    if name in _SOURCE_ROOT_NAMES:
        return (0, _SOURCE_ROOT_PRIORITY.get(name, 99), name)
    if name in _TEST_ROOT_NAMES:
        return (1, 0, name)
    if name in _DOC_ROOT_NAMES:
        return (2, 0, name)
    if name in _SCRIPT_ROOT_NAMES:
        return (3, 0, name)
    if name in _WORKFLOW_ROOT_NAMES:
        return (4, 0, name)
    if name.startswith("."):
        return (6, 0, name)
    return (5, 0, name)


def _language_rank(item: tuple[str, int]) -> tuple[int, int, str]:
    language, count = item
    non_code = {"json", "markdown", "toml", "yaml"}
    return (1 if language in non_code else 0, -count, language)


def _sample_rank(rel_text: str) -> tuple[int, int, str]:
    rel_path = Path(rel_text)
    first = rel_path.parts[0] if rel_path.parts else ""
    name = rel_path.name
    if first in _SOURCE_ROOT_NAMES:
        bucket = 0
        detail = _SOURCE_ROOT_PRIORITY.get(first, 99)
    elif first in _TEST_ROOT_NAMES:
        bucket = 1
        detail = 0
    elif first in _DOC_ROOT_NAMES:
        bucket = 2
        detail = 0
    elif first in _SCRIPT_ROOT_NAMES:
        bucket = 3
        detail = 0
    elif first in _WORKFLOW_ROOT_NAMES:
        bucket = 4
        detail = 0
    elif name in _ROOT_KEY_FILES or name in _MANIFEST_ECOSYSTEMS:
        bucket = 5
        detail = 0
    elif first == ".github":
        bucket = 6
        detail = 0
    else:
        bucket = 7
        detail = 0
    return (bucket, detail, len(rel_path.parts), rel_text)
