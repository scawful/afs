"""Zelda research toolkit for AFS mission runner.

Wraps read-only scanning of Oracle of Secrets, YAZE, and Mesen2 codebases
to feed the zelda-research mission's observe phase.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_git(repo: Path, *args: str, timeout: int = 10) -> str | None:
    """Run a read-only git command, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("git command failed in %s: %s", repo, exc)
    return None


def _count_pattern_in_files(files: list[Path], pattern: re.Pattern[str]) -> int:
    """Count total matches of *pattern* across *files*."""
    count = 0
    for f in files:
        try:
            text = f.read_text(errors="replace")
            count += len(pattern.findall(text))
        except OSError:
            continue
    return count


def _extract_labels_from_asm(files: list[Path]) -> set[str]:
    """Extract label definitions (Name:) from ASM files."""
    label_pat = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:", re.MULTILINE)
    labels: set[str] = set()
    for f in files:
        try:
            text = f.read_text(errors="replace")
            labels.update(label_pat.findall(text))
        except OSError:
            continue
    return labels


class ZeldaToolkit:
    """Read-only scanners for Zelda-related codebases."""

    # ------------------------------------------------------------------
    # Oracle of Secrets
    # ------------------------------------------------------------------
    def scan_oos_codebase(self, repo_path: Path) -> dict[str, Any]:
        """Scan Oracle of Secrets ASM repository.

        Returns counts of routines, TODOs/FIXMEs, include files, and
        a git status snapshot.
        """
        result: dict[str, Any] = {"available": False, "path": str(repo_path)}

        if not repo_path.is_dir():
            result["error"] = "directory not found"
            return result

        asm_files = sorted(repo_path.rglob("*.asm"))
        result["available"] = True
        result["asm_file_count"] = len(asm_files)

        # Count subroutine-like patterns (RTL, RTS, JSL, JSR)
        routine_pat = re.compile(r"\b(RTL|RTS|JSL|JSR)\b", re.IGNORECASE)
        result["routine_instructions"] = _count_pattern_in_files(asm_files, routine_pat)

        # TODOs / FIXMEs
        todo_pat = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
        result["todo_count"] = _count_pattern_in_files(asm_files, todo_pat)

        # Include directives (incsrc, incbin, %include, etc.)
        include_pat = re.compile(r"^\s*(incsrc|incbin|%include)\s+", re.MULTILINE | re.IGNORECASE)
        include_files: list[str] = []
        for f in asm_files:
            try:
                text = f.read_text(errors="replace")
                for m in include_pat.finditer(text):
                    line = text[m.start() : text.index("\n", m.start())] if "\n" in text[m.start() :] else text[m.start() :]
                    include_files.append(line.strip())
            except OSError:
                continue
        result["include_directives"] = include_files[:200]  # cap for sanity

        # Labels
        labels = _extract_labels_from_asm(asm_files)
        result["label_count"] = len(labels)

        # Git status
        branch = _safe_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        last_commit = _safe_git(repo_path, "log", "-1", "--format=%H %s")
        result["git_branch"] = branch
        result["git_last_commit"] = last_commit

        return result

    # ------------------------------------------------------------------
    # YAZE
    # ------------------------------------------------------------------
    def scan_yaze_status(self, repo_path: Path) -> dict[str, Any]:
        """Check YAZE build status, z3ed availability, recent changes."""
        result: dict[str, Any] = {"available": False, "path": str(repo_path)}

        if not repo_path.is_dir():
            result["error"] = "directory not found"
            return result

        result["available"] = True

        # Check for z3ed CLI source
        z3ed_dir = repo_path / "src" / "cli"
        result["z3ed_cli_present"] = z3ed_dir.is_dir()

        # Check for build directory
        build_dir = repo_path / "build"
        result["build_dir_exists"] = build_dir.is_dir()

        # Check for z3ed binary
        z3ed_binary = build_dir / "z3ed" if build_dir.is_dir() else None
        result["z3ed_binary_exists"] = z3ed_binary is not None and z3ed_binary.exists()

        # Check CMakeLists.txt exists
        result["has_cmake"] = (repo_path / "CMakeLists.txt").is_file()

        # Recent git activity
        branch = _safe_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        last_commit = _safe_git(repo_path, "log", "-1", "--format=%H %s")
        recent_log = _safe_git(repo_path, "log", "--oneline", "-5")
        result["git_branch"] = branch
        result["git_last_commit"] = last_commit
        result["git_recent_log"] = recent_log.splitlines() if recent_log else []

        # Source file counts
        cpp_files = list(repo_path.rglob("*.cc")) + list(repo_path.rglob("*.cpp"))
        header_files = list(repo_path.rglob("*.h"))
        result["cpp_file_count"] = len(cpp_files)
        result["header_file_count"] = len(header_files)

        # Check for zelda3 module
        zelda3_dir = repo_path / "src" / "zelda3"
        result["zelda3_module_present"] = zelda3_dir.is_dir()

        return result

    # ------------------------------------------------------------------
    # Mesen2
    # ------------------------------------------------------------------
    def scan_mesen2_status(self, fork_path: Path) -> dict[str, Any]:
        """Check Mesen2 fork status, socket API availability, recent patches."""
        result: dict[str, Any] = {"available": False, "path": str(fork_path)}

        if not fork_path.is_dir():
            result["error"] = "directory not found"
            return result

        result["available"] = True

        # Socket API / tools
        tools_dir = fork_path / "tools"
        result["tools_dir_exists"] = tools_dir.is_dir()

        mesen2ctl = tools_dir / "mesen2ctl" if tools_dir.is_dir() else None
        result["mesen2ctl_present"] = mesen2ctl is not None and mesen2ctl.exists()

        # Check for socket-related source files
        socket_files: list[str] = []
        for pattern in ("*socket*", "*Socket*", "*SOCKET*"):
            for f in fork_path.rglob(pattern):
                socket_files.append(str(f.relative_to(fork_path)))
        result["socket_api_files"] = socket_files[:50]

        # Build directory
        build_dir = fork_path / "build"
        result["build_dir_exists"] = build_dir.is_dir()

        # Git status
        branch = _safe_git(fork_path, "rev-parse", "--abbrev-ref", "HEAD")
        last_commit = _safe_git(fork_path, "log", "-1", "--format=%H %s")
        recent_log = _safe_git(fork_path, "log", "--oneline", "-5")
        result["git_branch"] = branch
        result["git_last_commit"] = last_commit
        result["git_recent_log"] = recent_log.splitlines() if recent_log else []

        # Lua scripts (often used for debugging)
        lua_files = list(fork_path.rglob("*.lua"))
        result["lua_script_count"] = len(lua_files)

        return result

    # ------------------------------------------------------------------
    # Cross-reference symbols
    # ------------------------------------------------------------------
    def cross_reference_symbols(
        self,
        oos_path: Path,
        disasm_path: Path,
    ) -> dict[str, Any]:
        """Find shared symbols/labels between OoS and a disassembly tree.

        Both paths should point to directories containing .asm files.
        Returns sets of shared labels and labels unique to each side.
        """
        result: dict[str, Any] = {
            "oos_path": str(oos_path),
            "disasm_path": str(disasm_path),
            "available": False,
        }

        oos_exists = oos_path.is_dir()
        disasm_exists = disasm_path.is_dir()
        result["oos_exists"] = oos_exists
        result["disasm_exists"] = disasm_exists

        if not oos_exists and not disasm_exists:
            result["error"] = "neither directory found"
            return result

        oos_asm = sorted(oos_path.rglob("*.asm")) if oos_exists else []
        disasm_asm = sorted(disasm_path.rglob("*.asm")) if disasm_exists else []

        oos_labels = _extract_labels_from_asm(oos_asm)
        disasm_labels = _extract_labels_from_asm(disasm_asm)

        shared = sorted(oos_labels & disasm_labels)
        oos_only = sorted(oos_labels - disasm_labels)
        disasm_only = sorted(disasm_labels - oos_labels)

        result["available"] = True
        result["oos_label_count"] = len(oos_labels)
        result["disasm_label_count"] = len(disasm_labels)
        result["shared_labels"] = shared[:500]
        result["shared_count"] = len(shared)
        result["oos_only_count"] = len(oos_only)
        result["disasm_only_count"] = len(disasm_only)
        # Include a sample of unique labels (capped) for orientation
        result["oos_only_sample"] = oos_only[:100]
        result["disasm_only_sample"] = disasm_only[:100]

        return result
