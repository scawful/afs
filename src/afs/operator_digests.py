"""Compact digests for noisy operator output."""

from __future__ import annotations

import re
from typing import Any

KIND_CHOICES = ("auto", "pytest", "traceback", "grep", "diffstat", "diagnostic", "generic")

_PYTEST_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<label>passed|failed|error|errors|skipped|xfailed|xpassed|warnings?)\b"
)
_PYTEST_FAILURE_RE = re.compile(r"^(?P<status>FAILED|ERROR)\s+(?P<name>.+)$")
_TRACEBACK_FRAME_RE = re.compile(
    r'^  File "(?P<path>.+?)", line (?P<line>\d+), in (?P<function>.+)$'
)
_GREP_LINE_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<text>.*)$")
_DIFFSTAT_FILE_RE = re.compile(
    r"^(?P<path>.+?)\s+\|\s+(?P<changes>\d+)(?:\s+(?P<bar>[+\-]+))?$"
)
_DIFFSTAT_SUMMARY_RE = re.compile(
    r"(?P<files>\d+)\s+files?\s+changed"
    r"(?:,\s*(?P<insertions>\d+)\s+insertions?\(\+\))?"
    r"(?:,\s*(?P<deletions>\d+)\s+deletions?\(-\))?"
)
_TSC_DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>.+)\((?P<line>\d+),(?P<column>\d+)\): "
    r"(?P<severity>error|warning) (?P<code>TS\d+): (?P<message>.+)$"
)
_MYPY_DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): (?P<severity>error|warning|note): "
    r"(?P<message>.+?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)
_RUFF_DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): "
    r"(?P<code>[A-Z]{1,6}\d{2,4})(?: \[[^\]]+\])? (?P<message>.+)$"
)
_ESLINT_DIAGNOSTIC_RE = re.compile(
    r"^\s*(?P<line>\d+):(?P<column>\d+)\s+"
    r"(?P<severity>error|warning)\s+"
    r"(?P<message>.+?)\s{2,}(?P<code>@?[\w./-]+)\s*$"
)
_ESLINT_SUMMARY_RE = re.compile(
    r"^[✖xX]\s+(?P<problems>\d+)\s+problems?"
    r"(?:\s+\((?P<errors>\d+)\s+errors?(?:,\s*(?P<warnings>\d+)\s+warnings?)?\))?$"
)
_TSC_SUMMARY_RE = re.compile(
    r"^Found (?P<errors>\d+) errors?(?: in (?P<files>\d+) files?)?",
    re.IGNORECASE,
)
_MYPY_SUMMARY_RE = re.compile(
    r"^Found (?P<errors>\d+) errors?(?: in (?P<files>\d+) files?)?",
    re.IGNORECASE,
)


def digest_operator_output(
    text: str,
    *,
    kind: str = "auto",
    max_items: int = 5,
) -> dict[str, Any]:
    """Return a compact digest for noisy command output."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    requested_kind = (kind or "auto").strip().lower()
    if requested_kind not in KIND_CHOICES:
        raise ValueError(f"unknown digest kind: {requested_kind}")

    effective_max_items = max(1, int(max_items))
    detected_kind = _detect_kind(text) if requested_kind == "auto" else requested_kind

    if detected_kind == "pytest":
        payload = _digest_pytest(text, max_items=effective_max_items)
    elif detected_kind == "traceback":
        payload = _digest_traceback(text, max_items=effective_max_items)
    elif detected_kind == "grep":
        payload = _digest_grep(text, max_items=effective_max_items)
    elif detected_kind == "diffstat":
        payload = _digest_diffstat(text, max_items=effective_max_items)
    elif detected_kind == "diagnostic":
        payload = _digest_diagnostic(text, max_items=effective_max_items)
    else:
        payload = _digest_generic(text, max_items=effective_max_items)

    payload["requested_kind"] = requested_kind
    payload["kind"] = detected_kind
    payload["line_count"] = len(text.splitlines())
    payload["digest_text"] = render_operator_digest(payload)
    return payload


def render_operator_digest(payload: dict[str, Any]) -> str:
    """Render a digest payload into compact markdown-ish text."""
    lines = [
        f"Kind: {payload.get('kind', 'generic')}",
        f"Summary: {payload.get('summary', 'No summary')}",
    ]
    highlights = payload.get("highlights", [])
    if isinstance(highlights, list) and highlights:
        lines.append("Highlights:")
        for item in highlights:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _detect_kind(text: str) -> str:
    if _looks_like_pytest(text):
        return "pytest"
    if "Traceback (most recent call last):" in text:
        return "traceback"
    if _looks_like_diagnostic(text):
        return "diagnostic"
    if _looks_like_diffstat(text):
        return "diffstat"
    if _grep_match_count(text) >= 2:
        return "grep"
    return "generic"


def _looks_like_pytest(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "short test summary info",
            "test session starts",
            "collected ",
            " passed",
            " failed",
            "\nFAILED ",
            "\nERROR ",
        )
    )


def _looks_like_diffstat(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    has_file_lines = any(_DIFFSTAT_FILE_RE.match(line) for line in lines)
    has_summary_line = any(_DIFFSTAT_SUMMARY_RE.search(line) for line in lines)
    return has_file_lines and has_summary_line


def _looks_like_diagnostic(text: str) -> bool:
    entries, summary_line = _parse_diagnostic_output(text)
    return bool(entries or summary_line)


def _grep_match_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if _parse_grep_line(line) is not None:
            count += 1
    return count


def _digest_pytest(text: str, *, max_items: int) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines()]
    summary_line = ""
    counts: dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "warnings": 0,
    }
    failures: list[str] = []

    for line in reversed(lines):
        if _PYTEST_COUNT_RE.search(line):
            summary_line = line.strip("= ")
            break

    for match in _PYTEST_COUNT_RE.finditer(summary_line):
        label = match.group("label")
        key = "errors" if label in {"error", "errors"} else label
        counts[key] += int(match.group("count"))

    for line in lines:
        failure_match = _PYTEST_FAILURE_RE.match(line.strip())
        if not failure_match:
            continue
        status = failure_match.group("status")
        name = failure_match.group("name")
        failures.append(f"{status} {name}")

    outcome = "failed" if counts["failed"] or counts["errors"] or failures else "passed"
    if not summary_line:
        if failures:
            summary_line = f"pytest {outcome}: {len(failures)} failing entries"
        else:
            summary_line = "pytest output"

    summary_parts = []
    for key in ("failed", "errors", "passed", "skipped", "xfailed", "xpassed", "warnings"):
        value = counts[key]
        if value:
            label = "error" if key == "errors" and value == 1 else key
            summary_parts.append(f"{value} {label}")
    summary = (
        f"pytest {outcome}: {', '.join(summary_parts)}"
        if summary_parts
        else f"pytest {outcome}"
    )

    limited_failures = failures[:max_items]
    highlights = []
    if summary_line:
        highlights.append(summary_line)
    highlights.extend(limited_failures)
    if not highlights:
        highlights = ["No pytest summary lines found."]

    return {
        "summary": summary,
        "highlights": highlights,
        "truncated": len(failures) > len(limited_failures),
        "details": {
            "outcome": outcome,
            "summary_line": summary_line,
            "counts": counts,
            "failing_tests": limited_failures,
            "failing_test_count": len(failures),
        },
    }


def _digest_traceback(text: str, *, max_items: int) -> dict[str, Any]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    frames: list[dict[str, Any]] = []
    exception_line = ""

    for index, line in enumerate(lines):
        match = _TRACEBACK_FRAME_RE.match(line)
        if not match:
            continue
        frame: dict[str, Any] = {
            "path": match.group("path"),
            "line": int(match.group("line")),
            "function": match.group("function"),
        }
        if index + 1 < len(lines):
            code_line = lines[index + 1].strip()
            if code_line and not _TRACEBACK_FRAME_RE.match(lines[index + 1]):
                frame["code"] = code_line
        frames.append(frame)

    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            exception_line = stripped
            break

    exception_type = exception_line
    exception_message = ""
    if ": " in exception_line:
        exception_type, exception_message = exception_line.split(": ", 1)

    limited_frames = frames[-max_items:]
    highlights = [exception_line or "Traceback detected"]
    for frame in limited_frames:
        highlights.append(
            f"{frame['path']}:{frame['line']} in {frame['function']}"
        )

    summary = exception_line or "traceback"
    return {
        "summary": summary,
        "highlights": highlights,
        "truncated": len(frames) > len(limited_frames),
        "details": {
            "exception_type": exception_type,
            "exception_message": exception_message,
            "frame_count": len(frames),
            "frames": limited_frames,
        },
    }


def _digest_grep(text: str, *, max_items: int) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    unique_paths: set[str] = set()

    for line in text.splitlines():
        parsed = _parse_grep_line(line)
        if parsed is None:
            continue
        matches.append(parsed)
        unique_paths.add(parsed["path"])

    limited_matches = matches[:max_items]
    highlights = [
        f"{match['path']}:{match['line']} {match['text']}".rstrip()
        for match in limited_matches
    ]
    summary = f"{len(matches)} matches across {len(unique_paths)} files"
    if not matches:
        summary = "No grep-style matches found"
        highlights = ["No grep-style matches found."]

    return {
        "summary": summary,
        "highlights": highlights,
        "truncated": len(matches) > len(limited_matches),
        "details": {
            "match_count": len(matches),
            "path_count": len(unique_paths),
            "matches": limited_matches,
        },
    }


def _digest_diffstat(text: str, *, max_items: int) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    summary_line = ""
    total_files = 0
    insertions = 0
    deletions = 0

    for line in text.splitlines():
        stripped = line.strip()
        file_match = _DIFFSTAT_FILE_RE.match(stripped)
        if file_match:
            files.append(
                {
                    "path": file_match.group("path"),
                    "changes": int(file_match.group("changes")),
                }
            )
            continue
        summary_match = _DIFFSTAT_SUMMARY_RE.search(stripped)
        if summary_match:
            summary_line = stripped
            total_files = int(summary_match.group("files"))
            insertions = int(summary_match.group("insertions") or 0)
            deletions = int(summary_match.group("deletions") or 0)

    ranked_files = sorted(files, key=lambda item: (-item["changes"], item["path"]))
    limited_files = ranked_files[:max_items]
    highlights = [summary_line] if summary_line else []
    highlights.extend(
        f"{item['path']} | {item['changes']} changed lines" for item in limited_files
    )
    if not highlights:
        highlights = ["No diffstat lines found."]

    if summary_line:
        summary = summary_line
    else:
        changed_lines = sum(item["changes"] for item in files)
        summary = f"{len(files)} files changed, {changed_lines} changed lines"

    return {
        "summary": summary,
        "highlights": highlights,
        "truncated": len(files) > len(limited_files),
        "details": {
            "file_count": total_files or len(files),
            "insertions": insertions,
            "deletions": deletions,
            "files": limited_files,
        },
    }


def _digest_diagnostic(text: str, *, max_items: int) -> dict[str, Any]:
    entries, summary_line = _parse_diagnostic_output(text)
    limited_entries = entries[:max_items]
    unique_paths = {entry["path"] for entry in entries}
    tool_names = sorted({str(entry["tool"]) for entry in entries if entry.get("tool")})

    error_count = sum(1 for entry in entries if entry["severity"] == "error")
    warning_count = sum(1 for entry in entries if entry["severity"] == "warning")
    note_count = sum(1 for entry in entries if entry["severity"] == "note")
    other_count = len(entries) - error_count - warning_count - note_count

    summary = _summarize_diagnostic_counts(
        error_count=error_count,
        warning_count=warning_count,
        note_count=note_count,
        other_count=other_count,
        path_count=len(unique_paths),
        fallback=summary_line,
    )

    highlights: list[str] = []
    if summary_line:
        highlights.append(summary_line)
    highlights.extend(_format_diagnostic_highlight(entry) for entry in limited_entries)
    if not highlights:
        highlights = ["No diagnostic entries found."]

    return {
        "summary": summary,
        "highlights": highlights,
        "truncated": len(entries) > len(limited_entries),
        "details": {
            "entry_count": len(entries),
            "error_count": error_count,
            "warning_count": warning_count,
            "note_count": note_count,
            "other_count": other_count,
            "path_count": len(unique_paths),
            "tools": tool_names,
            "summary_line": summary_line,
            "entries": limited_entries,
        },
    }


def _digest_generic(text: str, *, max_items: int) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    limited_lines = lines[:max_items]
    summary = "No output" if not lines else limited_lines[0]
    highlights = limited_lines or ["No output"]
    return {
        "summary": _trim_summary(summary),
        "highlights": highlights,
        "truncated": len(lines) > len(limited_lines),
        "details": {
            "excerpt": limited_lines,
        },
    }


def _parse_grep_line(line: str) -> dict[str, Any] | None:
    match = _GREP_LINE_RE.match(line.rstrip())
    if not match:
        return None
    return {
        "path": match.group("path"),
        "line": int(match.group("line")),
        "text": match.group("text").strip(),
    }


def _parse_diagnostic_output(text: str) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    summary_line = ""
    eslint_path: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            eslint_path = None
            continue

        tsc_match = _TSC_DIAGNOSTIC_RE.match(stripped)
        if tsc_match:
            entries.append(
                {
                    "path": tsc_match.group("path"),
                    "line": int(tsc_match.group("line")),
                    "column": int(tsc_match.group("column")),
                    "severity": tsc_match.group("severity"),
                    "code": tsc_match.group("code"),
                    "message": tsc_match.group("message").strip(),
                    "tool": "tsc",
                }
            )
            continue

        mypy_match = _MYPY_DIAGNOSTIC_RE.match(stripped)
        if mypy_match:
            entries.append(
                {
                    "path": mypy_match.group("path"),
                    "line": int(mypy_match.group("line")),
                    "column": None,
                    "severity": mypy_match.group("severity"),
                    "code": (mypy_match.group("code") or "").strip(),
                    "message": mypy_match.group("message").strip(),
                    "tool": "mypy",
                }
            )
            continue

        ruff_match = _RUFF_DIAGNOSTIC_RE.match(stripped)
        if ruff_match:
            entries.append(
                {
                    "path": ruff_match.group("path"),
                    "line": int(ruff_match.group("line")),
                    "column": int(ruff_match.group("column")),
                    "severity": "error",
                    "code": ruff_match.group("code"),
                    "message": ruff_match.group("message").strip(),
                    "tool": "ruff",
                }
            )
            continue

        if stripped.startswith(("/", "./", "../")) or re.match(r"^[A-Za-z]:[\\/]", stripped):
            if _looks_like_path_header(stripped):
                eslint_path = stripped
                continue

        eslint_match = _ESLINT_DIAGNOSTIC_RE.match(line)
        if eslint_match and eslint_path:
            entries.append(
                {
                    "path": eslint_path,
                    "line": int(eslint_match.group("line")),
                    "column": int(eslint_match.group("column")),
                    "severity": eslint_match.group("severity"),
                    "code": eslint_match.group("code"),
                    "message": eslint_match.group("message").strip(),
                    "tool": "eslint",
                }
            )
            continue

        if not summary_line and (
            _ESLINT_SUMMARY_RE.match(stripped)
            or _TSC_SUMMARY_RE.match(stripped)
            or _MYPY_SUMMARY_RE.match(stripped)
        ):
            summary_line = stripped

    return entries, summary_line


def _looks_like_path_header(line: str) -> bool:
    if line.startswith(("Found ", "Success:", "error", "warning", "note", "✖", "x ", "X ")):
        return False
    return bool(re.search(r"\.[A-Za-z0-9]+$", line))


def _format_diagnostic_highlight(entry: dict[str, Any]) -> str:
    location = f"{entry['path']}:{entry['line']}"
    column = entry.get("column")
    if isinstance(column, int):
        location = f"{location}:{column}"
    severity = str(entry.get("severity", "issue"))
    code = str(entry.get("code", "")).strip()
    message = str(entry.get("message", "")).strip()
    if code:
        return f"{location} {severity} [{code}] {message}".rstrip()
    return f"{location} {severity} {message}".rstrip()


def _summarize_diagnostic_counts(
    *,
    error_count: int,
    warning_count: int,
    note_count: int,
    other_count: int,
    path_count: int,
    fallback: str,
) -> str:
    parts: list[str] = []
    if error_count:
        parts.append(_format_count(error_count, "error"))
    if warning_count:
        parts.append(_format_count(warning_count, "warning"))
    if note_count:
        parts.append(_format_count(note_count, "note"))
    if other_count:
        parts.append(_format_count(other_count, "issue"))
    if parts:
        return f"{', '.join(parts)} across {_format_count(path_count, 'file')}"
    return fallback or "No diagnostic entries found"


def _format_count(value: int, singular: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {singular}{suffix}"


def _trim_summary(text: str, *, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
