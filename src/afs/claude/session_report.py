"""Generate corrected, AFS-friendly reports from Claude session logs.

Primary goals:
- Locate transcripts + per-session artifacts (subagents, tool-results, debug log)
- Produce a stable report that can be written into an AFS project context
- Prefer verifiable facts (paths + counts) over brittle "file:line" references
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ClaudeSessionPaths:
    """Resolved filesystem locations for a Claude session."""

    session_id: str
    transcript_path: Path | None
    artifacts_dir: Path | None
    debug_log_path: Path | None
    project_slug: str | None
    other_transcripts: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class SubagentSummary:
    """High-level view of a single subagent sidechain."""

    path: Path
    agent_id: str | None
    model: str | None
    cwd: str | None
    git_branch: str | None
    slug: str | None
    first_user_prompt: str | None
    last_assistant_text: str | None
    tool_calls: Counter[str] = field(default_factory=Counter)
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def is_compact_summary(self) -> bool:
        return bool(self.agent_id and self.agent_id.startswith("acompact-"))


@dataclass
class ClaudeSessionReport:
    paths: ClaudeSessionPaths
    cwd: str | None
    git_branch: str | None
    version: str | None
    slug: str | None
    start_timestamp: str | None
    end_timestamp: str | None
    models: tuple[str, ...]
    tool_calls: Counter[str]
    subagents: tuple[SubagentSummary, ...]


def discover_session_paths(
    session_id_or_prefix: str,
    *,
    claude_root: Path | None = None,
) -> ClaudeSessionPaths:
    """Locate transcript/debug/artifact paths for a session.

    `session_id_or_prefix` can be a full UUID or a short prefix (e.g. first 8).
    """
    claude_root = (claude_root or (Path.home() / ".claude")).expanduser().resolve()
    projects_root = claude_root / "projects"

    warnings: list[str] = []
    if not projects_root.exists():
        warnings.append(f"projects root not found: {projects_root}")
        return ClaudeSessionPaths(
            session_id=session_id_or_prefix,
            transcript_path=None,
            artifacts_dir=None,
            debug_log_path=claude_root / "debug" / f"{session_id_or_prefix}.txt",
            project_slug=None,
            warnings=tuple(warnings),
        )

    transcripts = _find_transcripts(projects_root, session_id_or_prefix)
    transcript_path: Path | None = None
    other: list[Path] = []

    if not transcripts:
        warnings.append(f"no transcript found for: {session_id_or_prefix}")
    elif len(transcripts) == 1:
        transcript_path = transcripts[0]
    else:
        # Prefer the most recently modified transcript if multiple exist.
        transcripts = sorted(transcripts, key=lambda p: p.stat().st_mtime, reverse=True)
        transcript_path = transcripts[0]
        other = list(transcripts[1:])
        warnings.append(
            f"multiple transcripts matched {session_id_or_prefix}; using newest"
        )

    session_id = transcript_path.stem if transcript_path else session_id_or_prefix
    project_slug = transcript_path.parent.name if transcript_path else None

    artifacts_dir = transcript_path.with_suffix("") if transcript_path else None
    if artifacts_dir and not artifacts_dir.is_dir():
        warnings.append(f"artifacts dir missing: {artifacts_dir}")
        artifacts_dir = None

    debug_log = claude_root / "debug" / f"{session_id}.txt"
    if not debug_log.exists():
        warnings.append(f"debug log missing: {debug_log}")

    return ClaudeSessionPaths(
        session_id=session_id,
        transcript_path=transcript_path,
        artifacts_dir=artifacts_dir,
        debug_log_path=debug_log,
        project_slug=project_slug,
        other_transcripts=tuple(other),
        warnings=tuple(warnings),
    )


def build_session_report(
    session_id_or_prefix: str,
    *,
    claude_root: Path | None = None,
    include_subagents: bool = True,
    max_subagent_chars: int = 4000,
) -> ClaudeSessionReport:
    """Build a structured report for a Claude session."""
    paths = discover_session_paths(session_id_or_prefix, claude_root=claude_root)

    transcript_meta: dict[str, Any] = {}
    if paths.transcript_path:
        transcript_meta = _extract_session_metadata(paths.transcript_path)

    subagents: list[SubagentSummary] = []
    if include_subagents and paths.artifacts_dir:
        subagent_dir = paths.artifacts_dir / "subagents"
        if subagent_dir.exists():
            for sub_path in sorted(subagent_dir.glob("*.jsonl")):
                subagents.append(
                    _summarize_subagent_log(sub_path, max_chars=max_subagent_chars)
                )

    models: set[str] = set()
    models.update(transcript_meta.get("models") or set())
    for sub in subagents:
        if sub.model:
            models.add(sub.model)

    tool_calls = Counter()
    tool_calls.update(transcript_meta.get("tool_calls") or Counter())
    for sub in subagents:
        tool_calls.update(sub.tool_calls)

    cwd = transcript_meta.get("cwd")
    git_branch = transcript_meta.get("git_branch")
    version = transcript_meta.get("version")
    slug = transcript_meta.get("slug")
    start_ts = transcript_meta.get("start_timestamp")
    end_ts = transcript_meta.get("end_timestamp")

    return ClaudeSessionReport(
        paths=paths,
        cwd=cwd if isinstance(cwd, str) else None,
        git_branch=git_branch if isinstance(git_branch, str) else None,
        version=version if isinstance(version, str) else None,
        slug=slug if isinstance(slug, str) else None,
        start_timestamp=start_ts if isinstance(start_ts, str) else None,
        end_timestamp=end_ts if isinstance(end_ts, str) else None,
        models=tuple(sorted(m for m in models if m)),
        tool_calls=tool_calls,
        subagents=tuple(subagents),
    )


def render_session_report_markdown(report: ClaudeSessionReport) -> str:
    """Render a human-friendly markdown report."""
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []

    lines.append(f"# Claude Session {report.paths.session_id}")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")

    lines.append("## Primary Artifacts")
    if report.paths.transcript_path:
        lines.append(f"- Transcript: `{report.paths.transcript_path}`")
    else:
        lines.append("- Transcript: (missing)")

    debug_log = report.paths.debug_log_path
    if debug_log and debug_log.exists():
        lines.append(f"- Debug log: `{debug_log}`")
    else:
        lines.append(f"- Debug log: `{debug_log}` (missing)")

    artifacts = report.paths.artifacts_dir
    if artifacts:
        lines.append(f"- Session artifacts dir: `{artifacts}`")
        subagents_dir = artifacts / "subagents"
        tool_results_dir = artifacts / "tool-results"
        sub_count = (
            sum(1 for _ in subagents_dir.glob("*.jsonl")) if subagents_dir.exists() else 0
        )
        tool_count = (
            sum(1 for _ in tool_results_dir.glob("*.txt"))
            if tool_results_dir.exists()
            else 0
        )
        lines.append(f"  - Subagents: {sub_count} file(s) in `{subagents_dir}`")
        lines.append(f"  - Tool results: {tool_count} file(s) in `{tool_results_dir}`")
    else:
        lines.append("- Session artifacts dir: (missing)")
    lines.append("")

    lines.append("## Session Metadata")
    if report.cwd:
        lines.append(f"- CWD: `{report.cwd}`")
    if report.git_branch:
        lines.append(f"- Git branch: `{report.git_branch}`")
    if report.version:
        lines.append(f"- Claude version: `{report.version}`")
    if report.slug:
        lines.append(f"- Slug: `{report.slug}`")
    if report.start_timestamp or report.end_timestamp:
        lines.append(
            f"- Transcript time range: `{report.start_timestamp or '?'} .. {report.end_timestamp or '?'}`"
        )
    if report.models:
        lines.append(f"- Models: {', '.join(f'`{m}`' for m in report.models)}")
    if report.tool_calls:
        top = ", ".join(
            f"{name} x{count}" for name, count in report.tool_calls.most_common(8)
        )
        lines.append(f"- Tools (top): {top}")
    lines.append("")

    non_compact = [s for s in report.subagents if not s.is_compact_summary]
    compact = [s for s in report.subagents if s.is_compact_summary]

    lines.append(f"## Subagents ({len(report.subagents)})")
    if not report.subagents:
        lines.append("- (none found)")
    else:
        if non_compact:
            lines.append("")
            lines.append("### Work Subagents")
            for sub in non_compact:
                lines.extend(_render_subagent_bullets(sub))
        if compact:
            lines.append("")
            lines.append("### Compact Summary Subagents")
            for sub in compact:
                lines.extend(_render_subagent_bullets(sub))

    if report.paths.other_transcripts:
        lines.append("")
        lines.append("## Other Matching Transcripts")
        for path in report.paths.other_transcripts:
            lines.append(f"- `{path}`")

    if report.paths.warnings:
        lines.append("")
        lines.append("## Warnings / Corrections")
        for warning in report.paths.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).rstrip() + "\n"


def _render_subagent_bullets(sub: SubagentSummary) -> list[str]:
    task = _one_line_task(sub.first_user_prompt)
    model = f" ({sub.model})" if sub.model else ""
    agent = sub.agent_id or sub.path.stem
    entry = f"- `{agent}`{model}: {task} (`{sub.path.name}`)"
    lines: list[str] = [entry]
    if sub.tool_calls:
        tools = ", ".join(
            f"{name} x{count}" for name, count in sub.tool_calls.most_common(6)
        )
        lines.append(f"  - Tools: {tools}")
    if sub.warnings:
        for warning in sub.warnings:
            lines.append(f"  - Warning: {warning}")
    return lines


def _one_line_task(prompt: str | None, *, max_chars: int = 140) -> str:
    if not prompt:
        return "(no prompt)"
    line = prompt.strip().splitlines()[0].strip()
    if not line:
        return "(no prompt)"
    lowered = line.lower()
    for prefix in ("i need to ", "please ", "can you "):
        if lowered.startswith(prefix):
            line = line[len(prefix) :].lstrip()
            break
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 3].rstrip() + "..."


def _find_transcripts(projects_root: Path, session_id_or_prefix: str) -> list[Path]:
    full = "-" in session_id_or_prefix and len(session_id_or_prefix) >= 32
    pattern = (
        f"{session_id_or_prefix}.jsonl"
        if full
        else f"{session_id_or_prefix}*.jsonl"
    )
    matches: list[Path] = []
    for project_dir in sorted(projects_root.iterdir()):
        if not project_dir.is_dir():
            continue
        for path in project_dir.glob(pattern):
            if path.is_file():
                matches.append(path)
    return matches


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text = item["text"].strip()
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()
    return ""


def _extract_tool_uses(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_use"
    ]


def _extract_session_metadata(transcript_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "cwd": None,
        "git_branch": None,
        "version": None,
        "slug": None,
        "start_timestamp": None,
        "end_timestamp": None,
        "models": set(),
        "tool_calls": Counter(),
    }

    for event in _iter_events(transcript_path):
        if meta["cwd"] is None:
            cwd = event.get("cwd")
            if isinstance(cwd, str) and cwd.strip():
                meta["cwd"] = cwd.strip()
        if meta["git_branch"] is None:
            branch = event.get("gitBranch")
            if isinstance(branch, str) and branch.strip():
                meta["git_branch"] = branch.strip()
        if meta["version"] is None:
            version = event.get("version")
            if isinstance(version, str) and version.strip():
                meta["version"] = version.strip()
        if meta["slug"] is None:
            slug = event.get("slug")
            if isinstance(slug, str) and slug.strip():
                meta["slug"] = slug.strip()

        ts = event.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            if meta["start_timestamp"] is None:
                meta["start_timestamp"] = ts
            meta["end_timestamp"] = ts

        message = event.get("message")
        if not isinstance(message, dict):
            continue
        model = message.get("model")
        if isinstance(model, str) and model.strip():
            meta["models"].add(model.strip())

        if message.get("role") != "assistant":
            continue
        for tool in _extract_tool_uses(message.get("content")):
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                meta["tool_calls"][name.strip()] += 1

    return meta


def _summarize_subagent_log(path: Path, *, max_chars: int) -> SubagentSummary:
    agent_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    slug: str | None = None
    model: str | None = None
    first_user: str | None = None
    last_assistant: str | None = None
    tool_calls: Counter[str] = Counter()
    start_ts: str | None = None
    end_ts: str | None = None
    warnings: list[str] = []

    for event in _iter_events(path):
        if agent_id is None:
            raw_agent = event.get("agentId")
            if isinstance(raw_agent, str) and raw_agent.strip():
                agent_id = raw_agent.strip()
        if cwd is None:
            raw_cwd = event.get("cwd")
            if isinstance(raw_cwd, str) and raw_cwd.strip():
                cwd = raw_cwd.strip()
        if git_branch is None:
            raw_branch = event.get("gitBranch")
            if isinstance(raw_branch, str) and raw_branch.strip():
                git_branch = raw_branch.strip()
        if slug is None:
            raw_slug = event.get("slug")
            if isinstance(raw_slug, str) and raw_slug.strip():
                slug = raw_slug.strip()

        ts = event.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            if start_ts is None:
                start_ts = ts
            end_ts = ts

        message = event.get("message")
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")
        if role == "user" and first_user is None:
            text = _extract_text(content)
            if text:
                first_user = text
            continue

        if role != "assistant":
            continue

        if model is None:
            raw_model = message.get("model")
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model.strip()

        for tool in _extract_tool_uses(content):
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                tool_calls[name.strip()] += 1

        text = _extract_text(content)
        if text:
            last_assistant = text

    if agent_id and not path.stem.endswith(agent_id):
        warnings.append(f"agent_id mismatch vs filename: {agent_id} vs {path.stem}")
    if last_assistant and len(last_assistant) > max_chars:
        last_assistant = last_assistant[: max_chars - 15].rstrip() + "\n...[truncated]"

    return SubagentSummary(
        path=path,
        agent_id=agent_id,
        model=model,
        cwd=cwd,
        git_branch=git_branch,
        slug=slug,
        first_user_prompt=first_user,
        last_assistant_text=last_assistant,
        tool_calls=tool_calls,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        warnings=tuple(warnings),
    )

