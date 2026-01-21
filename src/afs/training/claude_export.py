"""Export Claude Code logs into TrainingSample JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..generators.base import TrainingSample
from .redaction import redact_sample
from .scoring import QualityScorer, build_scoring_config

DEFAULT_CLAUDE_ROOTS = (
    Path.home() / ".claude",
    Path.home() / "src" / ".claude",
)


@dataclass
class ClaudeExportResult:
    """Summary of a Claude export run."""

    total_logs: int = 0
    total_messages: int = 0
    exported: int = 0
    skipped: int = 0
    filtered: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"logs={self.total_logs} messages={self.total_messages} "
            f"exported={self.exported} skipped={self.skipped} "
            f"filtered={self.filtered} errors={len(self.errors)}"
        )


def export_claude_logs_to_dataset(
    roots: Iterable[Path] | None,
    output_path: Path,
    *,
    scan_roots: Iterable[Path] | None = None,
    max_scan_depth: int = 4,
    default_domain: str = "claude",
    include_tools: bool = True,
    max_tool_output_chars: int = 2000,
    limit: int | None = None,
    require_quality: bool = True,
    min_quality_score: float = 0.5,
    score_profile: str = "generic",
    enable_asar: bool = False,
    redact: bool = True,
) -> ClaudeExportResult:
    """Export Claude Code logs into TrainingSample JSONL."""
    result = ClaudeExportResult()
    samples: list[TrainingSample] = []

    root_list = _discover_roots(roots, scan_roots, max_scan_depth=max_scan_depth)
    log_files = _collect_project_logs(root_list)
    result.total_logs = len(log_files)

    for log_path in log_files:
        for sample in _log_to_samples(
            log_path,
            default_domain=default_domain,
            include_tools=include_tools,
            max_tool_output_chars=max_tool_output_chars,
        ):
            result.total_messages += 1
            if redact:
                redact_sample(sample)
            samples.append(sample)
            result.exported += 1
            if limit and result.exported >= limit:
                break
        if limit and result.exported >= limit:
            break

    if require_quality and samples:
        scorer = QualityScorer(
            config=build_scoring_config(
                score_profile,
                enable_asar=enable_asar,
            )
        )
        scored = scorer.score_batch(samples, update_samples=True)
        filtered_samples: list[TrainingSample] = []
        for sample, score in zip(samples, scored, strict=False):
            if score.overall >= min_quality_score:
                filtered_samples.append(sample)
            else:
                result.filtered += 1
        samples = filtered_samples

    result.exported = len(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    return result


def _discover_roots(
    roots: Iterable[Path] | None,
    scan_roots: Iterable[Path] | None,
    *,
    max_scan_depth: int,
) -> list[Path]:
    resolved: list[Path] = []
    for root in roots or _default_roots():
        resolved.append(_normalize_root(Path(root)))

    if roots is None and scan_roots is None:
        scan_roots = [Path.home() / "src"]

    for scan_root in scan_roots or []:
        scan_root = Path(scan_root).expanduser()
        resolved.extend(_scan_for_claude(scan_root, max_depth=max_scan_depth))

    deduped: list[Path] = []
    seen = set()
    for root in resolved:
        if not root.exists() or not root.is_dir():
            continue
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _default_roots() -> list[Path]:
    roots: list[Path] = []
    for root in DEFAULT_CLAUDE_ROOTS:
        if root.exists():
            roots.append(root)
    return roots or [DEFAULT_CLAUDE_ROOTS[0]]


def _normalize_root(root: Path) -> Path:
    root = root.expanduser()
    if root.name == "projects":
        return root.parent
    return root


def _scan_for_claude(scan_root: Path, *, max_depth: int) -> list[Path]:
    if not scan_root.exists() or not scan_root.is_dir():
        return []
    results: list[Path] = []
    base_depth = len(scan_root.parts)
    for path in scan_root.rglob(".claude"):
        if not path.is_dir():
            continue
        depth = len(path.parts) - base_depth
        if depth <= max_depth:
            results.append(path)
    return results


def _collect_project_logs(roots: Iterable[Path]) -> list[Path]:
    log_files: list[Path] = []
    for root in roots:
        projects_dir = root / "projects"
        if not projects_dir.exists():
            continue
        log_files.extend(projects_dir.rglob("*.jsonl"))
    return sorted(log_files)


def _log_to_samples(
    path: Path,
    *,
    default_domain: str,
    include_tools: bool,
    max_tool_output_chars: int,
) -> Iterable[TrainingSample]:
    last_user: str | None = None
    tool_calls: dict[str, dict[str, Any]] = {}
    tool_outputs: list[dict[str, Any]] = []

    for event in _iter_events(path):
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            tool_results = _extract_tool_results(content)
            if tool_results:
                for result in tool_results:
                    tool_use_id = result.get("tool_use_id")
                    call = tool_calls.pop(tool_use_id, {}) if tool_use_id else {}
                    output = _coerce_text(result.get("content"))
                    if not output:
                        continue
                    tool_outputs.append(
                        {
                            "name": call.get("name", "tool"),
                            "input": call.get("input"),
                            "output": output,
                            "is_error": bool(result.get("is_error", False)),
                        }
                    )
                continue
            user_text = _extract_text(content)
            if user_text:
                last_user = user_text
                tool_calls.clear()
                tool_outputs.clear()
            continue

        if role != "assistant":
            continue

        tool_uses = _extract_tool_uses(content)
        for tool in tool_uses:
            tool_id = str(tool.get("id") or "")
            if tool_id:
                tool_calls[tool_id] = {
                    "name": tool.get("name") or "tool",
                    "input": tool.get("input"),
                }

        response_text = _extract_text(content)
        if not response_text or not last_user:
            continue

        tool_context, tool_meta = _build_tool_context(
            tool_outputs,
            include_tools=include_tools,
            max_chars=max_tool_output_chars,
        )
        sample = TrainingSample(
            instruction=last_user,
            output=response_text,
            input=tool_context,
            domain=default_domain,
            source="claude",
            teacher_model=str(message.get("model") or ""),
            timestamp=str(event.get("timestamp") or ""),
        )
        sample._metadata = {
            "claude_log_path": str(path),
            "claude_session_id": event.get("sessionId"),
            "claude_agent_id": event.get("agentId"),
            "claude_slug": event.get("slug"),
            "claude_version": event.get("version"),
            "claude_git_branch": event.get("gitBranch"),
            "claude_cwd": event.get("cwd"),
            "claude_request_id": event.get("requestId"),
            "claude_tool_calls": tool_meta,
        }
        tool_outputs.clear()
        yield sample


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = _coerce_text(item.get("text"))
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


def _extract_tool_results(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_result"
    ]


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _build_tool_context(
    tool_outputs: list[dict[str, Any]],
    *,
    include_tools: bool,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    if not include_tools or not tool_outputs:
        return "", []

    context_lines: list[str] = []
    meta_calls: list[dict[str, Any]] = []
    for tool in tool_outputs:
        name = str(tool.get("name") or "tool")
        output = _truncate_text(str(tool.get("output") or ""), max_chars)
        if output:
            context_lines.append(f"{name} output:\n{output}")
        meta_calls.append(
            {
                "name": name,
                "input": tool.get("input"),
                "is_error": tool.get("is_error"),
            }
        )

    if not context_lines:
        return "", meta_calls
    return "Tool outputs:\n" + "\n\n".join(context_lines), meta_calls


def _truncate_text(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"
