"""Export Gemini CLI logs into TrainingSample JSONL."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any

from ..generators.base import TrainingSample
from .scoring import build_scoring_config, QualityScorer
from .redaction import redact_sample


DEFAULT_GEMINI_ROOTS = (
    Path.home() / ".gemini",
    Path.home() / "src" / ".gemini",
)


@dataclass
class GeminiExportResult:
    """Summary of a Gemini export run."""

    total_sessions: int = 0
    total_messages: int = 0
    exported: int = 0
    skipped: int = 0
    filtered: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"sessions={self.total_sessions} messages={self.total_messages} "
            f"exported={self.exported} skipped={self.skipped} "
            f"filtered={self.filtered} errors={len(self.errors)}"
        )


def export_gemini_logs_to_dataset(
    roots: Iterable[Path] | None,
    output_path: Path,
    *,
    scan_roots: Iterable[Path] | None = None,
    max_scan_depth: int = 4,
    default_domain: str = "gemini",
    include_tools: bool = True,
    include_thoughts: bool = False,
    max_tool_output_chars: int = 2000,
    limit: int | None = None,
    require_quality: bool = True,
    min_quality_score: float = 0.5,
    score_profile: str = "generic",
    enable_asar: bool = False,
    redact: bool = True,
) -> GeminiExportResult:
    """Export Gemini CLI sessions into TrainingSample JSONL."""
    result = GeminiExportResult()
    samples: list[TrainingSample] = []

    root_list = _discover_roots(roots, scan_roots, max_scan_depth=max_scan_depth)
    session_files = _collect_session_files(root_list)
    result.total_sessions = len(session_files)

    for session_path in session_files:
        session = _load_session(session_path)
        if not session:
            result.skipped += 1
            continue
        session_samples = _session_to_samples(
            session,
            session_path=session_path,
            default_domain=default_domain,
            include_tools=include_tools,
            include_thoughts=include_thoughts,
            max_tool_output_chars=max_tool_output_chars,
        )
        result.total_messages += len(session_samples)
        for sample in session_samples:
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
        for sample, score in zip(samples, scored):
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
        resolved.extend(_scan_for_gemini(scan_root, max_depth=max_scan_depth))

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
    for root in DEFAULT_GEMINI_ROOTS:
        if root.exists():
            roots.append(root)
    return roots or [DEFAULT_GEMINI_ROOTS[0]]


def _normalize_root(root: Path) -> Path:
    root = root.expanduser()
    if root.name == "tmp":
        return root.parent
    return root


def _scan_for_gemini(scan_root: Path, *, max_depth: int) -> list[Path]:
    if not scan_root.exists() or not scan_root.is_dir():
        return []
    results: list[Path] = []
    base_depth = len(scan_root.parts)
    for path in scan_root.rglob(".gemini"):
        if not path.is_dir():
            continue
        depth = len(path.parts) - base_depth
        if depth <= max_depth:
            results.append(path)
    return results


def _collect_session_files(roots: Iterable[Path]) -> list[Path]:
    session_files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("session-*.json"):
            if "tmp" not in path.parts:
                continue
            session_files.append(path)
    return sorted(session_files)


def _load_session(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _session_to_samples(
    session: dict[str, Any],
    *,
    session_path: Path,
    default_domain: str,
    include_tools: bool,
    include_thoughts: bool,
    max_tool_output_chars: int,
) -> list[TrainingSample]:
    messages = session.get("messages")
    if not isinstance(messages, list):
        return []

    session_id = session.get("sessionId") or session_path.stem
    project_hash = session.get("projectHash") or ""
    system_messages: list[str] = []
    last_user: str | None = None
    samples: list[TrainingSample] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_type = str(msg.get("type", "")).lower()
        content = msg.get("content")
        if msg_type == "system":
            if isinstance(content, str) and content.strip():
                system_messages.append(content.strip())
            continue
        if msg_type == "user":
            if isinstance(content, str) and content.strip():
                last_user = content.strip()
            continue
        if msg_type in {"gemini", "assistant", "model"}:
            if not last_user or not isinstance(content, str) or not content.strip():
                continue
            tool_context, tool_meta = _build_tool_context(
                msg.get("toolCalls"),
                include_tools=include_tools,
                max_chars=max_tool_output_chars,
            )
            input_text = _build_input(system_messages, tool_context)
            thought_text = (
                _format_thoughts(msg.get("thoughts"))
                if include_thoughts
                else None
            )

            sample = TrainingSample(
                instruction=last_user,
                output=content.strip(),
                input=input_text,
                thinking=thought_text,
                domain=default_domain,
                source="gemini",
                teacher_model=str(msg.get("model") or ""),
                timestamp=str(msg.get("timestamp") or ""),
            )
            sample._metadata = {
                "gemini_session_id": session_id,
                "gemini_project_hash": project_hash,
                "gemini_session_path": str(session_path),
                "gemini_message_id": msg.get("id"),
                "gemini_tokens": msg.get("tokens"),
                "gemini_tool_calls": tool_meta,
                "gemini_tool_names": [tool.get("name") for tool in tool_meta if tool.get("name")],
            }
            samples.append(sample)

    return samples


def _build_input(system_messages: list[str], tool_context: str) -> str:
    parts: list[str] = []
    if system_messages:
        parts.append("System:\n" + "\n".join(system_messages))
    if tool_context:
        parts.append(tool_context)
    return "\n\n".join(parts).strip()


def _build_tool_context(
    tool_calls: Any,
    *,
    include_tools: bool,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    if not include_tools or not isinstance(tool_calls, list):
        return "", []

    context_lines: list[str] = []
    meta_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "tool")
        status = call.get("status")
        args = call.get("args")
        output = _extract_tool_output(call.get("result"))
        output = _truncate_text(output, max_chars)
        if output:
            context_lines.append(f"{name} output:\n{output}")
        meta_calls.append(
            {
                "name": name,
                "status": status,
                "args": args,
            }
        )

    if not context_lines:
        return "", meta_calls
    return "Tool outputs:\n" + "\n\n".join(context_lines), meta_calls


def _extract_tool_output(result: Any) -> str:
    outputs: list[str] = []
    if isinstance(result, list):
        for item in result:
            outputs.append(_extract_tool_output(item))
    elif isinstance(result, dict):
        if "functionResponse" in result:
            outputs.append(_extract_tool_output(result.get("functionResponse")))
        if "response" in result:
            outputs.append(_extract_tool_output(result.get("response")))
        if "output" in result and result.get("output") is not None:
            outputs.append(str(result.get("output")))
        if "content" in result and result.get("content") is not None:
            outputs.append(str(result.get("content")))
    elif isinstance(result, str):
        outputs.append(result)
    return "\n".join([item for item in outputs if item])


def _truncate_text(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


def _format_thoughts(thoughts: Any) -> str | None:
    if not isinstance(thoughts, list):
        return None
    lines: list[str] = []
    for item in thoughts:
        if isinstance(item, dict):
            subject = str(item.get("subject") or "").strip()
            description = str(item.get("description") or "").strip()
            if subject and description:
                lines.append(f"{subject}: {description}")
            elif description:
                lines.append(description)
        elif isinstance(item, str) and item.strip():
            lines.append(item.strip())
    if not lines:
        return None
    return "\n".join(lines)
