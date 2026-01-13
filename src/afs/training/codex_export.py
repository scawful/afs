"""Export Codex CLI session logs into TrainingSample JSONL or AFS history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any

from ..generators.base import TrainingSample
from ..history import append_history_event
from .scoring import build_scoring_config, QualityScorer
from .redaction import redact_sample, redact_text


DEFAULT_CODEX_ROOTS = (
    Path.home() / ".codex",
    Path.home() / "src" / ".codex",
)


@dataclass
class CodexExportResult:
    """Summary of a Codex export run."""

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


@dataclass
class CodexHistoryImportResult:
    """Summary of a Codex history import run."""

    total_logs: int = 0
    total_messages: int = 0
    total_tools: int = 0
    imported_model: int = 0
    imported_tools: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"logs={self.total_logs} messages={self.total_messages} "
            f"tools={self.total_tools} model={self.imported_model} "
            f"tool_events={self.imported_tools} skipped={self.skipped} "
            f"errors={len(self.errors)}"
        )


def export_codex_logs_to_dataset(
    roots: Iterable[Path] | None,
    output_path: Path,
    *,
    scan_roots: Iterable[Path] | None = None,
    max_scan_depth: int = 4,
    default_domain: str = "codex",
    include_tools: bool = True,
    include_system: bool = False,
    max_tool_output_chars: int = 2000,
    limit: int | None = None,
    require_quality: bool = True,
    min_quality_score: float = 0.5,
    score_profile: str = "generic",
    enable_asar: bool = False,
    redact: bool = True,
) -> CodexExportResult:
    """Export Codex CLI sessions into TrainingSample JSONL."""
    result = CodexExportResult()
    samples: list[TrainingSample] = []

    root_list = _discover_roots(roots, scan_roots, max_scan_depth=max_scan_depth)
    log_files = _collect_log_files(root_list)
    result.total_logs = len(log_files)

    for log_path in log_files:
        for sample in _log_to_samples(
            log_path,
            default_domain=default_domain,
            include_tools=include_tools,
            include_system=include_system,
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


def import_codex_logs_to_history(
    roots: Iterable[Path] | None,
    history_root: Path,
    *,
    scan_roots: Iterable[Path] | None = None,
    max_scan_depth: int = 4,
    include_tools: bool = True,
    include_system: bool = False,
    max_tool_output_chars: int = 4000,
    limit: int | None = None,
    redact: bool = True,
) -> CodexHistoryImportResult:
    """Import Codex CLI sessions into AFS history logs."""
    result = CodexHistoryImportResult()

    root_list = _discover_roots(roots, scan_roots, max_scan_depth=max_scan_depth)
    log_files = _collect_log_files(root_list)
    result.total_logs = len(log_files)

    for log_path in log_files:
        session_meta: dict[str, Any] = {}
        system_text = ""
        current_model = ""
        current_cwd = ""
        last_user: str | None = None
        tool_calls: dict[str, dict[str, Any]] = {}

        for event in _iter_events(log_path):
            event_type = event.get("type")
            if event_type == "session_meta":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    session_meta = payload
                    if include_system:
                        system_text = _coerce_text(payload.get("instructions"))
                continue
            if event_type == "turn_context":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    model = payload.get("model")
                    cwd = payload.get("cwd")
                    if model:
                        current_model = str(model)
                    if cwd:
                        current_cwd = str(cwd)
                continue
            if event_type != "response_item":
                continue

            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            payload_type = payload.get("type")

            if payload_type == "custom_tool_call":
                call_id = _coerce_text(payload.get("call_id") or payload.get("callId"))
                if not call_id:
                    continue
                tool_calls[call_id] = {
                    "name": payload.get("name") or "tool",
                    "input": _parse_tool_input(payload.get("input")),
                }
                continue

            if payload_type == "custom_tool_call_output":
                result.total_tools += 1
                if not include_tools:
                    continue
                call_id = _coerce_text(payload.get("call_id") or payload.get("callId"))
                call = tool_calls.pop(call_id, {}) if call_id else {}
                tool_name = call.get("name") or "tool"
                tool_input = call.get("input")
                output_text, output_meta = _parse_tool_output(payload.get("output"))
                if not output_text:
                    result.skipped += 1
                    continue
                output_text = _truncate_text(output_text, max_tool_output_chars)
                tool_input = _redact_value(tool_input) if redact else tool_input
                output_text = _redact_text(output_text) if redact else output_text

                success = output_meta.get("exit_code")
                if isinstance(success, int):
                    success = success == 0
                elif isinstance(success, bool):
                    success = success
                else:
                    success = None

                metadata = _base_metadata(session_meta, log_path, current_model, current_cwd)
                metadata.update(
                    {
                        "tool": tool_name,
                        "tool_call_id": call_id,
                        "tool_success": success,
                        "tool_metadata": output_meta,
                    }
                )

                append_history_event(
                    history_root=history_root,
                    event_type="tool",
                    source="codex",
                    op=tool_name,
                    metadata=metadata,
                    payload={
                        "input": tool_input,
                        "output": output_text,
                    },
                    timestamp=_coerce_text(event.get("timestamp")),
                    redact_sensitive=redact,
                )
                result.imported_tools += 1
                continue

            if payload_type != "message":
                continue

            role = payload.get("role")
            content = payload.get("content")
            message_text = _extract_message_text(content)
            if not message_text:
                continue

            if role == "user":
                last_user = message_text
                continue
            if role != "assistant":
                continue

            result.total_messages += 1
            if not last_user:
                result.skipped += 1
                continue

            prompt = _redact_text(last_user) if redact else last_user
            response = _redact_text(message_text) if redact else message_text
            system = _redact_text(system_text) if redact else system_text

            payload_data = {
                "prompt": prompt,
                "response": response,
            }
            if include_system and system:
                payload_data["system"] = system

            metadata = _base_metadata(session_meta, log_path, current_model, current_cwd)
            append_history_event(
                history_root=history_root,
                event_type="model",
                source="codex",
                op="completion",
                metadata=metadata,
                payload=payload_data,
                timestamp=_coerce_text(event.get("timestamp")),
                redact_sensitive=redact,
            )
            result.imported_model += 1
            last_user = None

            if limit and result.imported_model >= limit:
                break
        if limit and result.imported_model >= limit:
            break

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
        resolved.extend(_scan_for_codex(scan_root, max_depth=max_scan_depth))

    deduped: list[Path] = []
    seen = set()
    for root in resolved:
        if not root.exists():
            continue
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _default_roots() -> list[Path]:
    roots: list[Path] = []
    for root in DEFAULT_CODEX_ROOTS:
        if root.exists():
            roots.append(root)
    return roots or [DEFAULT_CODEX_ROOTS[0]]


def _normalize_root(root: Path) -> Path:
    root = root.expanduser()
    if root.name == "sessions":
        return root.parent
    return root


def _scan_for_codex(scan_root: Path, *, max_depth: int) -> list[Path]:
    if not scan_root.exists() or not scan_root.is_dir():
        return []
    results: list[Path] = []
    base_depth = len(scan_root.parts)
    for path in scan_root.rglob(".codex"):
        if not path.is_dir():
            continue
        depth = len(path.parts) - base_depth
        if depth <= max_depth:
            results.append(path)
    return results


def _collect_log_files(roots: Iterable[Path]) -> list[Path]:
    log_files: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() == ".jsonl" and root.name.startswith("rollout-"):
                log_files.append(root)
            continue
        sessions_dir = root / "sessions"
        if root.name == "sessions":
            sessions_dir = root
        if not sessions_dir.exists():
            continue
        log_files.extend(sessions_dir.rglob("rollout-*.jsonl"))
    return sorted(log_files)


def _log_to_samples(
    path: Path,
    *,
    default_domain: str,
    include_tools: bool,
    include_system: bool,
    max_tool_output_chars: int,
) -> Iterable[TrainingSample]:
    session_meta: dict[str, Any] = {}
    system_text = ""
    current_model = ""
    current_cwd = ""
    last_user: str | None = None
    tool_calls: dict[str, dict[str, Any]] = {}
    tool_outputs: list[dict[str, Any]] = []

    for event in _iter_events(path):
        event_type = event.get("type")
        if event_type == "session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                session_meta = payload
                if include_system:
                    system_text = _coerce_text(payload.get("instructions"))
            continue
        if event_type == "turn_context":
            payload = event.get("payload")
            if isinstance(payload, dict):
                model = payload.get("model")
                cwd = payload.get("cwd")
                if model:
                    current_model = str(model)
                if cwd:
                    current_cwd = str(cwd)
            continue
        if event_type != "response_item":
            continue

        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")

        if payload_type == "custom_tool_call":
            call_id = _coerce_text(payload.get("call_id") or payload.get("callId"))
            if not call_id:
                continue
            tool_calls[call_id] = {
                "name": payload.get("name") or "tool",
                "input": _parse_tool_input(payload.get("input")),
            }
            continue

        if payload_type == "custom_tool_call_output":
            call_id = _coerce_text(payload.get("call_id") or payload.get("callId"))
            call = tool_calls.pop(call_id, {}) if call_id else {}
            tool_name = call.get("name") or "tool"
            tool_input = call.get("input")
            output_text, output_meta = _parse_tool_output(payload.get("output"))
            if output_text:
                tool_outputs.append(
                    {
                        "name": tool_name,
                        "input": tool_input,
                        "output": _truncate_text(output_text, max_tool_output_chars),
                        "metadata": output_meta,
                    }
                )
            continue

        if payload_type != "message":
            continue

        role = payload.get("role")
        content = payload.get("content")
        message_text = _extract_message_text(content)
        if not message_text:
            continue

        if role == "user":
            last_user = message_text
            tool_calls.clear()
            tool_outputs.clear()
            continue
        if role != "assistant":
            continue
        if not last_user:
            continue

        tool_context, tool_meta = _build_tool_context(
            tool_outputs,
            include_tools=include_tools,
            max_chars=max_tool_output_chars,
        )
        input_text = _build_input(system_text, tool_context, include_system=include_system)

        sample = TrainingSample(
            instruction=last_user,
            output=message_text,
            input=input_text,
            domain=default_domain,
            source="codex",
            teacher_model=current_model,
            timestamp=_coerce_text(event.get("timestamp")),
        )
        sample._metadata = {
            "codex_log_path": str(path),
            "codex_session_id": session_meta.get("id"),
            "codex_cli_version": session_meta.get("cli_version"),
            "codex_originator": session_meta.get("originator"),
            "codex_cwd": current_cwd or session_meta.get("cwd"),
            "codex_model": current_model,
            "codex_tool_calls": tool_meta,
            "codex_tool_names": [tool.get("name") for tool in tool_meta if tool.get("name")],
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


def _build_input(system_text: str, tool_context: str, *, include_system: bool) -> str:
    parts: list[str] = []
    if include_system and system_text:
        parts.append("System:\n" + system_text)
    if tool_context:
        parts.append(tool_context)
    return "\n\n".join(parts).strip()


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
                "metadata": tool.get("metadata"),
            }
        )

    if not context_lines:
        return "", meta_calls
    return "Tool outputs:\n" + "\n\n".join(context_lines), meta_calls


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        return str(text).strip() if text else ""
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                texts.append(str(text).strip())
        return "\n".join(texts).strip()
    return ""


def _parse_tool_input(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed
        except json.JSONDecodeError:
            return value
    return value


def _parse_tool_output(value: Any) -> tuple[str, dict[str, Any]]:
    if value is None:
        return "", {}
    if isinstance(value, dict):
        output = value.get("output") or value.get("result")
        metadata = value.get("metadata")
        output_text = _coerce_text(output) if output is not None else ""
        return output_text, metadata if isinstance(metadata, dict) else {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value, {}
        if isinstance(parsed, dict):
            output = parsed.get("output") or parsed.get("result")
            metadata = parsed.get("metadata")
            if output is None:
                return json.dumps(parsed, ensure_ascii=False), metadata if isinstance(metadata, dict) else {}
            return _coerce_text(output), metadata if isinstance(metadata, dict) else {}
        return _coerce_text(parsed), {}
    return _coerce_text(value), {}


def _truncate_text(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _redact_text(text: str) -> str:
    redacted, _ = redact_text(text)
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {key: _redact_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _base_metadata(
    session_meta: dict[str, Any],
    log_path: Path,
    current_model: str,
    current_cwd: str,
) -> dict[str, Any]:
    metadata = {
        "codex_log_path": str(log_path),
        "codex_session_id": session_meta.get("id"),
        "codex_cli_version": session_meta.get("cli_version"),
        "codex_originator": session_meta.get("originator"),
        "codex_cwd": current_cwd or session_meta.get("cwd"),
        "codex_model": current_model,
    }
    return metadata
