"""Export history events into TrainingSample JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any

from ..generators.base import TrainingSample
from ..history import iter_history_events
from .scoring import build_scoring_config, QualityScorer
from .redaction import redact_sample


@dataclass
class HistoryExportResult:
    """Summary of a history export run."""

    total_events: int = 0
    exported: int = 0
    skipped: int = 0
    filtered: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"total={self.total_events} exported={self.exported} "
            f"skipped={self.skipped} filtered={self.filtered} "
            f"errors={len(self.errors)}"
        )


def export_history_to_dataset(
    history_root: Path,
    output_path: Path,
    *,
    event_types: Iterable[str] | None = None,
    include_tools: bool = False,
    include_fs: bool = False,
    include_cli: bool = False,
    default_domain: str = "history",
    tool_domain: str = "history_tools",
    limit: int | None = None,
    require_quality: bool = True,
    min_quality_score: float = 0.5,
    score_profile: str = "generic",
    enable_asar: bool = False,
    redact: bool = True,
) -> HistoryExportResult:
    """Export history events into TrainingSample JSONL."""
    result = HistoryExportResult()
    samples: list[TrainingSample] = []

    event_set = {str(item) for item in event_types} if event_types else None

    for event in iter_history_events(history_root, event_types=event_set, include_payloads=True):
        result.total_events += 1
        sample = _event_to_sample(
            event,
            include_tools=include_tools,
            include_fs=include_fs,
            include_cli=include_cli,
            default_domain=default_domain,
            tool_domain=tool_domain,
        )
        if sample is None:
            result.skipped += 1
            continue
        if redact:
            redact_sample(sample)
        samples.append(sample)
        result.exported += 1
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


def _event_to_sample(
    event: dict[str, Any],
    *,
    include_tools: bool,
    include_fs: bool,
    include_cli: bool,
    default_domain: str,
    tool_domain: str,
) -> TrainingSample | None:
    event_type = str(event.get("type", "")).lower()
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}

    if event_type == "model":
        prompt, system = _extract_prompt(payload)
        response = payload.get("response")
        if not prompt or not response:
            return None
        domain = metadata.get("domain") if isinstance(metadata.get("domain"), str) else default_domain
        sample = TrainingSample(
            instruction=str(prompt).strip(),
            output=str(response).strip(),
            input=str(system).strip(),
            domain=domain,
            source="history",
        )
    elif event_type == "tool" and include_tools:
        if metadata.get("success") is False:
            return None
        tool_name = metadata.get("tool") or payload.get("tool") or "tool"
        input_data = payload.get("input")
        output = payload.get("output")
        if not output:
            return None
        instruction = f"Tool {tool_name} input:\\n{_format_payload(input_data)}"
        sample = TrainingSample(
            instruction=instruction,
            output=str(output).strip(),
            input="",
            domain=tool_domain,
            source="history_tool",
        )
    elif event_type == "fs" and include_fs:
        output = payload.get("content")
        if not output:
            return None
        instruction = f"FS {metadata.get('op', 'read')} {metadata.get('relative_path', '')}".strip()
        sample = TrainingSample(
            instruction=instruction,
            output=str(output).strip(),
            input="",
            domain=tool_domain,
            source="history_fs",
        )
    elif event_type == "cli" and include_cli:
        argv = metadata.get("argv")
        if not argv:
            return None
        instruction = "CLI invocation"
        sample = TrainingSample(
            instruction=instruction,
            output=_format_payload(argv),
            input="",
            domain=tool_domain,
            source="history_cli",
        )
    else:
        return None

    sample._metadata = {
        "history_event_id": event.get("id"),
        "history_type": event_type,
        "history_source": event.get("source"),
        "history_op": event.get("op"),
        "history_metadata": metadata,
        "history_payload_ref": event.get("payload_ref"),
    }
    return sample


def _extract_prompt(payload: dict[str, Any]) -> tuple[str | None, str]:
    prompt = payload.get("prompt")
    system = payload.get("system") or ""
    messages = payload.get("messages")
    if prompt:
        return str(prompt), str(system)
    if isinstance(messages, list):
        system_messages = []
        last_user = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system" and content:
                system_messages.append(content)
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        system_text = "\n".join(system_messages).strip()
        return (str(last_user).strip() if last_user else None, system_text)
    return None, str(system)


def _format_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)
