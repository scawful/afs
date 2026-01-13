"""Export Antigravity trajectory summaries into TrainingSample JSONL."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any
from urllib.parse import urlparse

from ..generators.base import TrainingSample
from .scoring import build_scoring_config, QualityScorer
from .redaction import redact_sample


DEFAULT_ANTIGRAVITY_DB = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Antigravity"
    / "User"
    / "globalStorage"
    / "state.vscdb"
)
DEFAULT_STATE_KEYS = (
    "antigravityUnifiedStateSync.trajectorySummaries",
    "unifiedStateSync.trajectorySummaries",
)

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class AntigravityExportResult:
    """Summary of an Antigravity export run."""

    total_payloads: int = 0
    exported: int = 0
    skipped: int = 0
    filtered: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"payloads={self.total_payloads} exported={self.exported} "
            f"skipped={self.skipped} filtered={self.filtered} "
            f"errors={len(self.errors)}"
        )


def export_antigravity_to_dataset(
    db_path: Path,
    output_path: Path,
    *,
    state_keys: Iterable[str] | None = None,
    default_domain: str = "general",
    limit: int | None = None,
    include_paths_content: bool = False,
    max_path_chars: int = 2000,
    require_quality: bool = True,
    min_quality_score: float = 0.5,
    score_profile: str = "generic",
    enable_asar: bool = False,
    redact: bool = True,
) -> AntigravityExportResult:
    """Export Antigravity trajectory summaries into TrainingSample JSONL."""
    result = AntigravityExportResult()
    samples: list[TrainingSample] = []

    payloads = _load_state_payloads(
        db_path, state_keys=state_keys or DEFAULT_STATE_KEYS
    )
    if not payloads:
        result.errors.append(f"No Antigravity state payloads found in {db_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return result

    summary_payloads: list[bytes] = []
    for payload in payloads:
        decoded = _decode_nested_payloads(payload)
        summary_payloads.extend(decoded if decoded else [payload])

    deduped_payloads = _dedupe_payloads(summary_payloads)
    result.total_payloads = len(deduped_payloads)

    for payload in deduped_payloads:
        sample = _payload_to_sample(
            payload,
            default_domain=default_domain,
            include_paths_content=include_paths_content,
            max_path_chars=max_path_chars,
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


def _load_state_payloads(
    db_path: Path,
    *,
    state_keys: Iterable[str],
) -> list[bytes]:
    payloads: list[bytes] = []
    if not db_path.exists():
        return payloads
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error:
        return payloads
    try:
        cur = conn.cursor()
        for key in state_keys:
            cur.execute("SELECT value FROM ItemTable WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row or row[0] is None:
                continue
            raw = row[0]
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                payloads.append(base64.b64decode(raw))
            except (ValueError, TypeError):
                continue
    finally:
        conn.close()
    return payloads


def _decode_nested_payloads(payload: bytes, max_depth: int = 5) -> list[bytes]:
    decoded_payloads: list[bytes] = []
    seen: set[str] = set()
    stack: list[tuple[bytes, int]] = [(payload, 0)]
    while stack:
        buf, depth = stack.pop()
        if depth > max_depth or not buf:
            continue
        digest = hashlib.sha256(buf).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        for val in _iter_proto_bytes(buf):
            if not val:
                continue
            if depth < max_depth and len(val) > 2:
                stack.append((val, depth + 1))
            decoded = _decode_base64_bytes(val)
            if decoded:
                decoded_payloads.append(decoded)
                stack.append((decoded, depth + 1))
    return decoded_payloads


def _payload_to_sample(
    payload: bytes,
    *,
    default_domain: str,
    include_paths_content: bool,
    max_path_chars: int,
) -> TrainingSample | None:
    json_objects = _extract_json_objects(payload)
    if not json_objects:
        return None

    task_obj: dict[str, Any] | None = None
    response_obj: dict[str, Any] | None = None
    for obj in json_objects:
        if _is_task_object(obj):
            task_obj = obj
        if _is_response_object(obj):
            response_obj = obj

    if not response_obj:
        return None
    message = response_obj.get("Message")
    if not isinstance(message, str) or not message.strip():
        return None

    instruction = _choose_instruction(task_obj, response_obj, payload)
    if not instruction:
        return None

    input_text = _build_input(
        task_obj,
        response_obj,
        include_paths_content=include_paths_content,
        max_path_chars=max_path_chars,
    )

    sample = TrainingSample(
        instruction=instruction,
        output=message.strip(),
        input=input_text,
        domain=default_domain,
        source="antigravity",
    )
    sample._metadata = _build_metadata(task_obj, response_obj, payload)
    return sample


def _extract_json_objects(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("latin-1", errors="ignore")
    objects: list[dict[str, Any]] = []
    start = None
    in_str = False
    escape = False
    brace_count = 0
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                brace_count = 1
                in_str = False
                escape = False
        else:
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
            else:
                if ch == "\"":
                    in_str = True
                elif ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        segment = text[start : i + 1]
                        if (
                            "\"Message\"" not in segment
                            and "\"TaskName\"" not in segment
                            and "\"TaskSummary\"" not in segment
                        ):
                            start = None
                            continue
                        try:
                            obj = json.loads(segment)
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict):
                            objects.append(obj)
                        start = None
    return objects


def _choose_instruction(
    task_obj: dict[str, Any] | None,
    response_obj: dict[str, Any],
    payload: bytes,
) -> str | None:
    task_name = _string_value(task_obj, "TaskName")
    if task_name and task_name != "%SAME%":
        return task_name

    task_summary = _string_value(task_obj, "TaskSummary")
    if task_summary and task_summary != "%SAME%":
        return task_summary

    response_task_name = _string_value(response_obj, "TaskName")
    if response_task_name and response_task_name != "%SAME%":
        return response_task_name

    title = _extract_title(payload)
    if title:
        return title

    return "Complete the task."


def _build_input(
    task_obj: dict[str, Any] | None,
    response_obj: dict[str, Any],
    *,
    include_paths_content: bool,
    max_path_chars: int,
) -> str:
    parts: list[str] = []
    task_summary = _string_value(task_obj, "TaskSummary")
    if task_summary and task_summary != "%SAME%":
        parts.append(f"Task summary: {task_summary}")
    task_status = _string_value(task_obj, "TaskStatus")
    if task_status and task_status != "%SAME%":
        parts.append(f"Task status: {task_status}")
    mode = _string_value(task_obj, "Mode")
    if mode and mode != "%SAME%":
        parts.append(f"Mode: {mode}")

    paths = _list_value(response_obj, "PathsToReview")
    if paths:
        parts.append("Paths to review:\n" + "\n".join(f"- {p}" for p in paths))
        if include_paths_content:
            parts.extend(
                _load_paths_content(paths, max_chars=max_path_chars)
            )

    blocked = response_obj.get("BlockedOnUser")
    if blocked is not None:
        parts.append(f"Blocked on user: {blocked}")

    return "\n".join(parts).strip()


def _build_metadata(
    task_obj: dict[str, Any] | None,
    response_obj: dict[str, Any] | None,
    payload: bytes,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "antigravity_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if task_obj:
        metadata.update(
            {
                "antigravity_mode": task_obj.get("Mode"),
                "antigravity_predicted_task_size": task_obj.get("PredictedTaskSize"),
                "antigravity_task_name": task_obj.get("TaskName"),
                "antigravity_task_status": task_obj.get("TaskStatus"),
                "antigravity_task_summary": task_obj.get("TaskSummary"),
                "antigravity_wait_for_previous_tools": task_obj.get(
                    "waitForPreviousTools"
                ),
            }
        )
    if response_obj:
        metadata.update(
            {
                "antigravity_blocked_on_user": response_obj.get("BlockedOnUser"),
                "antigravity_confidence_score": response_obj.get("ConfidenceScore"),
                "antigravity_confidence_justification": response_obj.get(
                    "ConfidenceJustification"
                ),
                "antigravity_paths_to_review": response_obj.get("PathsToReview"),
                "antigravity_should_auto_proceed": response_obj.get(
                    "ShouldAutoProceed"
                ),
            }
        )
    return metadata


def _decode_varint(buf: bytes, idx: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while True:
        if idx >= len(buf):
            raise ValueError("varint out of range")
        byte = buf[idx]
        idx += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, idx
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")


def _iter_proto_bytes(buf: bytes) -> Iterable[bytes]:
    idx = 0
    while idx < len(buf):
        try:
            key, idx = _decode_varint(buf, idx)
        except ValueError:
            break
        wire = key & 0x7
        if wire == 0:
            try:
                _, idx = _decode_varint(buf, idx)
            except ValueError:
                break
        elif wire == 1:
            idx += 8
        elif wire == 2:
            try:
                length, idx = _decode_varint(buf, idx)
            except ValueError:
                break
            value = buf[idx : idx + length]
            idx += length
            yield value
        elif wire == 5:
            idx += 4
        else:
            break


def _decode_base64_bytes(val: bytes) -> bytes | None:
    try:
        text = val.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) < 64 or len(text) % 4 != 0 or not _BASE64_RE.match(text):
        return None
    try:
        return base64.b64decode(text)
    except (ValueError, TypeError):
        return None


def _is_task_object(obj: dict[str, Any]) -> bool:
    return any(key in obj for key in ("TaskName", "TaskSummary", "TaskStatus", "Mode"))


def _is_response_object(obj: dict[str, Any]) -> bool:
    return "Message" in obj


def _string_value(obj: dict[str, Any] | None, key: str) -> str | None:
    if not obj:
        return None
    value = obj.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _list_value(obj: dict[str, Any] | None, key: str) -> list[str]:
    if not obj:
        return []
    value = obj.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _extract_title(payload: bytes) -> str | None:
    strings = _extract_strings(payload)
    for value in strings:
        candidate = value.strip()
        if not candidate:
            continue
        if candidate == "%SAME%":
            continue
        if len(candidate) < 6 or len(candidate) > 120:
            continue
        if _UUID_RE.match(candidate):
            continue
        if _BASE64_RE.match(candidate):
            continue
        if candidate.startswith(("file://", "cci:", "http://", "https://")):
            continue
        if "/" in candidate and " " not in candidate:
            continue
        if candidate.lower() in {"master", "main", "develop"}:
            continue
        if " " not in candidate:
            continue
        return candidate
    return None


def _extract_strings(payload: bytes, max_depth: int = 4) -> list[str]:
    strings: list[str] = []
    seen: set[str] = set()
    stack: list[tuple[bytes, int]] = [(payload, 0)]
    while stack:
        buf, depth = stack.pop()
        if depth > max_depth or not buf:
            continue
        digest = hashlib.sha256(buf).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        for val in _iter_proto_bytes(buf):
            if not val:
                continue
            try:
                text = val.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text:
                strings.append(text)
                decoded = _decode_base64_bytes(val)
                if decoded:
                    stack.append((decoded, depth + 1))
            if depth < max_depth and len(val) > 2:
                stack.append((val, depth + 1))
    return strings


def _dedupe_payloads(payloads: list[bytes]) -> list[bytes]:
    seen: set[str] = set()
    deduped: list[bytes] = []
    for payload in payloads:
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        deduped.append(payload)
    return deduped


def _load_paths_content(paths: list[str], *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for raw in paths:
        path = _normalize_path(raw)
        if path is None or not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not content:
            continue
        snippet = content[:max_chars]
        chunks.append(f"Path content: {path}\n{snippet}")
    return chunks


def _normalize_path(raw: str) -> Path | None:
    if raw.startswith("cci:"):
        idx = raw.find("file://")
        if idx >= 0:
            raw = raw[idx:]
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        if parsed.path:
            return Path(parsed.path)
        return None
    if raw.startswith("/"):
        return Path(raw)
    if raw.startswith("~"):
        return Path(raw).expanduser()
    return None
