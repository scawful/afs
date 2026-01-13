"""Export memory entries into TrainingSample JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..generators.base import TrainingSample
from .scoring import build_scoring_config, QualityScorer
from .redaction import redact_sample


@dataclass
class MemoryExportResult:
    """Summary of a memory export run."""

    total_entries: int = 0
    exported: int = 0
    skipped: int = 0
    filtered: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"total={self.total_entries} exported={self.exported} "
            f"skipped={self.skipped} filtered={self.filtered} "
            f"errors={len(self.errors)}"
        )


def export_memory_to_dataset(
    memory_root: Path,
    output_path: Path,
    *,
    default_domain: str = "memory",
    allow_raw: bool = False,
    allow_raw_tags: Iterable[str] | None = None,
    default_instruction: str = "Recall the following memory entry.",
    include_tags: Iterable[str] | None = None,
    exclude_tags: Iterable[str] | None = None,
    limit: int | None = None,
    require_quality: bool = True,
    min_quality_score: float = 0.5,
    score_profile: str = "generic",
    enable_asar: bool = False,
    redact: bool = True,
) -> MemoryExportResult:
    """Export memory entries into TrainingSample JSONL."""
    result = MemoryExportResult()
    samples: list[TrainingSample] = []

    include_set = _normalize_tags(include_tags)
    exclude_set = _normalize_tags(exclude_tags)
    allow_raw_set = _normalize_tags(allow_raw_tags)

    for entry, source_path in _iter_memory_entries(memory_root):
        result.total_entries += 1
        sample = _entry_to_sample(
            entry,
            source_path=source_path,
            default_domain=default_domain,
            allow_raw=allow_raw,
            allow_raw_tags=allow_raw_set,
            default_instruction=default_instruction,
            include_tags=include_set,
            exclude_tags=exclude_set,
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


def _iter_memory_entries(memory_root: Path) -> Iterable[tuple[dict, Path]]:
    if not memory_root.exists():
        return []

    entries: list[tuple[dict, Path]] = []
    for path in sorted(memory_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".jsonl":
            entries.extend(_load_jsonl_entries(path))
        elif path.suffix.lower() == ".json":
            entries.extend(_load_json_entries(path))
        elif path.suffix.lower() in {".md", ".txt"}:
            entries.extend(_load_text_entry(path))
    return entries


def _load_jsonl_entries(path: Path) -> list[tuple[dict, Path]]:
    entries: list[tuple[dict, Path]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append((json.loads(line), path))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return entries
    return entries


def _load_json_entries(path: Path) -> list[tuple[dict, Path]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    entries: list[tuple[dict, Path]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                entries.append((item, path))
    elif isinstance(raw, dict):
        entries.append((raw, path))
    return entries


def _load_text_entry(path: Path) -> list[tuple[dict, Path]]:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not content:
        return []
    return [({"content": content}, path)]


def _entry_to_sample(
    entry: dict,
    *,
    source_path: Path,
    default_domain: str,
    allow_raw: bool,
    allow_raw_tags: set[str],
    default_instruction: str,
    include_tags: set[str],
    exclude_tags: set[str],
) -> TrainingSample | None:
    if not isinstance(entry, dict):
        return None

    tags = _extract_tags(entry)
    if include_tags and not (include_tags & tags):
        return None
    if exclude_tags and (exclude_tags & tags):
        return None

    instruction = entry.get("instruction") or entry.get("prompt") or entry.get("user")
    output = entry.get("output") or entry.get("response") or entry.get("assistant")
    input_text = entry.get("input") or ""
    thinking = entry.get("thinking")

    if output is None and allow_raw:
        output = entry.get("content") or entry.get("text")
        if output:
            if allow_raw_tags and not (allow_raw_tags & tags):
                return None
            instruction = instruction or default_instruction

    if not instruction or not output:
        return None

    sample = TrainingSample(
        instruction=str(instruction).strip(),
        output=str(output).strip(),
        input=str(input_text).strip(),
        thinking=str(thinking).strip() if isinstance(thinking, str) else None,
        domain=str(entry.get("domain") or default_domain),
        source=str(entry.get("source") or "memory"),
    )

    metadata = dict(entry.get("_metadata") or {})
    metadata.update(
        {
            "memory_source_path": str(source_path),
            "memory_tags": sorted(tags),
        }
    )
    if "confidence" in entry:
        metadata["memory_confidence"] = entry.get("confidence")
    sample._metadata = metadata
    return sample


def _extract_tags(entry: dict) -> set[str]:
    tags: list[str] = []
    raw_tags = entry.get("tags") or entry.get("labels")
    if isinstance(raw_tags, str):
        tags.append(raw_tags)
    elif isinstance(raw_tags, list):
        tags.extend([tag for tag in raw_tags if isinstance(tag, str)])
    raw_tag = entry.get("tag")
    if isinstance(raw_tag, str):
        tags.append(raw_tag)
    return {tag.strip().lower() for tag in tags if tag and tag.strip()}


def _normalize_tags(tags: Iterable[str] | None) -> set[str]:
    if not tags:
        return set()
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}
