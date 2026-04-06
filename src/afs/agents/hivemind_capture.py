"""Hivemind capture: scan a thoughts.org file for candidate hivemind entries.

Reads dated entries from a thoughts.org file within a lookback window and
proposes new hivemind candidates (fears / satisfactions / preferences /
decisions / knowledge) to ``<hivemind_dir>/proposals-YYYY-MM-DD.json``.

Two extraction modes:

  - heuristic (default): keyword/regex pass that surfaces candidate sentences
    grouped by category. Cheap, deterministic, no LLM dependency.

  - llm (--llm): pipes the dated entries through ``aiq`` (or any compatible
    ``--provider`` override) and asks for structured JSON proposals. Slower
    but produces more cohesive entries. Falls back to heuristic on failure.

Path resolution (highest precedence first):
    1. CLI flags ``--thoughts`` / ``--hivemind-dir``
    2. ``AFS_JOURNAL_THOUGHTS`` (or ``AFS_JOURNAL_ROOT/thoughts.org``) for the source
    3. ``AFS_HIVEMIND_PROPOSALS_DIR`` (or
       ``AFS_PERSONAL_CONTEXT_ROOT/hivemind``) for the output
    4. Generic fallback: ``~/.local/share/afs/journal/thoughts.org`` and
       ``~/.config/afs/personal/hivemind``

Never auto-merges into the canonical hivemind files. Proposals always have
``golden=false`` and require manual review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    now_iso,
)
from .journal_agent import default_thoughts_path
from ..personal_context import default_context_root

AGENT_NAME = "hivemind-capture"
AGENT_DESCRIPTION = (
    "Scan a thoughts.org file for candidate hivemind entries and write a "
    "proposals JSON file for manual review."
)

AGENT_CAPABILITIES = {
    "mount_types": [],
    "topics": ["personal:hivemind"],
    "tools": [],
    "description": (
        "Reads dated entries from a thoughts.org file within a lookback window "
        "and proposes hivemind candidates (fears, satisfactions, preferences, "
        "decisions, knowledge). Writes to "
        "<hivemind_dir>/proposals-YYYY-MM-DD.json. Never auto-merges."
    ),
}


def default_hivemind_dir() -> Path:
    """Resolve the default hivemind proposals directory.

    Order: ``AFS_HIVEMIND_PROPOSALS_DIR`` env var, then ``<personal_context>/hivemind``.
    """
    explicit = os.environ.get("AFS_HIVEMIND_PROPOSALS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    return default_context_root() / "hivemind"

_THOUGHT_DATE_RE = re.compile(r"^\*\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*$")
_MONTHS = {
    m.lower(): i + 1
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
}

_CATEGORIES = ("fears", "satisfactions", "preferences", "decisions", "knowledge")

# Heuristic keyword sets per category — surface sentences containing any.
# These are intentionally noisy: the user reviews proposals before merging.
_HEURISTICS: dict[str, tuple[str, ...]] = {
    "fears": (
        "afraid", "scared", "worry", "worried", "anxious", "terrified",
        "what if", "i fear", "dread", "panic", "can't shake",
    ),
    "satisfactions": (
        "love", "loved", "satisfying", "felt good", "recharged", "alive",
        "in flow", "connected", "joy", "happy", "made my day", "best part",
    ),
    "preferences": (
        "i prefer", "i like", "i don't like", "i hate", "i want", "i need",
        "i wish", "i refuse", "rather", "always", "never",
    ),
    "decisions": (
        "i decided", "i'm going to", "decided to", "i'll", "made up my mind",
        "i should", "i need to", "leaning toward", "thinking about",
    ),
    "knowledge": (
        "i realized", "i learned", "turns out", "the thing is", "the truth is",
        "actually", "it's clear", "i think", "i know now",
    ),
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _parse_thought_date(line: str) -> date | None:
    m = _THOUGHT_DATE_RE.match(line.strip())
    if not m:
        return None
    day_str, month_str, year_str = m.groups()
    month = _MONTHS.get(month_str.lower())
    if not month:
        return None
    try:
        return date(int(year_str), month, int(day_str))
    except ValueError:
        return None


def extract_recent_entries(
    text: str,
    cutoff: date,
) -> list[tuple[date, str]]:
    """Return (date, body) tuples for thoughts.org entries on or after cutoff."""
    entries: list[tuple[date, list[str]]] = []
    current_date: date | None = None
    current_body: list[str] = []

    for line in text.splitlines():
        d = _parse_thought_date(line)
        if d is not None:
            if current_date is not None and current_date >= cutoff:
                entries.append((current_date, current_body))
            current_date = d
            current_body = []
            continue
        if current_date is not None:
            current_body.append(line)

    if current_date is not None and current_date >= cutoff:
        entries.append((current_date, current_body))

    return [(d, "\n".join(body).strip()) for d, body in entries]


def _split_sentences(body: str) -> list[str]:
    """Split a block into rough sentences, ignoring blank-only fragments."""
    flat = re.sub(r"\s+", " ", body).strip()
    if not flat:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(flat) if s.strip()]


def heuristic_extract(
    entries: list[tuple[date, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Heuristic candidate extraction by keyword match."""
    proposals: dict[str, list[dict[str, Any]]] = {cat: [] for cat in _CATEGORIES}

    for entry_date, body in entries:
        sentences = _split_sentences(body)
        for sent in sentences:
            lowered = sent.lower()
            for category, keywords in _HEURISTICS.items():
                if any(kw in lowered for kw in keywords):
                    proposals[category].append(
                        {
                            "content": sent,
                            "source": f"thoughts.org:{entry_date.isoformat()}",
                            "confidence": None,
                            "golden": False,
                            "created": entry_date.isoformat(),
                            "tags": [],
                            "extraction": "heuristic",
                        }
                    )
                    break  # one category per sentence to keep noise down

    return proposals


class LLMExtractError(Exception):
    """Raised when LLM extraction fails. Caller decides whether to fall back."""


_RECOGNIZED_KEYS = set(_CATEGORIES) | {
    "fear",
    "satisfaction",
    "preference",
    "decision",
    "knowledge_item",
}


def _has_recognized_key(parsed: dict) -> bool:
    return any(
        isinstance(k, str) and k.strip().lower() in _RECOGNIZED_KEYS
        for k in parsed.keys()
    )


def _iter_balanced_objects(text: str):
    """Yield every balanced JSON object substring found in ``text``."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : idx + 1]
                    break
        start = text.find("{", start + 1)


def _extract_json_object(text: str) -> dict | None:
    """Pull a JSON object out of LLM output, tolerating several formats.

    Tries, in order:
      1. Fenced ```json { ... } ``` (preferred)
      2. Fenced ``` { ... } ``` (no language tag)
      3. Whole text as JSON
      4. Balanced brace scan — preferring objects whose keys include
         recognized hivemind categories. Falls back to the first valid
         object if none of them match.
    """
    text = text.strip()
    if not text:
        return None

    fence_patterns = (
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    )
    for pattern in fence_patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except json.JSONDecodeError:
        pass

    first_valid: dict | None = None
    for candidate in _iter_balanced_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if _has_recognized_key(parsed):
            return parsed
        if first_valid is None:
            first_valid = parsed
    return first_valid


def _validate_and_normalize_proposals(
    parsed: dict,
) -> dict[str, list[dict[str, Any]]]:
    """Coerce a parsed JSON object into the canonical proposals shape.

    Tolerates missing categories (treated as empty), category aliases
    (singular ↔ plural), and items missing optional fields. Raises
    LLMExtractError only if NO categories produce any valid items AND the
    object is structurally unusable.
    """
    # Accept either the expected key or a singular alias
    aliases = {
        "fear": "fears",
        "satisfaction": "satisfactions",
        "preference": "preferences",
        "decision": "decisions",
        "knowledge_item": "knowledge",
    }

    by_category: dict[str, list] = {cat: [] for cat in _CATEGORIES}
    for raw_key, raw_items in parsed.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip().lower()
        canonical = aliases.get(key, key)
        if canonical not in _CATEGORIES:
            continue
        if isinstance(raw_items, list):
            by_category[canonical].extend(raw_items)
        elif isinstance(raw_items, dict):
            by_category[canonical].append(raw_items)

    normalized: dict[str, list[dict[str, Any]]] = {cat: [] for cat in _CATEGORIES}
    total_items = 0
    for cat, items in by_category.items():
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("text") or item.get("body")
            if not isinstance(content, str) or not content.strip():
                continue
            source = item.get("source") or "thoughts.org"
            confidence = item.get("confidence")
            if not isinstance(confidence, (int, float)):
                confidence = None
            tags_raw = item.get("tags") or []
            if isinstance(tags_raw, list):
                tags = [t for t in tags_raw if isinstance(t, str)]
            else:
                tags = []
            normalized[cat].append(
                {
                    "content": content.strip(),
                    "source": source,
                    "confidence": confidence,
                    "golden": False,
                    "created": date.today().isoformat(),
                    "tags": tags,
                    "extraction": "llm",
                }
            )
            total_items += 1

    if total_items == 0:
        # Object had recognized keys but no usable items.
        # Return the empty shape — caller can decide if that's a real result.
        pass

    return normalized


def _llm_extract(
    entries: list[tuple[date, str]],
    aiq_path: str,
) -> tuple[dict[str, list[dict[str, Any]]] | None, str]:
    """Try to extract structured proposals via aiq.

    Returns ``(proposals, status)``. ``proposals`` is None on failure;
    ``status`` is a short human-readable reason.
    """
    if not entries:
        return {cat: [] for cat in _CATEGORIES}, "no entries"

    body_blob = "\n\n".join(
        f"## {d.isoformat()}\n{body}" for d, body in entries
    )

    prompt = (
        "You are extracting candidate personal hivemind entries from journal text. "
        "Return ONLY a JSON object with keys: fears, satisfactions, preferences, "
        "decisions, knowledge. Each value is a list of objects with these fields: "
        '{"content": str, "source": "thoughts.org:YYYY-MM-DD", '
        '"confidence": 0.0-1.0, "tags": [str]}. '
        "Be conservative — only include entries that express a clear personal "
        "pattern, fear, satisfaction, preference, or decision. Skip generic "
        "observations. Output the JSON object only — no prose, no markdown fences "
        "are required, but if you use them prefer ```json fences.\n\n"
        f"Journal text:\n{body_blob}"
    )

    try:
        result = subprocess.run(
            [aiq_path, "--no-stream", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return None, f"aiq binary not found: {aiq_path}"
    except subprocess.TimeoutExpired:
        return None, "aiq timed out after 120s"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else f"exit={result.returncode}"
        return None, f"aiq failed: {tail}"

    output = (result.stdout or "").strip()
    if not output:
        return None, "aiq returned empty output"

    parsed = _extract_json_object(output)
    if parsed is None:
        return None, "no JSON object found in aiq output"

    try:
        normalized = _validate_and_normalize_proposals(parsed)
    except LLMExtractError as exc:
        return None, str(exc)

    total = sum(len(items) for items in normalized.values())
    return normalized, f"llm produced {total} candidate(s)"


def _existing_content_set(hivemind_dir: Path) -> set[str]:
    """Return all existing hivemind 'content' strings to dedupe proposals."""
    seen: set[str] = set()
    for cat in _CATEGORIES:
        path = hivemind_dir / f"{cat}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    content = entry.get("content")
                    if isinstance(content, str):
                        seen.add(content.strip().lower())
    return seen


def _dedupe_against_existing(
    proposals: dict[str, list[dict[str, Any]]],
    existing: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Drop proposals whose content is already present in canonical files."""
    out: dict[str, list[dict[str, Any]]] = {}
    for cat, items in proposals.items():
        kept = []
        seen_local: set[str] = set()
        for item in items:
            key = item["content"].strip().lower()
            if key in existing or key in seen_local:
                continue
            seen_local.add(key)
            kept.append(item)
        out[cat] = kept
    return out


def write_proposals(
    proposals: dict[str, list[dict[str, Any]]],
    hivemind_dir: Path,
    today: date,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    hivemind_dir.mkdir(parents=True, exist_ok=True)
    out_path = hivemind_dir / f"proposals-{today.isoformat()}.json"
    if out_path.exists() and not overwrite:
        return out_path, {
            "action": "skipped",
            "reason": "proposals file already exists (pass --overwrite to refresh)",
            "path": str(out_path),
        }

    payload = {
        "generated_at": now_iso(),
        "schema_version": 1,
        "review_instructions": (
            "Review each proposal. Move accepted entries into the matching "
            "fears/satisfactions/preferences/decisions/knowledge.json file with "
            "a real id, confidence, and tags. Delete this proposals file when done."
        ),
        "proposals": proposals,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    counts = {cat: len(items) for cat, items in proposals.items()}
    return out_path, {
        "action": "created",
        "path": str(out_path),
        "counts": counts,
        "total": sum(counts.values()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--thoughts",
        default=None,
        help=(
            "Path to thoughts.org. Defaults to $AFS_JOURNAL_THOUGHTS, "
            "$AFS_JOURNAL_ROOT/thoughts.org, or "
            "~/.local/share/afs/journal/thoughts.org."
        ),
    )
    parser.add_argument(
        "--hivemind-dir",
        default=None,
        help=(
            "Hivemind directory for proposals + dedupe. Defaults to "
            "$AFS_HIVEMIND_PROPOSALS_DIR, $AFS_PERSONAL_CONTEXT_ROOT/hivemind, "
            "or ~/.config/afs/personal/hivemind."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Scan thoughts.org entries this many days back (default: 14).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use aiq (LLM) for structured extraction. Falls back to heuristic on failure.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="When --llm fails, return non-zero instead of falling back to heuristic.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refresh today's proposals file if it already exists.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)

    thoughts_path = (
        Path(args.thoughts).expanduser() if args.thoughts else default_thoughts_path()
    )
    hivemind_dir = (
        Path(args.hivemind_dir).expanduser()
        if args.hivemind_dir
        else default_hivemind_dir()
    )
    today = date.today()
    cutoff = today - timedelta(days=max(args.lookback_days, 1))

    started_at = now_iso()
    start = time.monotonic()

    notes: list[str] = []

    if not thoughts_path.exists():
        result = AgentResult(
            name=AGENT_NAME,
            status="warn",
            started_at=started_at,
            finished_at=now_iso(),
            duration_seconds=time.monotonic() - start,
            task="hivemind:capture",
            metrics={},
            notes=[f"thoughts.org not found at {thoughts_path}"],
            payload={},
        )
        emit_result(
            result,
            output_path=Path(args.output) if args.output else None,
            force_stdout=args.stdout,
            pretty=args.pretty,
        )
        return 0

    text = thoughts_path.read_text(encoding="utf-8")
    entries = extract_recent_entries(text, cutoff)

    extraction_mode = "heuristic"
    proposals: dict[str, list[dict[str, Any]]]
    llm_status: str | None = None
    if args.llm:
        aiq_path = shutil.which("aiq")
        if aiq_path:
            llm_result, llm_status = _llm_extract(entries, aiq_path)
            if llm_result is not None:
                proposals = llm_result
                extraction_mode = "llm"
                notes.append(f"llm extraction: {llm_status}")
            else:
                notes.append(f"llm extraction failed: {llm_status}")
                if args.no_fallback:
                    result = AgentResult(
                        name=AGENT_NAME,
                        status="error",
                        started_at=started_at,
                        finished_at=now_iso(),
                        duration_seconds=time.monotonic() - start,
                        task="hivemind:capture",
                        metrics={"entries_scanned": len(entries)},
                        notes=notes,
                        payload={"llm_status": llm_status, "extraction_mode": "llm"},
                    )
                    emit_result(
                        result,
                        output_path=Path(args.output) if args.output else None,
                        force_stdout=args.stdout,
                        pretty=args.pretty,
                    )
                    return 1
                notes.append("falling back to heuristic")
                proposals = heuristic_extract(entries)
        else:
            llm_status = "aiq not found in PATH"
            notes.append(f"llm extraction failed: {llm_status}")
            if args.no_fallback:
                result = AgentResult(
                    name=AGENT_NAME,
                    status="error",
                    started_at=started_at,
                    finished_at=now_iso(),
                    duration_seconds=time.monotonic() - start,
                    task="hivemind:capture",
                    metrics={"entries_scanned": len(entries)},
                    notes=notes,
                    payload={"llm_status": llm_status, "extraction_mode": "llm"},
                )
                emit_result(
                    result,
                    output_path=Path(args.output) if args.output else None,
                    force_stdout=args.stdout,
                    pretty=args.pretty,
                )
                return 1
            notes.append("falling back to heuristic")
            proposals = heuristic_extract(entries)
    else:
        proposals = heuristic_extract(entries)

    existing = _existing_content_set(hivemind_dir)
    deduped = _dedupe_against_existing(proposals, existing)
    dropped = sum(len(p) for p in proposals.values()) - sum(
        len(p) for p in deduped.values()
    )

    out_path, meta = write_proposals(deduped, hivemind_dir, today, overwrite=args.overwrite)
    notes.append(f"capture: {meta['action']} → {out_path}")
    if dropped:
        notes.append(f"deduped {dropped} proposal(s) already present in canonical files")

    metrics = {
        "entries_scanned": len(entries),
        "lookback_days": args.lookback_days,
        "extraction_mode": extraction_mode,
        "total_proposals": meta.get("total", 0),
    }
    if llm_status is not None:
        metrics["llm_status"] = llm_status

    finished_at = now_iso()
    duration = time.monotonic() - start

    result = AgentResult(
        name=AGENT_NAME,
        status="ok",
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        task="hivemind:capture",
        metrics=metrics,
        notes=notes,
        payload={
            "capture": meta,
            "extraction_mode": extraction_mode,
            "cutoff_date": cutoff.isoformat(),
        },
    )

    emit_result(
        result,
        output_path=Path(args.output) if args.output else None,
        force_stdout=args.stdout,
        pretty=args.pretty,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
