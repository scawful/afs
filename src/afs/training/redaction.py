"""Lightweight redaction helpers for training data exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..generators.base import TrainingSample


@dataclass
class RedactionRule:
    pattern: re.Pattern[str]
    replacement: str


REDACTION_RULES: list[RedactionRule] = [
    RedactionRule(
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    RedactionRule(re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_ACCESS_KEY]"),
    RedactionRule(
        re.compile(r"(?i)aws(.{0,20})?secret(.{0,20})?=[A-Za-z0-9/+=]{20,}"),
        "[REDACTED_AWS_SECRET]",
    ),
    RedactionRule(
        re.compile(r"ghp_[A-Za-z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"
    ),
    RedactionRule(
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    RedactionRule(
        re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"
    ),
    RedactionRule(
        re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED_API_KEY]"
    ),
    RedactionRule(
        re.compile(r"AIza[0-9A-Za-z-_]{35}"), "[REDACTED_GOOGLE_KEY]"
    ),
    RedactionRule(
        re.compile(r"(?:xox[baprs]-|xapp-)[A-Za-z0-9-]{10,}"),
        "[REDACTED_SLACK_TOKEN]",
    ),
    RedactionRule(
        re.compile(r"(?i)bearer\\s+[A-Za-z0-9._-]{20,}"),
        "Bearer [REDACTED_TOKEN]",
    ),
]


def redact_text(text: str) -> tuple[str, int]:
    """Redact secrets from text, returning updated text + count."""
    if not text:
        return text, 0
    redacted = text
    total = 0
    for rule in REDACTION_RULES:
        redacted, count = rule.pattern.subn(rule.replacement, redacted)
        total += count
    return redacted, total


def redact_sample(sample: TrainingSample) -> int:
    """Redact a sample in-place, returning number of replacements."""
    total = 0
    for field in ("instruction", "input", "output", "thinking"):
        value = getattr(sample, field, None)
        if not isinstance(value, str) or not value:
            continue
        redacted, count = redact_text(value)
        if count:
            setattr(sample, field, redacted)
            total += count

    if total:
        metadata = dict(sample._metadata or {})
        metadata["redacted"] = True
        metadata["redaction_count"] = total
        sample._metadata = metadata

    return total


def redact_samples(samples: Iterable[TrainingSample]) -> int:
    """Redact multiple samples, returning total replacements."""
    total = 0
    for sample in samples:
        total += redact_sample(sample)
    return total
