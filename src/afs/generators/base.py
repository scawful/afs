"""Base classes for training data generators."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Instruction Cleaning Utilities
# =============================================================================

# Patterns to strip from instructions
FILE_MARKER_PATTERNS = [
    # Address ranges like "; *$EE1ED-$EE213 LOCAL" or "*$12345-$12367 LONG"
    r";\s*\*?\$[0-9A-Fa-f]+-\$[0-9A-Fa-f]+\s*(LOCAL|LONG|JUMP\s*LOCATION|ALTERNATE\s*ENTRY\s*POINT|DATA|MAIN\s*ENTRY\s*POINT)?\.?",
    r"\*\$[0-9A-Fa-f]+-\$[0-9A-Fa-f]+\s*(LOCAL|LONG|JUMP\s*LOCATION|ALTERNATE\s*ENTRY\s*POINT|DATA|MAIN\s*ENTRY\s*POINT)?\.?",
    # Equals separator lines (4 or more)
    r"={4,}",
    # Dashed separator lines (4 or more)
    r"-{4,}",
    # TODO markers with separator lines
    r";\s*TODO\s*$",
    # Standalone address comments like "; $EE1ED" at end of line
    r";\s*\$[0-9A-Fa-f]+\s*$",
]

# Patterns that indicate an output is just a file marker (not a real summary)
BAD_OUTPUT_PATTERNS = [
    # Just an address range
    r"^\s*;\s*\*?\$[0-9A-Fa-f]+-\$[0-9A-Fa-f]+\s*(LOCAL|LONG|JUMP\s*LOCATION|ALTERNATE\s*ENTRY\s*POINT|DATA|MAIN\s*ENTRY\s*POINT)?\s*$",
    r"^\s*\*\$[0-9A-Fa-f]+-\$[0-9A-Fa-f]+\s*(LOCAL|LONG|JUMP\s*LOCATION|ALTERNATE\s*ENTRY\s*POINT|DATA|MAIN\s*ENTRY\s*POINT)?\s*$",
    # Just separator lines
    r"^\s*[=\-]{4,}\s*$",
    # Just a comment with address
    r"^\s*;\s*\$[0-9A-Fa-f]+-\$[0-9A-Fa-f]+\s*(DATA)?\s*$",
]

# Compile patterns for efficiency
_file_marker_regexes = [re.compile(p, re.IGNORECASE) for p in FILE_MARKER_PATTERNS]
_bad_output_regexes = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in BAD_OUTPUT_PATTERNS]


def clean_instruction(instruction: str) -> tuple[str, bool]:
    """
    Clean file markers and separators from an instruction.

    Removes patterns like:
    - "; *$EE1ED-$EE213 LOCAL"
    - "====" separator lines
    - "----" separator lines

    Returns:
        Tuple of (cleaned_instruction, was_modified)
    """
    original = instruction
    cleaned = instruction

    for pattern in _file_marker_regexes:
        cleaned = pattern.sub("", cleaned)

    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove trailing punctuation that was left behind
    cleaned = re.sub(r"[\s,;:]+$", "", cleaned).strip()

    return cleaned, cleaned != original


def is_malformed_output(output: str, instruction: str) -> bool:
    """
    Check if an output is malformed (e.g., just a file marker instead of content).

    This is particularly relevant for "Summarize" instructions where the output
    should be an actual summary, not a file location marker.

    Args:
        output: The output text to check
        instruction: The instruction (to check for summary requests)

    Returns:
        True if the output is malformed and needs regeneration
    """
    # Check if instruction is asking for a summary
    is_summary_request = "summarize" in instruction.lower()

    if not is_summary_request:
        return False

    # Check if output matches bad patterns
    output_stripped = output.strip()
    for pattern in _bad_output_regexes:
        if pattern.match(output_stripped):
            return True

    # Also check if output is suspiciously short for a summary
    if len(output_stripped) < 50 and is_summary_request:
        # Check if it looks like just an address or marker
        if re.match(r"^\s*[;\*\$0-9A-Fa-f\-\s]+\s*$", output_stripped):
            return True

    return False


def clean_sample_instruction(sample: "TrainingSample") -> "TrainingSample":
    """
    Clean a sample's instruction in-place and return it.

    Args:
        sample: The training sample to clean

    Returns:
        The same sample with cleaned instruction
    """
    cleaned, was_modified = clean_instruction(sample.instruction)
    if was_modified:
        sample.instruction = cleaned
        sample._metadata["instruction_cleaned"] = True
    return sample


@dataclass
class TrainingSample:
    """A single training sample for fine-tuning."""

    instruction: str
    output: str
    input: str = ""
    thinking: str | None = None  # Chain of thought reasoning (separate field)
    domain: str = "asm"
    source: str = "augmented"
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    quality_score: float = 0.0
    embedding: list[float] | None = None
    teacher_model: str = ""
    teacher_prompt: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    kg_entities: list[str] = field(default_factory=list)
    kg_validated: bool = False
    _metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Include metadata if present
        if self._metadata:
            data["_metadata"] = self._metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingSample:
        """Create from dictionary."""
        # Extract known fields
        known_fields = {
            "instruction", "output", "input", "thinking", "domain", "source",
            "sample_id", "quality_score", "embedding", "teacher_model",
            "teacher_prompt", "timestamp", "kg_entities", "kg_validated",
            "_metadata",
        }
        kwargs = {k: v for k, v in data.items() if k in known_fields}

        # Store extra fields in metadata
        extra = {k: v for k, v in data.items() if k not in known_fields}
        if extra:
            metadata = kwargs.get("_metadata", {})
            metadata.update(extra)
            kwargs["_metadata"] = metadata

        return cls(**kwargs)

    def populate_kg_entities(
        self,
        extractor: Any | None = None,
        validate: bool = False,
    ) -> "TrainingSample":
        """Populate kg_entities field by extracting entities from output.

        Uses the EntityExtractor to find ALTTP memory addresses and labels
        in the assembly code output.

        Args:
            extractor: EntityExtractor instance (creates one if None)
            validate: Whether to validate entity usage in context

        Returns:
            Self for chaining
        """
        if extractor is None:
            from afs.knowledge import EntityExtractor
            extractor = EntityExtractor()

        result = extractor.extract(self.output)
        self.kg_entities = result.entity_names()
        self.kg_validated = validate and extractor.validate_entities(
            result.entities, self.output
        ).get("is_valid", False)

        # Store extraction stats in metadata
        self._metadata["kg_extraction"] = result.stats

        return self


@dataclass
class GenerationResult:
    """Result of a generation run."""

    samples: list[TrainingSample] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    source_count: int = 0

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def success_rate(self) -> float:
        if self.source_count == 0:
            return 0.0
        return (self.source_count - self.skipped) / self.source_count


class BaseGenerator:
    """Base class for training data generators."""

    def __init__(self, name: str, domain: str = "generated"):
        self.name = name
        self.domain = domain

    def generate(self) -> GenerationResult:
        """Generate training samples. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate()")


def read_jsonl(path: Path) -> list[TrainingSample]:
    """Read training samples from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(TrainingSample.from_dict(data))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line: {e}")
    return samples


def write_jsonl(samples: list[TrainingSample], path: Path) -> int:
    """Write training samples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
