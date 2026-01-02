"""Output format handlers for Chain of Thought samples."""

from __future__ import annotations

from enum import Enum
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import TrainingSample


class CotFormat(str, Enum):
    """Supported Chain of Thought output formats."""

    SEPARATE = "separate"  # Thinking in dedicated field
    EMBEDDED = "embedded"  # Reasoning before output in same field
    SPECIAL_TOKENS = "special_tokens"  # <think>...</think> markers


def format_cot_sample(
    sample: "TrainingSample",
    thinking: str,
    format_type: CotFormat,
) -> "TrainingSample":
    """Format a sample with CoT based on specified format.

    Args:
        sample: Original training sample
        thinking: Generated reasoning/thinking text
        format_type: How to incorporate the thinking

    Returns:
        New TrainingSample with thinking incorporated
    """
    from ..base import TrainingSample

    if format_type == CotFormat.SEPARATE:
        # Store thinking in dedicated field, output unchanged
        return TrainingSample(
            instruction=sample.instruction,
            output=sample.output,
            input=sample.input,
            thinking=thinking,
            domain=sample.domain,
            source=f"{sample.source}:cot",
            sample_id=sample.sample_id,
            quality_score=sample.quality_score,
            embedding=None,  # Needs recalculation
            teacher_model=sample.teacher_model,
            teacher_prompt="",
            timestamp=sample.timestamp,
            kg_entities=sample.kg_entities,
            kg_validated=False,
            _metadata={
                **sample._metadata,
                "cot_format": "separate",
            },
        )

    elif format_type == CotFormat.EMBEDDED:
        # Embed reasoning before the actual output
        formatted_output = f"""Let me analyze this step by step:

{thinking}

Based on this analysis, here is the solution:

{sample.output}"""

        return TrainingSample(
            instruction=sample.instruction,
            output=formatted_output,
            input=sample.input,
            thinking=None,
            domain=sample.domain,
            source=f"{sample.source}:cot-embedded",
            sample_id=sample.sample_id,
            quality_score=sample.quality_score,
            embedding=None,
            teacher_model=sample.teacher_model,
            teacher_prompt="",
            timestamp=sample.timestamp,
            kg_entities=sample.kg_entities,
            kg_validated=False,
            _metadata={
                **sample._metadata,
                "cot_format": "embedded",
                "original_output": sample.output,
            },
        )

    elif format_type == CotFormat.SPECIAL_TOKENS:
        # Use special tokens like Claude's thinking format
        formatted_output = f"<think>\n{thinking}\n</think>\n\n{sample.output}"

        return TrainingSample(
            instruction=sample.instruction,
            output=formatted_output,
            input=sample.input,
            thinking=None,
            domain=sample.domain,
            source=f"{sample.source}:cot-tokens",
            sample_id=sample.sample_id,
            quality_score=sample.quality_score,
            embedding=None,
            teacher_model=sample.teacher_model,
            teacher_prompt="",
            timestamp=sample.timestamp,
            kg_entities=sample.kg_entities,
            kg_validated=False,
            _metadata={
                **sample._metadata,
                "cot_format": "special_tokens",
                "original_output": sample.output,
            },
        )

    else:
        raise ValueError(f"Unknown CoT format: {format_type}")


def extract_thinking_from_output(output: str) -> tuple[str | None, str]:
    """Extract thinking from output if present.

    Returns:
        Tuple of (thinking, clean_output)
    """
    # Check for special token format
    if output.startswith("<think>"):
        end_idx = output.find("</think>")
        if end_idx != -1:
            thinking = output[7:end_idx].strip()
            clean_output = output[end_idx + 8 :].strip()
            return thinking, clean_output

    # Check for embedded format
    if output.startswith("Let me analyze"):
        # Look for the solution marker
        marker = "Based on this analysis, here is the solution:"
        marker_idx = output.find(marker)
        if marker_idx != -1:
            thinking = output[: marker_idx].strip()
            # Remove the intro line
            if thinking.startswith("Let me analyze this step by step:"):
                thinking = thinking[33:].strip()
            clean_output = output[marker_idx + len(marker) :].strip()
            return thinking, clean_output

    return None, output
