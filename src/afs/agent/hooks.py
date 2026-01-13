"""Hooks for agent execution events.

Primary use case: auto-export quality interactions for training data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harness import AgentResult

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_OUTPUT_DIR = Path.home() / ".context" / "training_pools"


@dataclass
class HookConfig:
    """Configuration for training export hook."""

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    min_quality: float = 0.6
    domains: list[str] = field(default_factory=lambda: ["asm65816", "snes", "alttp"])
    include_tool_context: bool = True
    max_samples_per_file: int = 1000


@dataclass
class TrainingSample:
    """A single training sample in unified format.

    Compatible with existing training export infrastructure.
    """

    input_text: str
    output_text: str
    domain: str
    source: str  # "agent_harness", "claude_export", etc.
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input_text,
            "output": self.output_text,
            "domain": self.domain,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class TrainingExportHook:
    """Hook to auto-export high-quality agent interactions.

    Attaches to AgentHarness and exports interactions that meet
    quality thresholds to domain-specific training pools.

    Example:
        ```python
        hook = TrainingExportHook(HookConfig(min_quality=0.7))
        harness = AgentHarness("nayru-v5:latest", tools=TRIFORCE_TOOLS)
        harness.add_hook(hook)

        async with harness:
            result = await harness.run("Write DMA code")
            # High-quality interactions auto-exported
        ```
    """

    def __init__(self, config: HookConfig | None = None):
        self.config = config or HookConfig()
        self._scorer = None  # Lazy load

    def _ensure_scorer(self):
        """Lazily load quality scorer."""
        if self._scorer is None:
            try:
                from ..training.quality_scorer import QualityScorer
                self._scorer = QualityScorer()
            except ImportError:
                logger.warning("QualityScorer not available, using simple heuristics")
                self._scorer = SimpleQualityScorer()

    async def on_agent_complete(self, result: "AgentResult") -> None:
        """Called when agent completes a task.

        Converts the interaction to training samples, scores them,
        and exports those meeting quality threshold.
        """
        if not result.success:
            logger.debug("Skipping failed interaction")
            return

        # Convert to training samples
        samples = self._result_to_samples(result)

        if not samples:
            logger.debug("No samples extracted from interaction")
            return

        # Score and filter
        self._ensure_scorer()
        exported = 0

        for sample in samples:
            score = self._scorer.score(sample)

            if score >= self.config.min_quality:
                # Check domain relevance
                if self._is_relevant_domain(sample):
                    self._append_to_pool(sample, score)
                    exported += 1

        if exported:
            logger.info(f"Exported {exported} training samples")

    def _result_to_samples(self, result: "AgentResult") -> list[TrainingSample]:
        """Convert agent result to training samples.

        Creates samples from:
        1. User prompt -> final response (main sample)
        2. Tool executions (if include_tool_context enabled)
        """
        samples = []

        # Extract user prompt and response
        user_prompt = ""
        for msg in result.history:
            if msg["role"] == "user" and not user_prompt:
                user_prompt = msg.get("content", "")
                break

        if not user_prompt or not result.response:
            return samples

        # Detect domain from content
        domain = self._detect_domain(user_prompt, result.response)

        # Main sample: prompt -> response
        main_sample = TrainingSample(
            input_text=user_prompt,
            output_text=result.response,
            domain=domain,
            source="agent_harness",
            timestamp=datetime.now(),
            metadata={
                "model": result.metadata.get("model", "unknown"),
                "provider": result.metadata.get("provider", "unknown"),
                "iterations": result.iterations,
                "tool_count": len(result.tool_executions),
            },
        )
        samples.append(main_sample)

        # Tool context samples
        if self.config.include_tool_context and result.tool_executions:
            for te in result.tool_executions:
                if not te.result.success:
                    continue

                # Create sample showing tool usage
                tool_input = f"Tool: {te.name}\nArguments: {json.dumps(te.arguments, indent=2)}"
                tool_output = te.result.content

                if tool_output and len(tool_output) > 50:
                    tool_sample = TrainingSample(
                        input_text=tool_input,
                        output_text=tool_output,
                        domain=domain,
                        source="agent_harness_tool",
                        timestamp=te.timestamp,
                        metadata={
                            "tool_name": te.name,
                            "parent_prompt": user_prompt[:200],
                        },
                    )
                    samples.append(tool_sample)

        return samples

    def _detect_domain(self, prompt: str, response: str) -> str:
        """Detect the domain of a sample based on content."""
        combined = (prompt + response).lower()

        # Check for 65816/assembly indicators
        asm_keywords = [
            "lda", "sta", "jsr", "jsl", "rts", "rtl",
            "rep #$", "sep #$", "65816", "asar", "snes",
            "dma", "vram", "oam", "spc700",
        ]
        if any(kw in combined for kw in asm_keywords):
            return "asm65816"

        # Check for ALTTP-specific terms
        alttp_keywords = [
            "alttp", "zelda", "link", "dungeon", "overworld",
            "sprite", "tile", "palette", "room", "entrance",
        ]
        if any(kw in combined for kw in alttp_keywords):
            return "alttp"

        # Check for general SNES
        snes_keywords = [
            "snes", "super nintendo", "mode 7", "hdma",
            "bsnes", "snes9x", "emulator",
        ]
        if any(kw in combined for kw in snes_keywords):
            return "snes"

        return "general"

    def _is_relevant_domain(self, sample: TrainingSample) -> bool:
        """Check if sample is in a relevant domain."""
        return sample.domain in self.config.domains

    def _append_to_pool(self, sample: TrainingSample, score: float) -> None:
        """Append sample to domain-specific training pool."""
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Add quality score to metadata
        sample.metadata["quality_score"] = score
        sample.metadata["export_timestamp"] = datetime.now().isoformat()

        # Append to pool file
        pool_path = self.config.output_dir / f"{sample.domain}_pool.jsonl"

        try:
            with pool_path.open("a") as f:
                f.write(json.dumps(sample.to_dict()) + "\n")

            logger.debug(f"Appended sample to {pool_path}")

        except Exception as e:
            logger.error(f"Failed to write sample: {e}")


class SimpleQualityScorer:
    """Simple heuristic quality scorer.

    Used as fallback when full QualityScorer isn't available.
    """

    def score(self, sample: TrainingSample) -> float:
        """Score a sample based on simple heuristics."""
        score = 0.5  # Base score

        # Length factors
        input_len = len(sample.input_text)
        output_len = len(sample.output_text)

        # Reasonable length bonus
        if 50 < input_len < 2000:
            score += 0.1
        if 100 < output_len < 5000:
            score += 0.1

        # Code block bonus
        if "```" in sample.output_text:
            score += 0.15

        # Assembly-specific indicators
        asm_patterns = [
            "LDA", "STA", "JSR", "JMP", "BNE", "BEQ",
            "REP #$", "SEP #$", "org $", "lorom",
        ]
        for pattern in asm_patterns:
            if pattern in sample.output_text:
                score += 0.03

        # Domain relevance bonus
        if sample.domain in ["asm65816", "alttp", "snes"]:
            score += 0.1

        # Tool context bonus
        if sample.source == "agent_harness_tool":
            score += 0.05

        return min(1.0, score)


def create_training_hook(
    output_dir: Path | str | None = None,
    min_quality: float = 0.6,
    domains: list[str] | None = None,
) -> TrainingExportHook:
    """Create a configured training export hook.

    Args:
        output_dir: Where to write training pools
        min_quality: Minimum quality threshold (0-1)
        domains: Domains to export (default: asm65816, snes, alttp)

    Returns:
        Configured TrainingExportHook
    """
    config = HookConfig(
        output_dir=Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR,
        min_quality=min_quality,
        domains=domains or ["asm65816", "snes", "alttp"],
    )
    return TrainingExportHook(config)
