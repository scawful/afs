"""Multi-expert fusion for combining expert outputs.

Provides methods for merging outputs from multiple experts:
- Weighted merge
- Consensus decoding
- Attention-based fusion
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Available fusion strategies."""
    WEIGHTED = "weighted"
    CONCAT = "concat"
    BEST = "best"
    CONSENSUS = "consensus"


@dataclass
class ExpertOutput:
    """Output from a single expert."""
    expert_name: str
    content: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class FusionConfig:
    """Configuration for output fusion."""
    strategy: FusionStrategy = FusionStrategy.WEIGHTED
    expert_weights: dict[str, float] = field(default_factory=dict)
    min_confidence: float = 0.3
    separator: str = "\n\n---\n\n"


class ExpertFusion:
    """Fuses outputs from multiple experts."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse(self, outputs: list[ExpertOutput]) -> str:
        """Fuse multiple expert outputs into single response."""
        if not outputs:
            return ""

        if len(outputs) == 1:
            return outputs[0].content

        strategy = self.config.strategy

        if strategy == FusionStrategy.WEIGHTED:
            return self._weighted_merge(outputs)
        elif strategy == FusionStrategy.CONCAT:
            return self._concatenate(outputs)
        elif strategy == FusionStrategy.BEST:
            return self._select_best(outputs)
        elif strategy == FusionStrategy.CONSENSUS:
            return self._consensus(outputs)
        else:
            return self._concatenate(outputs)

    def _weighted_merge(self, outputs: list[ExpertOutput]) -> str:
        """Merge outputs with expert-specific weights."""
        # For text, we concatenate with weight-based ordering
        weighted = []
        for output in outputs:
            weight = self.config.expert_weights.get(
                output.expert_name,
                output.confidence
            )
            weighted.append((weight, output))

        # Sort by weight descending
        weighted.sort(key=lambda x: x[0], reverse=True)

        # Build merged output
        parts = []
        for weight, output in weighted:
            if output.confidence >= self.config.min_confidence:
                parts.append(f"**{output.expert_name.title()}** (confidence: {output.confidence:.2f}):\n{output.content}")

        return self.config.separator.join(parts)

    def _concatenate(self, outputs: list[ExpertOutput]) -> str:
        """Simply concatenate all outputs."""
        parts = [
            f"**{o.expert_name.title()}**:\n{o.content}"
            for o in outputs
            if o.confidence >= self.config.min_confidence
        ]
        return self.config.separator.join(parts)

    def _select_best(self, outputs: list[ExpertOutput]) -> str:
        """Select the highest confidence output."""
        best = max(outputs, key=lambda o: o.confidence)
        return best.content

    def _consensus(self, outputs: list[ExpertOutput]) -> str:
        """Find consensus among outputs (simplified)."""
        # For now, return weighted merge
        # Real implementation would use semantic similarity
        return self._weighted_merge(outputs)


@dataclass
class CompositeTask:
    """A task requiring multiple experts."""
    query: str
    required_experts: list[str]
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED
    sequence: bool = False  # True = sequential, False = parallel


class CompositeDataGenerator:
    """Generates training data for multi-expert tasks."""

    def __init__(self, fusion: ExpertFusion):
        self.fusion = fusion

    def create_composite_example(
        self,
        query: str,
        expert_outputs: dict[str, str],
    ) -> dict:
        """Create a training example for composite tasks."""
        outputs = [
            ExpertOutput(expert_name=name, content=content)
            for name, content in expert_outputs.items()
        ]

        fused = self.fusion.fuse(outputs)

        return {
            "instruction": query,
            "input": "",
            "output": fused,
            "metadata": {
                "experts": list(expert_outputs.keys()),
                "type": "composite_task",
            }
        }

    def generate_optimize_explain_example(
        self,
        code: str,
        optimized: str,
        explanation: str,
    ) -> dict:
        """Generate example for 'optimize and explain' task."""
        return self.create_composite_example(
            query=f"Optimize this code and explain the changes:\n```\n{code}\n```",
            expert_outputs={
                "din": f"Optimized code:\n```\n{optimized}\n```",
                "veran": f"Explanation:\n{explanation}",
            }
        )

    def generate_debug_fix_example(
        self,
        buggy_code: str,
        diagnosis: str,
        fixed_code: str,
    ) -> dict:
        """Generate example for 'debug and fix' task."""
        return self.create_composite_example(
            query=f"Find and fix the bug in this code:\n```\n{buggy_code}\n```",
            expert_outputs={
                "farore": f"Diagnosis:\n{diagnosis}",
                "nayru": f"Fixed code:\n```\n{fixed_code}\n```",
            }
        )
