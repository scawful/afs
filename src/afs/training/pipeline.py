"""Training data pipeline.

Provides core pipeline primitives used by extension-owned training and
continuous-learning workflows. Domain-specific pipeline stages belong in a
companion extension repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the training data pipeline."""

    input_paths: list[Path] = field(default_factory=list)
    output_dir: Path = Path("output")
    score_quality: bool = True
    batch_size: int = 32
    deduplicate: bool = True
    min_quality_score: float = 0.5


class DataPipeline:
    """Training data pipeline.

    Orchestrates loading, filtering, scoring, deduplication, and export
    of training samples. This base implementation handles the pipeline
    structure; domain-specific stages belong in a companion extension repo.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self) -> dict:
        """Run the pipeline and return summary statistics."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "input_paths": [str(p) for p in self.config.input_paths],
            "output_dir": str(self.config.output_dir),
            "status": "complete",
            "samples_processed": 0,
        }
