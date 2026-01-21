"""Data models for model registry and version management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class VersionStatus(str, Enum):
    """Status of a model version."""

    DRAFT = "draft"
    TRAINING = "training"
    COMPLETED = "completed"
    DEPLOYED = "deployed"
    PRODUCTION = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class EvaluationScores:
    """Evaluation metrics for a model version."""

    accuracy: float | None = None
    quality_score: float | None = None
    f1_score: float | None = None
    perplexity: float | None = None
    bleu_score: float | None = None
    rouge_scores: dict[str, float] = field(default_factory=dict)
    inference_speed_tokens_per_sec: float | None = None
    memory_usage_mb: float | None = None
    latency_ms: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluationScores:
        """Create from dictionary."""
        if isinstance(data, cls):
            return data
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise TypeError("EvaluationScores.from_dict expects a dict or EvaluationScores")
        # Extract known fields
        known_fields = {
            "accuracy",
            "quality_score",
            "f1_score",
            "perplexity",
            "bleu_score",
            "rouge_scores",
            "inference_speed_tokens_per_sec",
            "memory_usage_mb",
            "latency_ms",
        }
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        custom = {k: v for k, v in data.items() if k not in known_fields}
        return cls(**kwargs, custom_metrics=custom)


@dataclass
class TrainingMetadata:
    """Metadata about model training."""

    framework: str  # "mlx", "unsloth", "huggingface", "llama_cpp"
    base_model: str
    samples: int
    epochs: int
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_seq_length: int = 2048
    duration_hours: float | None = None
    cost_usd: float | None = None
    hardware: str = "unknown"  # "A100", "RTX4090", "M2 Max", etc.
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainingMetadata:
        """Create from dictionary."""
        if not data:
            raise ValueError("TrainingMetadata requires data")
        return cls(**data)


@dataclass
class ModelVersion:
    """A single version of a model."""

    model_name: str
    version: str  # e.g., "v1", "v2.1", "v1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: VersionStatus = VersionStatus.DRAFT

    # File locations
    lora_path: str | None = None
    gguf_path: str | None = None
    checkpoint_path: str | None = None
    config_path: str | None = None

    # Training info
    training: TrainingMetadata | None = None

    # Evaluation
    evaluation_scores: EvaluationScores = field(default_factory=EvaluationScores)

    # Lineage
    parent_version: str | None = None  # For fine-tunes of fine-tunes
    training_data_sources: list[str] = field(default_factory=list)
    git_commit: str | None = None

    # Deployment
    deployed: bool = False
    deployed_at: str | None = None
    deployment_notes: str = ""

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "status": self.status.value,
            "lora_path": self.lora_path,
            "gguf_path": self.gguf_path,
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "training": self.training.to_dict() if self.training else None,
            "evaluation_scores": self.evaluation_scores.to_dict(),
            "parent_version": self.parent_version,
            "training_data_sources": self.training_data_sources,
            "git_commit": self.git_commit,
            "deployed": self.deployed,
            "deployed_at": self.deployed_at,
            "deployment_notes": self.deployment_notes,
            "tags": self.tags,
            "notes": self.notes,
        }
        return data

    @property
    def training_data(self) -> list[str]:
        return self.training_data_sources

    @training_data.setter
    def training_data(self, value: list[str]) -> None:
        self.training_data_sources = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        """Create from dictionary."""
        # Handle status enum
        status_str = data.pop("status", "draft")
        status = VersionStatus(status_str) if isinstance(status_str, str) else status_str

        # Handle training metadata
        training_data = data.pop("training", None)
        training = (
            TrainingMetadata.from_dict(training_data) if training_data else None
        )

        # Handle training data alias
        training_data = data.pop("training_data", None)
        if training_data is not None and "training_data_sources" not in data:
            data["training_data_sources"] = training_data

        # Handle evaluation scores
        eval_data = data.pop("evaluation_scores", {})
        evaluation_scores = EvaluationScores.from_dict(eval_data)

        return cls(
            **data,
            status=status,
            training=training,
            evaluation_scores=evaluation_scores,
        )


@dataclass
class ModelMetadata:
    """Metadata for a complete model (all versions)."""

    model_name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    owner: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    versions: list[ModelVersion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VersionInfo:
    """Summary information about a version."""

    model_name: str
    version: str
    status: str
    created_at: str
    deployed: bool
    accuracy: float | None
    parent_version: str | None
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
