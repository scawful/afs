"""Model registry and version management system for AFS.

Provides comprehensive tracking of:
- Model versions and metadata
- Training lineage and history
- Evaluation scores and performance metrics
- File locations (LoRA, GGUF, checkpoints)
- Deployment status and rollback capability

Example:
    ```python
    from afs.registry import ModelRegistry, ModelVersion

    # Create registry
    registry = ModelRegistry()

    # Register a new model version
    version = registry.register_model(
        model_name="majora",
        base_model="Qwen2.5-Coder-7B",
        training_data=["oracle", "toolbench"],
        evaluation_scores={"accuracy": 0.85, "speed": 120},
        lora_path="/path/to/majora-v1-lora",
        gguf_path="/path/to/majora-v1-Q8_0.gguf",
    )

    # Get model info
    model = registry.get_model("majora")
    print(f"Latest version: {model.latest_version}")

    # List all versions
    versions = registry.list_versions("majora")

    # Compare versions
    diff = registry.compare_versions("majora", "v1", "v2")

    # Rollback
    registry.rollback("majora", "v1")
    ```
"""

from __future__ import annotations

from .database import ModelRegistry
from .lineage import LineageTracker
from .models import (
    EvaluationScores,
    ModelMetadata,
    ModelVersion,
    TrainingMetadata,
    VersionStatus,
    VersionInfo,
)

__all__ = [
    "ModelRegistry",
    "ModelVersion",
    "ModelMetadata",
    "TrainingMetadata",
    "EvaluationScores",
    "VersionStatus",
    "VersionInfo",
    "LineageTracker",
]
