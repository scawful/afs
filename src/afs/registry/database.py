"""JSON database backend for model registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvaluationScores, ModelMetadata, ModelVersion, TrainingMetadata, VersionStatus


class ModelRegistry:
    """JSON-backed registry for tracking model versions and metadata.

    Stores all model metadata in a JSON file with hierarchical structure:
    ```
    {
      "majora": {
        "metadata": { ... },
        "versions": {
          "v1": { ... },
          "v2": { ... }
        }
      }
    }
    ```

    Example:
        ```python
        registry = ModelRegistry()

        # Register new version
        version = registry.register_model(
            model_name="majora",
            version="v1",
            base_model="Qwen2.5-Coder-7B",
            samples=223,
            epochs=3,
            lora_path="/path/to/lora",
            gguf_path="/path/to/gguf",
        )

        # Get latest version
        latest = registry.get_latest("majora")

        # List all versions
        versions = registry.list_versions("majora")

        # Compare versions
        diff = registry.compare_versions("majora", "v1", "v2")

        # Rollback
        registry.rollback("majora", "v1")
        ```
    """

    DEFAULT_PATH = Path.home() / ".context" / "training" / "registry.json"

    def __init__(self, registry_path: Path | None = None):
        """Initialize registry.

        Args:
            registry_path: Path to registry JSON file. Defaults to ~/.context/training/registry.json
        """
        self.registry_path = self._normalize_registry_path(
            Path(registry_path or self.DEFAULT_PATH)
        )
        self.models: dict[str, dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _normalize_registry_path(path: Path) -> Path:
        if path.exists() and path.is_dir():
            return path / "registry.json"
        if path.suffix == "":
            return path / "registry.json"
        return path

    def _load(self) -> None:
        """Load registry from disk."""
        self.registry_path = self._normalize_registry_path(self.registry_path)
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "version" in data:
                        # Format with metadata wrapper
                        self.models = data.get("models", {})
                    else:
                        # Legacy format or direct models dict
                        self.models = data
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to load registry: {e}")
                self.models = {}
        else:
            self.models = {}

    def _save(self) -> None:
        """Save registry to disk."""
        self.registry_path = self._normalize_registry_path(self.registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "models": self.models,
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _init_model(self, model_name: str) -> None:
        """Initialize model entry if it doesn't exist."""
        if model_name not in self.models:
            self.models[model_name] = {
                "metadata": {
                    "model_name": model_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "description": "",
                    "owner": None,
                    "tags": [],
                    "notes": "",
                },
                "versions": {},
            }

    def register_model(
        self,
        model_name: str,
        version: str | None = None,
        base_model: str = "",
        samples: int = 0,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        max_seq_length: int = 2048,
        framework: str = "unsloth",
        training_data: list[str] | None = None,
        lora_path: str | None = None,
        gguf_path: str | None = None,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        evaluation_scores: dict[str, float] | None = None,
        duration_hours: float | None = None,
        cost_usd: float | None = None,
        hardware: str = "unknown",
        git_commit: str | None = None,
        parent_version: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_name: Name of the model (e.g., "majora")
            version: Version string (e.g., "v1"). Auto-increments if None.
            base_model: Base model used (e.g., "Qwen2.5-Coder-7B")
            samples: Number of training samples
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate used
            max_seq_length: Maximum sequence length
            framework: Training framework
            training_data: List of training data sources
            lora_path: Path to LoRA weights
            gguf_path: Path to GGUF quantized model
            checkpoint_path: Path to full checkpoint
            config_path: Path to model config
            evaluation_scores: Dict of evaluation metrics
            duration_hours: Training duration in hours
            cost_usd: Training cost in USD
            hardware: Hardware used for training
            git_commit: Git commit hash of code
            parent_version: Parent version for fine-tuned models
            tags: Tags for categorization
            notes: Free-form notes

        Returns:
            Created ModelVersion
        """
        self._init_model(model_name)

        # Auto-increment version if not provided
        if version is None:
            existing = list(self.models[model_name]["versions"].keys())
            if existing:
                # Extract numeric parts and increment
                versions_nums = []
                for v in existing:
                    try:
                        # Handle v1, v2.1, etc.
                        num = v.lstrip("v").split(".")[0]
                        versions_nums.append(int(num))
                    except (ValueError, IndexError):
                        pass
                next_num = max(versions_nums) + 1 if versions_nums else 1
            else:
                next_num = 1
            version = f"v{next_num}"

        # Create training metadata
        training = TrainingMetadata(
            framework=framework,
            base_model=base_model,
            samples=samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            duration_hours=duration_hours,
            cost_usd=cost_usd,
            hardware=hardware,
        )

        # Create evaluation scores
        if isinstance(evaluation_scores, EvaluationScores):
            eval_scores = evaluation_scores
        else:
            eval_scores = EvaluationScores.from_dict(evaluation_scores)

        # Create version
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            lora_path=lora_path,
            gguf_path=gguf_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            training=training,
            evaluation_scores=eval_scores,
            parent_version=parent_version,
            training_data_sources=training_data or [],
            git_commit=git_commit,
            tags=tags or [],
            notes=notes,
        )

        # Store in registry
        self.models[model_name]["versions"][version] = model_version.to_dict()
        self._save()

        return model_version

    def get_model(self, model_name: str) -> ModelMetadata | None:
        """Get model metadata.

        Args:
            model_name: Name of the model

        Returns:
            ModelMetadata or None if not found
        """
        if model_name not in self.models:
            return None

        metadata_dict = self.models[model_name].get("metadata", {})
        metadata = ModelMetadata.from_dict(metadata_dict)
        metadata.versions = self.list_versions(model_name)
        return metadata

    def get_version(self, model_name: str, version: str) -> ModelVersion | None:
        """Get a specific model version.

        Args:
            model_name: Name of the model
            version: Version string (e.g., "v1")

        Returns:
            ModelVersion or None if not found
        """
        if model_name not in self.models:
            return None

        version_dict = self.models[model_name]["versions"].get(version)
        if version_dict is None:
            return None

        return ModelVersion.from_dict(version_dict)

    def get_latest(self, model_name: str) -> ModelVersion | None:
        """Get the latest version of a model.

        Args:
            model_name: Name of the model

        Returns:
            Latest ModelVersion or None if model not found
        """
        if model_name not in self.models:
            return None

        versions = self.models[model_name]["versions"]
        if not versions:
            return None

        # Sort by version (v1 < v2 < v10, etc.)
        sorted_versions = sorted(
            versions.items(),
            key=lambda x: (
                int(x[0].lstrip("v").split(".")[0]),
                [int(p) for p in x[0].lstrip("v").split(".")[1:]] or [0],
            ),
        )

        latest_version_dict = sorted_versions[-1][1]
        return ModelVersion.from_dict(latest_version_dict)

    def list_versions(self, model_name: str, status: str | None = None) -> list[ModelVersion]:
        """List all versions of a model.

        Args:
            model_name: Name of the model
            status: Optional status filter (e.g., "deployed")

        Returns:
            List of ModelVersion objects sorted by version number
        """
        if model_name not in self.models:
            return []

        versions = self.models[model_name]["versions"]
        result = []

        for version_dict in versions.values():
            version = ModelVersion.from_dict(version_dict)
            if status is None or version.status.value == status:
                result.append(version)

        # Sort by version
        return sorted(
            result,
            key=lambda v: (
                int(v.version.lstrip("v").split(".")[0]),
                [int(p) for p in v.version.lstrip("v").split(".")[1:]] or [0],
            ),
        )

    def list_models(self) -> list[str]:
        """List all registered models.

        Returns:
            List of model names sorted alphabetically
        """
        return sorted(self.models.keys())

    def compare_versions(
        self, model_name: str, version1: str, version2: str
    ) -> dict[str, Any]:
        """Compare two versions of a model.

        Args:
            model_name: Name of the model
            version1: First version string
            version2: Second version string

        Returns:
            Dict with differences between versions
        """
        v1 = self.get_version(model_name, version1)
        v2 = self.get_version(model_name, version2)

        if not v1 or not v2:
            raise ValueError(f"Version not found for {model_name}")

        def _extract_metrics(version: ModelVersion) -> dict[str, Any]:
            """Extract comparable metrics from a version."""
            return {
                "version": version.version,
                "created_at": version.created_at,
                "status": version.status.value,
                "base_model": version.training.base_model if version.training else None,
                "samples": version.training.samples if version.training else None,
                "epochs": version.training.epochs if version.training else None,
                "learning_rate": version.training.learning_rate if version.training else None,
                "accuracy": version.evaluation_scores.accuracy,
                "f1_score": version.evaluation_scores.f1_score,
                "perplexity": version.evaluation_scores.perplexity,
                "bleu_score": version.evaluation_scores.bleu_score,
                "inference_speed": version.evaluation_scores.inference_speed_tokens_per_sec,
                "deployed": version.deployed,
                "parent_version": version.parent_version,
            }

        v1_dict = _extract_metrics(v1)
        v2_dict = _extract_metrics(v2)

        # Find differences
        differences = {}
        all_keys = set(v1_dict.keys()) | set(v2_dict.keys())

        for key in all_keys:
            v1_val = v1_dict.get(key)
            v2_val = v2_dict.get(key)
            if v1_val != v2_val:
                differences[key] = {"v1": v1_val, "v2": v2_val}

        return differences

    def update_evaluation_scores(
        self, model_name: str, version: str, **scores: float
    ) -> None:
        """Update evaluation scores for a version.

        Args:
            model_name: Name of the model
            version: Version string
            **scores: Score name-value pairs
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        if version not in self.models[model_name]["versions"]:
            raise ValueError(f"Version not found: {version}")

        version_dict = self.models[model_name]["versions"][version]
        eval_scores = EvaluationScores.from_dict(version_dict.get("evaluation_scores", {}))

        # Update known scores
        for name, value in scores.items():
            if hasattr(eval_scores, name):
                setattr(eval_scores, name, value)
            else:
                eval_scores.custom_metrics[name] = value

        version_dict["evaluation_scores"] = eval_scores.to_dict()
        self._save()

    def set_deployed(self, model_name: str, version: str, deployed: bool = True) -> None:
        """Mark a version as deployed or not.

        Args:
            model_name: Name of the model
            version: Version string
            deployed: Whether the version is deployed
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        if version not in self.models[model_name]["versions"]:
            raise ValueError(f"Version not found: {version}")

        version_dict = self.models[model_name]["versions"][version]
        version_dict["deployed"] = deployed
        if deployed:
            version_dict["deployed_at"] = datetime.now(timezone.utc).isoformat()
            version_dict["status"] = VersionStatus.DEPLOYED.value

        self._save()

    def rollback(self, model_name: str, version: str) -> None:
        """Rollback to a previous version (marks previous as deployed).

        Args:
            model_name: Name of the model
            version: Version to rollback to
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        if version not in self.models[model_name]["versions"]:
            raise ValueError(f"Version not found: {version}")

        # Mark any deployed versions as deprecated before deploying target.
        for version_name, version_dict in self.models[model_name]["versions"].items():
            if version_dict.get("deployed") and version_name != version:
                version_dict["deployed"] = False
                version_dict["status"] = VersionStatus.DEPRECATED.value

        # Mark target version as deployed
        self.set_deployed(model_name, version, True)

        self._save()

    def update_metadata(
        self,
        model_name: str,
        description: str | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        """Update model metadata.

        Args:
            model_name: Name of the model
            description: Model description
            owner: Model owner
            tags: Tags for categorization
            notes: Notes about the model
        """
        self._init_model(model_name)

        metadata = self.models[model_name]["metadata"]
        if description is not None:
            metadata["description"] = description
        if owner is not None:
            metadata["owner"] = owner
        if tags is not None:
            metadata["tags"] = tags
        if notes is not None:
            metadata["notes"] = notes

        self._save()

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a version from the registry.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            True if deleted, False if not found
        """
        if model_name not in self.models:
            return False

        if version in self.models[model_name]["versions"]:
            del self.models[model_name]["versions"][version]
            self._save()
            return True

        return False

    def summary(self, model_name: str | None = None) -> str:
        """Generate a summary of models and versions.

        Args:
            model_name: Optional model to summarize. If None, summarize all.

        Returns:
            Formatted summary string
        """
        lines = ["Model Registry Summary", "=" * 60]

        if model_name:
            # Summarize single model
            if model_name not in self.models:
                return f"Model not found: {model_name}"

            lines.append(f"\n{model_name}")
            versions = self.list_versions(model_name)
            lines.append(f"  Versions: {len(versions)}")

            for v in sorted(versions, key=lambda x: x.version):
                status = "●" if v.deployed else "○"
                accuracy = f"accuracy={v.evaluation_scores.accuracy:.3f}" if v.evaluation_scores.accuracy else ""
                lines.append(f"  {status} {v.version} ({v.status.value}) {accuracy}")

        else:
            # Summarize all models
            for model_name in sorted(self.models.keys()):
                versions = self.list_versions(model_name)
                deployed = sum(1 for v in versions if v.deployed)
                lines.append(f"\n{model_name}: {len(versions)} versions ({deployed} deployed)")

                for v in sorted(versions, key=lambda x: x.version)[-3:]:  # Show last 3
                    status = "●" if v.deployed else "○"
                    accuracy = f"acc={v.evaluation_scores.accuracy:.3f}" if v.evaluation_scores.accuracy else ""
                    lines.append(f"  {status} {v.version} {accuracy}")

        return "\n".join(lines)
