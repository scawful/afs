"""Tests for model registry system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from afs.registry import (
    EvaluationScores,
    LineageTracker,
    ModelRegistry,
    ModelVersion,
    TrainingMetadata,
    VersionStatus,
)


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_create_version(self) -> None:
        """Test creating a model version."""
        v = ModelVersion(
            model_name="test",
            version="v1",
            lora_path="/path/to/lora",
        )

        assert v.model_name == "test"
        assert v.version == "v1"
        assert v.lora_path == "/path/to/lora"
        assert v.status == VersionStatus.DRAFT

    def test_version_to_dict(self) -> None:
        """Test converting version to dict."""
        v = ModelVersion(
            model_name="test",
            version="v1",
            tags=["test"],
        )

        d = v.to_dict()
        assert d["model_name"] == "test"
        assert d["version"] == "v1"
        assert d["tags"] == ["test"]

    def test_version_from_dict(self) -> None:
        """Test creating version from dict."""
        data = {
            "model_name": "test",
            "version": "v1",
            "status": "draft",
            "evaluation_scores": {"accuracy": 0.85},
        }

        v = ModelVersion.from_dict(data)
        assert v.model_name == "test"
        assert v.version == "v1"
        assert v.evaluation_scores.accuracy == 0.85


class TestEvaluationScores:
    """Tests for EvaluationScores."""

    def test_create_scores(self) -> None:
        """Test creating evaluation scores."""
        scores = EvaluationScores(
            accuracy=0.85,
            f1_score=0.82,
        )

        assert scores.accuracy == 0.85
        assert scores.f1_score == 0.82

    def test_scores_to_dict(self) -> None:
        """Test converting scores to dict."""
        scores = EvaluationScores(accuracy=0.85)
        d = scores.to_dict()
        assert d["accuracy"] == 0.85

    def test_scores_from_dict(self) -> None:
        """Test creating scores from dict."""
        data = {"accuracy": 0.85, "custom": 0.9}
        scores = EvaluationScores.from_dict(data)
        assert scores.accuracy == 0.85
        assert scores.custom_metrics["custom"] == 0.9


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def temp_registry(self) -> Path:
        """Create temporary registry path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "registry.json"

    @pytest.fixture
    def registry(self, temp_registry: Path) -> ModelRegistry:
        """Create registry with temporary path."""
        return ModelRegistry(temp_registry)

    def test_register_model(self, registry: ModelRegistry) -> None:
        """Test registering a model version."""
        v = registry.register_model(
            model_name="test",
            base_model="base",
            samples=100,
            epochs=1,
        )

        assert v.model_name == "test"
        assert v.version == "v1"
        assert v.training.samples == 100

    def test_auto_increment_version(self, registry: ModelRegistry) -> None:
        """Test automatic version incrementing."""
        registry.register_model(
            model_name="test",
            base_model="base",
            samples=100,
            epochs=1,
        )

        v2 = registry.register_model(
            model_name="test",
            base_model="base",
            samples=100,
            epochs=1,
        )

        assert v2.version == "v2"

    def test_explicit_version(self, registry: ModelRegistry) -> None:
        """Test registering with explicit version."""
        v = registry.register_model(
            model_name="test",
            version="v2.1",
            base_model="base",
        )

        assert v.version == "v2.1"

    def test_get_model(self, registry: ModelRegistry) -> None:
        """Test getting model metadata."""
        registry.register_model(
            model_name="test",
            base_model="base",
        )

        model = registry.get_model("test")
        assert model is not None
        assert model.model_name == "test"

    def test_get_version(self, registry: ModelRegistry) -> None:
        """Test getting specific version."""
        registry.register_model(
            model_name="test",
            base_model="base",
        )

        v = registry.get_version("test", "v1")
        assert v is not None
        assert v.version == "v1"

    def test_get_latest(self, registry: ModelRegistry) -> None:
        """Test getting latest version."""
        registry.register_model(model_name="test", base_model="base")
        registry.register_model(model_name="test", base_model="base")

        latest = registry.get_latest("test")
        assert latest is not None
        assert latest.version == "v2"

    def test_list_versions(self, registry: ModelRegistry) -> None:
        """Test listing versions."""
        registry.register_model(model_name="test", base_model="base")
        registry.register_model(model_name="test", base_model="base")

        versions = registry.list_versions("test")
        assert len(versions) == 2
        assert versions[0].version == "v1"
        assert versions[1].version == "v2"

    def test_list_models(self, registry: ModelRegistry) -> None:
        """Test listing all models."""
        registry.register_model(model_name="model1", base_model="base")
        registry.register_model(model_name="model2", base_model="base")

        models = registry.list_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_compare_versions(self, registry: ModelRegistry) -> None:
        """Test comparing versions."""
        registry.register_model(
            model_name="test",
            base_model="base1",
            evaluation_scores={"accuracy": 0.80},
        )

        registry.register_model(
            model_name="test",
            base_model="base2",
            evaluation_scores={"accuracy": 0.85},
        )

        diff = registry.compare_versions("test", "v1", "v2")
        assert "accuracy" in diff
        assert diff["accuracy"]["v1"] == 0.80
        assert diff["accuracy"]["v2"] == 0.85

    def test_update_scores(self, registry: ModelRegistry) -> None:
        """Test updating evaluation scores."""
        registry.register_model(model_name="test", base_model="base")
        registry.update_evaluation_scores("test", "v1", accuracy=0.90)

        v = registry.get_version("test", "v1")
        assert v is not None
        assert v.evaluation_scores.accuracy == 0.90

    def test_set_deployed(self, registry: ModelRegistry) -> None:
        """Test marking version as deployed."""
        registry.register_model(model_name="test", base_model="base")
        registry.set_deployed("test", "v1", deployed=True)

        v = registry.get_version("test", "v1")
        assert v is not None
        assert v.deployed is True

    def test_rollback(self, registry: ModelRegistry) -> None:
        """Test rollback functionality."""
        registry.register_model(model_name="test", base_model="base")
        registry.set_deployed("test", "v1", deployed=True)

        registry.register_model(model_name="test", base_model="base")
        registry.rollback("test", "v2")

        v1 = registry.get_version("test", "v1")
        v2 = registry.get_version("test", "v2")

        assert v1 is not None
        assert v1.deployed is False
        assert v2 is not None
        assert v2.deployed is True

    def test_delete_version(self, registry: ModelRegistry) -> None:
        """Test deleting a version."""
        registry.register_model(model_name="test", base_model="base")
        registry.register_model(model_name="test", base_model="base")

        deleted = registry.delete_version("test", "v1")
        assert deleted is True

        v = registry.get_version("test", "v1")
        assert v is None

    def test_persistence(self, temp_registry: Path) -> None:
        """Test that registry persists to disk."""
        reg1 = ModelRegistry(temp_registry)
        reg1.register_model(model_name="test", base_model="base")

        # Create new registry pointing to same path
        reg2 = ModelRegistry(temp_registry)
        versions = reg2.list_versions("test")
        assert len(versions) == 1


class TestLineageTracker:
    """Tests for LineageTracker."""

    @pytest.fixture
    def temp_lineage(self) -> Path:
        """Create temporary lineage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "lineage.json"

    @pytest.fixture
    def tracker(self, temp_lineage: Path) -> LineageTracker:
        """Create tracker with temporary path."""
        return LineageTracker(temp_lineage)

    def test_add_version(self, tracker: LineageTracker) -> None:
        """Test adding version to lineage."""
        tracker.add_version(
            model_name="test",
            version="v1",
            base_model="base",
        )

        lineage = tracker.get_lineage("test", "v1")
        assert lineage["base_model"] == "base"

    def test_get_ancestors(self, tracker: LineageTracker) -> None:
        """Test getting ancestor versions."""
        tracker.add_version(model_name="test", version="v1")
        tracker.add_version(model_name="test", version="v2", parent_version="v1")
        tracker.add_version(model_name="test", version="v3", parent_version="v2")

        ancestors = tracker.get_ancestors("test", "v3")
        assert ancestors == ["v2", "v1"]

    def test_get_descendants(self, tracker: LineageTracker) -> None:
        """Test getting descendant versions."""
        tracker.add_version(model_name="test", version="v1")
        tracker.add_version(model_name="test", version="v2", parent_version="v1")
        tracker.add_version(model_name="test", version="v3", parent_version="v1")

        descendants = tracker.get_descendants("test", "v1")
        assert "v2" in descendants
        assert "v3" in descendants

    def test_get_data_lineage(self, tracker: LineageTracker) -> None:
        """Test getting complete data lineage."""
        tracker.add_version(
            model_name="test",
            version="v1",
            training_data=["data1"],
        )

        tracker.add_version(
            model_name="test",
            version="v2",
            parent_version="v1",
            training_data=["data2"],
        )

        lineage = tracker.get_data_lineage("test", "v2")
        assert lineage["v1"] == ["data1"]
        assert lineage["v2"] == ["data2"]

    def test_build_tree(self, tracker: LineageTracker) -> None:
        """Test building version tree."""
        tracker.add_version(model_name="test", version="v1")
        tracker.add_version(model_name="test", version="v2", parent_version="v1")

        tree = tracker.build_tree("test")
        assert "v1" in tree
        assert "v2" in tree

    def test_get_training_history(self, tracker: LineageTracker) -> None:
        """Test getting training history."""
        tracker.add_version(model_name="test", version="v1")
        tracker.add_version(model_name="test", version="v2")

        history = tracker.get_training_history("test")
        assert len(history) == 2
        assert history[0]["version"] == "v1"
        assert history[1]["version"] == "v2"

    def test_persistence(self, temp_lineage: Path) -> None:
        """Test that lineage persists to disk."""
        tracker1 = LineageTracker(temp_lineage)
        tracker1.add_version(model_name="test", version="v1")

        tracker2 = LineageTracker(temp_lineage)
        history = tracker2.get_training_history("test")
        assert len(history) == 1


class TestIntegration:
    """Integration tests for registry and lineage."""

    @pytest.fixture
    def setup(self) -> tuple[Path, Path]:
        """Setup temp directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            registry_path = tmpdir / "registry.json"
            lineage_path = tmpdir / "lineage.json"
            yield registry_path, lineage_path

    def test_full_workflow(self, setup: tuple[Path, Path]) -> None:
        """Test complete workflow."""
        registry_path, lineage_path = setup
        registry = ModelRegistry(registry_path)
        tracker = LineageTracker(lineage_path)

        # Register v1
        v1 = registry.register_model(
            model_name="oracle",
            base_model="qwen",
            samples=100,
            evaluation_scores={"accuracy": 0.80},
        )

        tracker.add_version(
            model_name="oracle",
            version="v1",
            base_model="qwen",
            training_data=["data1"],
        )

        # Deploy v1
        registry.set_deployed("oracle", "v1", deployed=True)

        # Register v2 (fine-tune)
        v2 = registry.register_model(
            model_name="oracle",
            base_model="oracle:v1",
            samples=50,
            parent_version="v1",
            evaluation_scores={"accuracy": 0.82},
        )

        tracker.add_version(
            model_name="oracle",
            version="v2",
            parent_version="v1",
            training_data=["data1", "data2"],
        )

        # Rollback to v2
        registry.rollback("oracle", "v2")

        # Verify
        assert registry.get_latest("oracle").version == "v2"
        assert registry.get_latest("oracle").deployed is True
        assert registry.get_version("oracle", "v1").deployed is False

        ancestors = tracker.get_ancestors("oracle", "v2")
        assert "v1" in ancestors
