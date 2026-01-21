"""Tests for continuous learning system."""

from pathlib import Path

import pytest

from afs.continuous import (
    ABTestManager,
    ContinuousLearningLoop,
    DataGeneratorConfig,
    LoopConfig,
    ModelStatus,
    ModelVersion,
    RetrainTrigger,
    TrainingDataGenerator,
    TriggerConfig,
    TriggerType,
    UsageLogger,
)


class TestUsageLogger:
    """Test usage logging."""

    def test_log_usage(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        record_id = logger.log(
            query="test query",
            response="test response",
            model="test-model",
            quality_score=0.8,
        )

        assert len(record_id) == 12  # MD5 hash truncated to 12 chars

        # Verify record exists
        records = list(logger.get_records())
        assert len(records) == 1
        assert records[0].query == "test query"

    def test_record_feedback(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        record_id = logger.log(
            query="test", response="test", model="test", quality_score=0.8
        )

        success = logger.record_feedback(record_id, feedback=1, feedback_text="good")
        assert success

        # Verify feedback
        records = list(logger.get_records(with_feedback_only=True))
        assert len(records) == 1
        assert records[0].user_feedback == 1

    def test_statistics(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        # Log some records
        for i in range(10):
            record_id = logger.log(
                query=f"query {i}",
                response=f"response {i}",
                model="test",
                quality_score=0.7 + i * 0.02,
            )
            if i < 3:
                logger.record_feedback(record_id, feedback=1 if i % 2 == 0 else -1)

        stats = logger.get_statistics()
        assert stats["total"] == 10
        assert stats["with_feedback"] == 3
        assert stats["positive_feedback"] == 2
        assert stats["negative_feedback"] == 1


class TestTrainingDataGenerator:
    """Test training data generation."""

    def test_generate_data(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        # Log some records
        for i in range(20):
            record_id = logger.log(
                query=f"query {i}",
                response=f"response {i}",
                model="test",
                quality_score=0.7 + i * 0.01,
            )
            if i % 3 == 0:
                logger.record_feedback(record_id, feedback=1)

        # Generate data
        config = DataGeneratorConfig(
            min_quality_score=0.75,
            min_user_feedback=1,
            deduplicate=True,
        )
        generator = TrainingDataGenerator(logger, config)
        result = generator.generate(tmp_path / "training.jsonl")

        assert result.total_candidates == 20
        assert result.final_count > 0
        assert result.output_path.exists()

    def test_deduplication(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        # Log duplicates
        for _ in range(3):
            logger.log(
                query="same query", response="same response", model="test", quality_score=0.8
            )

        config = DataGeneratorConfig(
            min_quality_score=0.5, deduplicate=True, min_user_feedback=None
        )
        generator = TrainingDataGenerator(logger, config)
        result = generator.generate(tmp_path / "training.jsonl")

        assert result.total_candidates == 3
        assert result.duplicates_removed == 2
        assert result.final_count == 1


class TestRetrainTrigger:
    """Test retraining triggers."""

    def test_sample_count_trigger(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        config = TriggerConfig(
            enable_sample_count=True,
            enable_scheduled=False,  # Disable scheduled for this test
            min_new_samples=10,
            cooldown_hours=0,
        )
        trigger = RetrainTrigger(logger, config)

        # Below threshold
        for i in range(5):
            logger.log(query=f"q{i}", response=f"r{i}", model="test", quality_score=0.8)

        result = trigger.check_triggers()
        assert not result.triggered

        # Reach threshold
        for i in range(5, 15):
            logger.log(query=f"q{i}", response=f"r{i}", model="test", quality_score=0.8)

        result = trigger.check_triggers()
        assert result.triggered
        assert result.trigger_type == TriggerType.SAMPLE_COUNT

    def test_cooldown(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")

        config = TriggerConfig(
            enable_sample_count=True,
            min_new_samples=5,
            cooldown_hours=1,
        )
        trigger = RetrainTrigger(logger, config)

        # Log samples
        for i in range(10):
            logger.log(query=f"q{i}", response=f"r{i}", model="test", quality_score=0.8)

        # First trigger should work
        result = trigger.check_triggers()
        assert result.triggered

        # Mark completed
        trigger.mark_retrain_completed()

        # Second trigger should be blocked by cooldown
        result = trigger.check_triggers()
        assert not result.triggered
        assert "cooldown" in result.reason.lower()


class TestABTestManager:
    """Test A/B testing."""

    def test_deploy_challenger(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")
        ab_test = ABTestManager(logger, state_file=tmp_path / "ab_state.json")

        # Set champion
        ab_test.champion = ModelVersion(
            id="champ", name="v1", path=Path("/tmp/v1"), status=ModelStatus.CHAMPION
        )

        # Deploy challenger
        challenger = ab_test.deploy_challenger("v2", Path("/tmp/v2"), traffic_weight=0.1)

        assert challenger.status == ModelStatus.CHALLENGER
        assert ab_test.traffic_split.challenger_weight == 0.1
        assert ab_test.traffic_split.champion_weight == 0.9

    def test_route_request(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")
        ab_test = ABTestManager(logger)

        # Setup models
        ab_test.champion = ModelVersion(
            id="champ", name="v1", path=Path("/tmp/v1"), status=ModelStatus.CHAMPION
        )
        ab_test.challenger = ModelVersion(
            id="chal", name="v2", path=Path("/tmp/v2"), status=ModelStatus.CHALLENGER
        )
        ab_test.traffic_split.champion_weight = 0.9
        ab_test.traffic_split.challenger_weight = 0.1

        # Route many requests
        routes = {"champion": 0, "challenger": 0}
        for _ in range(1000):
            routed = ab_test.route_request()
            if routed == ab_test.champion:
                routes["champion"] += 1
            else:
                routes["challenger"] += 1

        # Should be roughly 90/10 split
        champ_ratio = routes["champion"] / 1000
        assert 0.85 < champ_ratio < 0.95

    def test_promote_challenger(self, tmp_path):
        logger = UsageLogger(tmp_path / "test.db")
        ab_test = ABTestManager(logger)

        ab_test.champion = ModelVersion(
            id="champ", name="v1", path=Path("/tmp/v1"), status=ModelStatus.CHAMPION
        )
        ab_test.challenger = ModelVersion(
            id="chal", name="v2", path=Path("/tmp/v2"), status=ModelStatus.CHALLENGER
        )

        success = ab_test.promote_challenger()
        assert success
        assert ab_test.champion.name == "v2"
        assert ab_test.champion.status == ModelStatus.CHAMPION
        assert ab_test.challenger is None


class TestContinuousLearningLoop:
    """Test main continuous learning loop."""

    def test_loop_initialization(self, tmp_path):
        config = LoopConfig(
            db_path=tmp_path / "test.db",
            output_dir=tmp_path / "output",
        )
        loop = ContinuousLearningLoop(config)

        assert loop.usage_logger is not None
        assert loop.auto_retrainer is not None
        assert loop.status.total_retrains == 0

    def test_log_usage_and_feedback(self, tmp_path):
        config = LoopConfig(db_path=tmp_path / "test.db", output_dir=tmp_path / "output")
        loop = ContinuousLearningLoop(config)

        record_id = loop.log_usage(
            query="test query",
            response="test response",
            model="test",
            quality_score=0.8,
        )

        success = loop.record_feedback(record_id, feedback=1)
        assert success

        stats = loop.get_statistics()
        assert stats["total"] == 1
        assert stats["with_feedback"] == 1

    def test_iteration_without_trigger(self, tmp_path):
        config = LoopConfig(
            db_path=tmp_path / "test.db",
            output_dir=tmp_path / "output",
            trigger_config=TriggerConfig(min_new_samples=1000),  # High threshold
        )
        loop = ContinuousLearningLoop(config)

        # Log a few samples (below threshold)
        for i in range(5):
            loop.log_usage(f"q{i}", f"r{i}", "test", quality_score=0.8)

        result = loop.run_iteration()
        assert not result["retrain_triggered"]

    def test_get_statistics(self, tmp_path):
        config = LoopConfig(db_path=tmp_path / "test.db", output_dir=tmp_path / "output")
        loop = ContinuousLearningLoop(config)

        for i in range(10):
            loop.log_usage(f"q{i}", f"r{i}", "test", quality_score=0.8)

        stats = loop.get_statistics()
        assert stats["total"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
