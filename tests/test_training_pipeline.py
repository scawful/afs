"""Comprehensive tests for training pipeline.

Tests data generation, quality scoring, rehearsal buffers, and full pipeline execution.
"""

import json

import pytest

from afs.generators.base import TrainingSample
from afs.training.pipeline import DataPipeline, PipelineConfig
from afs.training.rehearsal import RehearsalBuffer, RehearsalBufferConfig


class TestTrainingSample:
    """Test TrainingSample data structure."""

    def test_create_sample(self):
        sample = TrainingSample(
            instruction="Test question",
            output="Test answer",
            thinking="Test reasoning",
            domain="test",
            source="test_source"
        )

        assert sample.instruction == "Test question"
        assert sample.output == "Test answer"
        assert sample.domain == "test"

    def test_to_dict(self):
        sample = TrainingSample(
            instruction="Q",
            output="A",
            thinking="T",
            domain="d",
            source="s"
        )

        d = sample.to_dict()
        assert isinstance(d, dict)
        assert d["instruction"] == "Q"
        assert d["output"] == "A"

    def test_quality_score_default(self):
        sample = TrainingSample(
            instruction="Q",
            output="A",
            thinking="T",
            domain="d",
            source="s"
        )

        # Default quality score should be >= 0
        assert sample.quality_score >= 0.0
        assert sample.quality_score <= 1.0


class TestRehearsalBuffer:
    """Test rehearsal buffer for preventing catastrophic forgetting."""

    def test_create_buffer(self):
        config = RehearsalBufferConfig()
        buffer = RehearsalBuffer(config=config)

        assert buffer.config == config
        assert len(buffer.samples) == 0

    def test_load_from_jsonl(self, temp_dir):
        # Create test JSONL
        jsonl_path = temp_dir / "test.jsonl"
        samples = [
            {"instruction": "Q1", "output": "A1", "thinking": "T1", "domain": "d", "source": "s", "quality_score": 0.8},
            {"instruction": "Q2", "output": "A2", "thinking": "T2", "domain": "d", "source": "s", "quality_score": 0.6},
        ]

        with open(jsonl_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')

        # Load into buffer
        buffer = RehearsalBuffer()
        loaded = buffer.load_from_jsonl(jsonl_path)

        assert loaded == 2
        assert len(buffer.samples) == 2

    def test_select_top_samples(self):
        buffer = RehearsalBuffer(
            config=RehearsalBufferConfig(top_ratio=0.5, quality_threshold=0.0)
        )

        # Add samples with varying quality
        for i in range(10):
            buffer.samples.append(TrainingSample(
                instruction=f"Q{i}",
                output=f"A{i}",
                thinking="T",
                domain="d",
                source="s",
                quality_score=i / 10.0
            ))

        # Select top 50%
        retained = buffer.select_top_samples()

        assert retained == 5
        assert len(buffer.samples) == 5
        # Check that highest quality samples were kept
        assert all(s.quality_score >= 0.5 for s in buffer.samples)

    def test_merge_with_new_data(self):
        # Create buffer with old samples
        buffer = RehearsalBuffer()
        for i in range(5):
            buffer.samples.append(TrainingSample(
                instruction=f"Old{i}",
                output=f"A{i}",
                thinking="T",
                domain="d",
                source="old",
                quality_score=0.8
            ))

        # Create new samples
        new_samples = [
            TrainingSample(
                instruction=f"New{i}",
                output=f"A{i}",
                thinking="T",
                domain="d",
                source="new",
                quality_score=0.7
            )
            for i in range(10)
        ]

        # Merge with 30% rehearsal ratio
        merged = buffer.merge_with_new_data(new_samples, rehearsal_ratio=0.3, shuffle=False)

        # Should have new samples + some rehearsal samples
        assert len(merged) > len(new_samples)
        # Check that both old and new samples are present
        assert any(s.source == "old" for s in merged)
        assert any(s.source == "new" for s in merged)

    def test_save_and_load(self, temp_dir):
        # Create buffer
        buffer1 = RehearsalBuffer()
        buffer1.samples = [
            TrainingSample(
                instruction="Q",
                output="A",
                thinking="T",
                domain="d",
                source="s",
                quality_score=0.9
            )
        ]

        # Save
        save_path = temp_dir / "buffer.jsonl"
        saved = buffer1.save(save_path)
        assert saved == 1

        # Load into new buffer
        buffer2 = RehearsalBuffer()
        loaded = buffer2.load_from_jsonl(save_path)
        assert loaded == 1
        assert buffer2.samples[0].instruction == "Q"


class TestTrainingPipeline:
    """Test full training pipeline."""

    @pytest.mark.slow
    def test_pipeline_basic_flow(self, temp_dir, sample_jsonl_file):
        # Create basic pipeline config
        config = PipelineConfig(
            input_paths=[sample_jsonl_file],
            output_dir=temp_dir,
            expand_vocab=False,
            extract_entities=False,
            score_quality=False,
            apply_phase1_augment=False,
            apply_phase2_augment=False,
            deduplicate=False,
            split_data=False,
        )

        # Run pipeline
        pipeline = DataPipeline(config)
        result = pipeline.run()

        assert result.input_count > 0
        assert result.output_count > 0

    @pytest.mark.slow
    def test_pipeline_with_quality_filtering(self, temp_dir, sample_jsonl_file):
        config = PipelineConfig(
            input_paths=[sample_jsonl_file],
            output_dir=temp_dir,
            min_quality_score=0.7,
            expand_vocab=False,
            extract_entities=False,
            apply_phase1_augment=False,
            apply_phase2_augment=False,
            deduplicate=False,
            split_data=False,
        )

        pipeline = DataPipeline(config)
        result = pipeline.run()

        # Should filter some samples
        assert result.output_count <= result.input_count


@pytest.mark.integration
class TestIntegration:
    """Integration tests for full workflows."""

    @pytest.mark.slow
    def test_full_training_workflow(self, temp_dir):
        """Test complete workflow: generate → process → train-ready."""

        # 1. Generate samples
        samples = [
            TrainingSample(
                instruction=f"Question {i}",
                output=f"Answer {i}",
                thinking="Reasoning",
                domain="test",
                source="integration_test",
                quality_score=0.5 + (i * 0.1)
            )
            for i in range(10)
        ]

        # 2. Save to JSONL
        raw_path = temp_dir / "raw.jsonl"
        with open(raw_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + '\n')

        # 3. Process through pipeline
        output_dir = temp_dir / "processed"
        config = PipelineConfig(
            input_paths=[raw_path],
            output_dir=output_dir,
            min_quality_score=0.0,
            expand_vocab=False,
            extract_entities=False,
            score_quality=False,
            apply_phase1_augment=False,
            apply_phase2_augment=False,
            deduplicate=False,
        )

        pipeline = DataPipeline(config)
        result = pipeline.run()

        # 4. Verify outputs
        assert result.output_count > 0
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()

    @pytest.mark.slow
    def test_rehearsal_workflow(self, temp_dir):
        """Test rehearsal buffer workflow."""

        # 1. Create v1 data
        v1_samples = [
            TrainingSample(
                instruction=f"V1 Q{i}",
                output=f"V1 A{i}",
                thinking="T",
                domain="d",
                source="v1",
                quality_score=0.9
            )
            for i in range(20)
        ]

        v1_path = temp_dir / "v1.jsonl"
        with open(v1_path, 'w') as f:
            for s in v1_samples:
                f.write(json.dumps(s.to_dict()) + '\n')

        # 2. Build rehearsal buffer
        buffer = RehearsalBuffer(config=RehearsalBufferConfig(top_ratio=0.3))
        buffer.load_from_jsonl(v1_path, version="v1")
        buffer.select_top_samples()

        rehearsal_path = temp_dir / "rehearsal.jsonl"
        buffer.save(rehearsal_path)

        # 3. Create v2 data
        v2_samples = [
            TrainingSample(
                instruction=f"V2 Q{i}",
                output=f"V2 A{i}",
                thinking="T",
                domain="d",
                source="v2",
                quality_score=0.8
            )
            for i in range(30)
        ]

        # 4. Merge with rehearsal
        merged = buffer.merge_with_new_data(v2_samples, rehearsal_ratio=0.3)

        # 5. Verify mix
        assert len(merged) > len(v2_samples)
        v1_count = sum(1 for s in merged if s.source == "v1")
        v2_count = sum(1 for s in merged if s.source == "v2")
        assert v1_count > 0
        assert v2_count > 0


# Property-based tests
@pytest.mark.slow
class TestProperties:
    """Property-based tests for invariants."""

    def test_pipeline_output_smaller_or_equal(self, temp_dir, sample_jsonl_file):
        """Pipeline should never increase sample count (only filter/dedupe)."""

        config = PipelineConfig(
            input_paths=[sample_jsonl_file],
            output_dir=temp_dir,
            min_quality_score=0.0,
            expand_vocab=False,
            extract_entities=False,
            score_quality=False,
            apply_phase1_augment=False,  # No augmentation to ensure this property
            apply_phase2_augment=False,
            deduplicate=False,
        )

        pipeline = DataPipeline(config)
        result = pipeline.run()

        assert result.output_count <= result.input_count

    def test_quality_scores_bounded(self):
        """Quality scores should always be in [0, 1]."""

        for _ in range(100):
            sample = TrainingSample(
                instruction="Q",
                output="A",
                thinking="T",
                domain="d",
                source="s"
            )

            assert 0.0 <= sample.quality_score <= 1.0
