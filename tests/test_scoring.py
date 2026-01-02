"""Tests for training/scoring.py quality scoring module."""

from __future__ import annotations

import pytest

from afs.training.scoring import (
    QualityScore,
    QualityScorer,
    ScoringConfig,
    ScoringWeights,
    analyze_scores,
)
from afs.generators.base import TrainingSample


class TestScoringWeights:
    """Tests for ScoringWeights dataclass."""

    def test_default_weights_sum_to_one(self) -> None:
        weights = ScoringWeights()
        total = weights.electra + weights.asar + weights.entity + weights.length
        assert total == pytest.approx(1.0)

    def test_normalize_returns_unit_sum(self) -> None:
        weights = ScoringWeights(electra=2.0, asar=2.0, entity=2.0, length=2.0)
        normalized = weights.normalize()
        total = normalized.electra + normalized.asar + normalized.entity + normalized.length
        assert total == pytest.approx(1.0)
        assert normalized.electra == pytest.approx(0.25)

    def test_normalize_handles_zero_weights(self) -> None:
        weights = ScoringWeights(electra=0, asar=0, entity=0, length=0)
        normalized = weights.normalize()
        # Should return default equal weights
        assert normalized.electra == pytest.approx(0.25)
        assert normalized.asar == pytest.approx(0.25)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = ScoringWeights(electra=0.5, asar=0.2, entity=0.2, length=0.1)
        data = original.to_dict()
        restored = ScoringWeights.from_dict(data)
        assert restored.electra == original.electra
        assert restored.asar == original.asar
        assert restored.entity == original.entity
        assert restored.length == original.length


class TestQualityScorer:
    """Tests for QualityScorer class."""

    @pytest.fixture
    def simple_sample(self) -> TrainingSample:
        """A simple valid assembly sample."""
        return TrainingSample(
            instruction="Write a routine to load health",
            output="""; Load Link's health
LDA $7EF36C   ; Current health
STA $00       ; Store in scratch
RTS
""",
            domain="asm",
        )

    @pytest.fixture
    def short_sample(self) -> TrainingSample:
        """A very short sample that should get low length score."""
        return TrainingSample(
            instruction="NOP",
            output="NOP",
            domain="asm",
        )

    @pytest.fixture
    def scorer(self) -> QualityScorer:
        """Scorer without ELECTRA model (uses default 0.5)."""
        config = ScoringConfig(
            electra_model_path=None,
            skip_asar_if_no_model=True,  # Skip asar validation in tests
        )
        return QualityScorer(config=config)

    def test_score_returns_quality_score(self, scorer, simple_sample) -> None:
        result = scorer.score(simple_sample)
        assert isinstance(result, QualityScore)
        assert 0.0 <= result.overall <= 1.0
        assert 0.0 <= result.electra_score <= 1.0
        assert 0.0 <= result.entity_coverage <= 1.0
        assert 0.0 <= result.length_score <= 1.0

    def test_score_populates_components(self, scorer, simple_sample) -> None:
        result = scorer.score(simple_sample)
        assert "electra" in result.components
        assert "asar" in result.components
        assert "entity" in result.components
        assert "length" in result.components

    def test_score_extracts_entities(self, scorer, simple_sample) -> None:
        result = scorer.score(simple_sample)
        # $7EF36C is a known ALTTP address (link_health)
        assert result.entity_count > 0
        assert result.known_entity_count > 0

    def test_short_sample_gets_low_length_score(self, scorer, short_sample) -> None:
        result = scorer.score(short_sample)
        # Very short output should have low length score
        assert result.length_score < 0.5

    def test_score_batch_updates_samples(self, scorer, simple_sample) -> None:
        samples = [simple_sample]
        scores = scorer.score_batch(samples, update_samples=True)
        assert len(scores) == 1
        # Check that sample.quality_score was updated
        assert simple_sample.quality_score > 0.0
        assert "_metadata" in dir(simple_sample)
        assert "quality_components" in simple_sample._metadata

    def test_score_batch_without_update(self, scorer, simple_sample) -> None:
        sample = TrainingSample(
            instruction="Test",
            output="LDA $00\nSTA $01\nRTS",
            domain="asm",
        )
        original_score = sample.quality_score
        scorer.score_batch([sample], update_samples=False)
        # Score should not be updated
        assert sample.quality_score == original_score


class TestLengthScoring:
    """Tests for length-based quality scoring."""

    @pytest.fixture
    def scorer(self) -> QualityScorer:
        config = ScoringConfig(
            min_output_length=50,
            ideal_output_length=500,
            max_output_length=5000,
            skip_asar_if_no_model=True,
        )
        return QualityScorer(config=config)

    def test_too_short_gets_low_score(self, scorer) -> None:
        score = scorer._compute_length_score("x" * 10)  # Way below min
        assert score < 0.5

    def test_minimum_length_gets_half_score(self, scorer) -> None:
        score = scorer._compute_length_score("x" * 50)
        assert score == pytest.approx(0.5, abs=0.1)

    def test_ideal_length_gets_full_score(self, scorer) -> None:
        score = scorer._compute_length_score("x" * 500)
        assert score == pytest.approx(1.0, abs=0.1)

    def test_too_long_gets_reduced_score(self, scorer) -> None:
        score = scorer._compute_length_score("x" * 6000)  # Beyond max
        assert score == pytest.approx(0.5, abs=0.1)


class TestAnalyzeScores:
    """Tests for analyze_scores function."""

    def test_analyze_empty_list(self) -> None:
        result = analyze_scores([])
        assert result["count"] == 0

    def test_analyze_returns_statistics(self) -> None:
        scores = [
            QualityScore(
                overall=0.8,
                electra_score=0.9,
                asar_valid=True,
                entity_coverage=0.7,
                length_score=0.8,
            ),
            QualityScore(
                overall=0.6,
                electra_score=0.5,
                asar_valid=False,
                entity_coverage=0.5,
                length_score=0.8,
            ),
        ]
        result = analyze_scores(scores)

        assert result["count"] == 2
        assert result["overall"]["mean"] == pytest.approx(0.7)
        assert result["overall"]["min"] == 0.6
        assert result["overall"]["max"] == 0.8
        assert result["asar_pass_rate"] == pytest.approx(0.5)

    def test_analyze_includes_histogram(self) -> None:
        scores = [
            QualityScore(
                overall=0.85,
                electra_score=0.9,
                asar_valid=True,
                entity_coverage=0.7,
                length_score=0.8,
            ),
        ]
        result = analyze_scores(scores)
        assert "histogram" in result["overall"]
        # Score 0.85 should be in 0.8-0.9 bucket
        assert result["overall"]["histogram"]["0.8-0.9"] == 1
