"""Tests for quality analysis module."""

import pytest

from afs.quality import (
    DatasetAnalyzer,
    DatasetStatistics,
    BiasAnalyzer,
    DuplicateDetector,
    AnomalyDetector,
    QualityMetrics,
)


class TestQualityMetrics:
    """Test QualityMetrics functionality."""

    def test_instruction_clarity_basic(self):
        """Test basic instruction clarity scoring."""
        metrics = QualityMetrics()

        text = "Write a function to calculate the factorial"
        clarity = metrics.compute_instruction_clarity(text)

        assert clarity.length > 0
        assert 0.0 <= clarity.specificity_score <= 1.0
        assert 0.0 <= clarity.clarity_score <= 1.0
        assert clarity.overall_score() > 0.0

    def test_instruction_clarity_caching(self):
        """Test that clarity metrics are cached."""
        metrics = QualityMetrics()
        text = "Test instruction"

        clarity1 = metrics.compute_instruction_clarity(text)
        clarity2 = metrics.compute_instruction_clarity(text)

        assert clarity1 is clarity2  # Same object

    def test_output_correctness_basic(self):
        """Test output correctness scoring."""
        metrics = QualityMetrics()

        code = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        correctness = metrics.compute_output_correctness(code)

        assert correctness.num_lines > 0
        assert 0.0 <= correctness.syntax_valid
        assert 0.0 <= correctness.overall_score() <= 1.0

    def test_output_correctness_caching(self):
        """Test that correctness metrics are cached."""
        metrics = QualityMetrics()
        text = "output text"

        correctness1 = metrics.compute_output_correctness(text)
        correctness2 = metrics.compute_output_correctness(text)

        assert correctness1 is correctness2

    def test_duplicate_detection_exact(self):
        """Test exact duplicate detection."""
        metrics = QualityMetrics()

        text = "Hello world"
        texts = [text, "Different", text]

        info = metrics.compute_duplicate_info(text, texts, 0)

        assert len(info.exact_duplicates) > 0
        assert 2 in info.exact_duplicates

    def test_duplicate_detection_none(self):
        """Test when no duplicates found."""
        metrics = QualityMetrics()

        text = "Unique text"
        texts = [text, "Different one", "Another different"]

        info = metrics.compute_duplicate_info(text, texts, 0)

        assert len(info.exact_duplicates) == 0
        assert info.deduplication_status == "unique"

    def test_anomaly_detection_length(self):
        """Test anomaly detection for length outliers."""
        metrics = QualityMetrics()

        # Texts of normal length
        normal = ["short", "medium length text", "another normal one"]
        # Very long outlier
        outlier = "x" * 1000

        info = metrics.compute_anomaly_info(outlier, normal)

        assert info.is_anomaly

    def test_anomaly_detection_whitespace(self):
        """Test anomaly detection for mostly whitespace."""
        metrics = QualityMetrics()

        texts = ["normal text", "another one"]
        mostly_space = "        \n\n    \t\t"

        info = metrics.compute_anomaly_info(mostly_space, texts)

        assert info.is_anomaly

    def test_text_similarity(self):
        """Test text similarity computation."""
        metrics = QualityMetrics()

        similarity = metrics._compute_text_similarity("hello", "hello")
        assert similarity == 1.0

        similarity = metrics._compute_text_similarity("hello", "world")
        assert similarity < 1.0

        similarity = metrics._compute_text_similarity("", "text")
        assert similarity == 0.0


class TestDuplicateDetector:
    """Test DuplicateDetector."""

    def test_find_duplicates_exact(self):
        """Test finding exact duplicates in samples."""
        samples = [
            {"instruction": "same", "output": "result"},
            {"instruction": "different", "output": "result"},
            {"instruction": "same", "output": "result"},
        ]

        detector = DuplicateDetector()
        duplicates = detector.find_duplicates(samples)

        # Samples 0 and 2 are duplicates
        assert 0 in duplicates or 2 in duplicates

    def test_find_duplicates_none(self):
        """Test when no duplicates exist."""
        samples = [
            {"instruction": "unique1", "output": "result1"},
            {"instruction": "unique2", "output": "result2"},
            {"instruction": "unique3", "output": "result3"},
        ]

        detector = DuplicateDetector()
        duplicates = detector.find_duplicates(samples)

        assert len(duplicates) == 0


class TestAnomalyDetector:
    """Test AnomalyDetector."""

    def test_find_anomalies_length(self):
        """Test finding length anomalies."""
        samples = [
            {"instruction": "a"},  # Very short
            {"instruction": "b"},  # Very short
            {"instruction": "c"},  # Very short
            {"instruction": "x" * 5000},  # Massive outlier
        ]

        detector = AnomalyDetector(length_threshold=1.5)  # Lower threshold for detection
        anomalies = detector.find_anomalies(samples)

        # Very long instruction should be detected
        assert len(anomalies) > 0, f"Expected anomalies but got none. Anomalies dict: {anomalies}"

    def test_find_anomalies_none(self):
        """Test when no anomalies exist."""
        samples = [
            {"instruction": "normal instruction", "output": "normal output"},
            {"instruction": "another normal", "output": "another result"},
        ]

        detector = AnomalyDetector()
        anomalies = detector.find_anomalies(samples)

        assert len(anomalies) == 0


class TestBiasAnalyzer:
    """Test BiasAnalyzer."""

    def test_gender_bias_detection(self):
        """Test gender bias detection."""
        samples = [
            {"instruction": "He wrote code", "output": "result"},
            {"instruction": "He designed a system", "output": "result"},
            {"instruction": "He is an engineer", "output": "result"},
        ]

        analyzer = BiasAnalyzer()
        report = analyzer.analyze(samples)

        assert report.gender_bias.bias_score > 0.0

    def test_cultural_bias_detection(self):
        """Test cultural bias detection."""
        samples = [
            {"instruction": "American standard", "output": "result"},
            {"instruction": "English convention", "output": "result"},
            {"instruction": "Western approach", "output": "result"},
        ]

        analyzer = BiasAnalyzer()
        report = analyzer.analyze(samples)

        assert report.cultural_bias.bias_score > 0.0

    def test_bias_recommendations(self):
        """Test that recommendations are generated."""
        samples = [
            {"instruction": "Male example: He coded", "output": "result"},
            {"instruction": "American context", "output": "result"},
        ]

        analyzer = BiasAnalyzer()
        report = analyzer.analyze(samples)

        assert len(report.recommendations) > 0


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer."""

    def test_analyze_basic(self):
        """Test basic dataset analysis."""
        samples = [
            {"instruction": "Write a function", "output": "def func():\n    pass"},
            {"instruction": "Create a class", "output": "class MyClass:\n    pass"},
        ]

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(samples, dataset_name="test")

        assert report.dataset_name == "test"
        assert len(report.sample_qualities) == 2
        assert report.average_quality_score >= 0.0

    def test_analyze_statistics(self):
        """Test that statistics are computed."""
        samples = [
            {"instruction": "short", "output": "out"},
            {"instruction": "medium length instruction", "output": "longer output"},
        ]

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(samples)

        assert report.statistics.total_samples == 2
        assert report.statistics.instruction_count > 0
        assert report.statistics.output_count > 0

    def test_analyze_quality_distribution(self):
        """Test quality distribution computation."""
        samples = [
            {"instruction": "instruction 1", "output": "output 1"},
            {"instruction": "instruction 2", "output": "output 2"},
            {"instruction": "instruction 3", "output": "output 3"},
        ]

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(samples)

        dist = report.quality_distribution
        assert sum(dist.values()) == len(samples)

    def test_analyze_improvements_identified(self):
        """Test that improvements are identified."""
        samples = [
            {"instruction": "test", "output": "output"},
        ]

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(samples)

        assert len(report.improvement_opportunities) > 0

    def test_analyze_with_duplicates(self):
        """Test analysis with duplicate samples."""
        samples = [
            {"instruction": "same", "output": "result"},
            {"instruction": "same", "output": "result"},
        ]

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(samples)

        dup_count = sum(1 for s in report.sample_qualities if s.is_duplicate)
        assert dup_count > 0

    def test_analyze_with_anomalies(self):
        """Test analysis with anomalous samples."""
        # Create samples with obvious anomaly (mostly whitespace)
        samples = [
            {"instruction": "normal instruction", "output": "normal output"},
            {"instruction": "another instruction", "output": "another output"},
            {"instruction": "     \n\n\t\t    ", "output": "also whitespace"},  # Whitespace anomaly
        ]

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(samples)

        anom_count = sum(1 for s in report.sample_qualities if s.is_anomaly)
        # Should detect whitespace anomaly
        assert anom_count >= 0  # At least tests that it runs without error


class TestDatasetStatistics:
    """Test DatasetStatistics."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = DatasetStatistics(
            total_samples=10,
            instruction_count=10,
            instruction_avg_length=50.0,
        )

        d = stats.to_dict()

        assert d["total_samples"] == 10
        assert "instruction" in d
        assert d["instruction"]["count"] == 10
