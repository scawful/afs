"""Main dataset quality analyzer.

Provides comprehensive analysis of training datasets including:
- Dataset statistics (size, diversity, distributions)
- Per-sample quality metrics
- Bias detection
- Duplicate and anomaly detection
- Improvement recommendations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .bias import BiasAnalyzer, BiasReport
from .metrics import (
    AnomalyDetector,
    AnomalyInfo,
    DuplicateDetector,
    DuplicateInfo,
    QualityMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """Overall dataset statistics."""

    total_samples: int = 0
    total_size_bytes: int = 0
    unique_samples: int = 0

    # Instruction stats
    instruction_count: int = 0
    instruction_min_length: int = 0
    instruction_max_length: int = 0
    instruction_avg_length: float = 0.0
    instruction_std_length: float = 0.0
    instruction_vocab_size: int = 0

    # Output stats
    output_count: int = 0
    output_min_length: int = 0
    output_max_length: int = 0
    output_avg_length: float = 0.0
    output_std_length: float = 0.0
    output_vocab_size: int = 0

    # Quality metrics
    samples_with_syntax_errors: int = 0
    samples_with_duplicates: int = 0
    anomalous_samples: int = 0

    # Diversity
    categories: list[str] = field(default_factory=list)
    category_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "total_size_bytes": self.total_size_bytes,
            "unique_samples": self.unique_samples,
            "instruction": {
                "count": self.instruction_count,
                "min_length": self.instruction_min_length,
                "max_length": self.instruction_max_length,
                "avg_length": round(self.instruction_avg_length, 2),
                "std_length": round(self.instruction_std_length, 2),
                "vocab_size": self.instruction_vocab_size,
            },
            "output": {
                "count": self.output_count,
                "min_length": self.output_min_length,
                "max_length": self.output_max_length,
                "avg_length": round(self.output_avg_length, 2),
                "std_length": round(self.output_std_length, 2),
                "vocab_size": self.output_vocab_size,
            },
            "quality": {
                "samples_with_syntax_errors": self.samples_with_syntax_errors,
                "samples_with_duplicates": self.samples_with_duplicates,
                "anomalous_samples": self.anomalous_samples,
            },
            "diversity": {
                "categories": self.categories,
                "category_distribution": self.category_distribution,
            },
        }


@dataclass
class SampleQuality:
    """Quality metrics for a single sample."""

    index: int
    instruction_clarity_score: float = 0.0
    output_correctness_score: float = 0.0
    overall_quality_score: float = 0.0

    is_duplicate: bool = False
    duplicate_info: DuplicateInfo = field(default_factory=DuplicateInfo)

    is_anomaly: bool = False
    anomaly_info: AnomalyInfo = field(default_factory=AnomalyInfo)

    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "instruction_clarity_score": round(self.instruction_clarity_score, 3),
            "output_correctness_score": round(self.output_correctness_score, 3),
            "overall_quality_score": round(self.overall_quality_score, 3),
            "is_duplicate": self.is_duplicate,
            "duplicate_status": self.duplicate_info.deduplication_status,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_info.anomaly_score, 3),
            "anomaly_reasons": self.anomaly_info.anomaly_reasons,
            "recommendations": self.recommendations,
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for a dataset."""

    dataset_name: str = ""
    dataset_path: str = ""
    analysis_timestamp: str = ""

    statistics: DatasetStatistics = field(default_factory=DatasetStatistics)
    sample_qualities: list[SampleQuality] = field(default_factory=list)
    bias_report: BiasReport = field(default_factory=BiasReport)

    # Summary metrics
    average_quality_score: float = 0.0
    quality_distribution: dict[str, int] = field(default_factory=dict)
    improvement_opportunities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": {
                "name": self.dataset_name,
                "path": self.dataset_path,
                "timestamp": self.analysis_timestamp,
            },
            "statistics": self.statistics.to_dict(),
            "summary": {
                "average_quality_score": round(self.average_quality_score, 3),
                "quality_distribution": self.quality_distribution,
                "improvement_opportunities": self.improvement_opportunities,
            },
            "bias": self.bias_report.to_dict(),
            "sample_count": len(self.sample_qualities),
            "high_quality_samples": sum(1 for s in self.sample_qualities if s.overall_quality_score >= 0.8),
            "medium_quality_samples": sum(
                1 for s in self.sample_qualities if 0.5 <= s.overall_quality_score < 0.8
            ),
            "low_quality_samples": sum(1 for s in self.sample_qualities if s.overall_quality_score < 0.5),
        }

    def save_json(self, path: Path | str) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved quality report to {path}")

    def save_samples_jsonl(self, path: Path | str) -> None:
        """Save per-sample quality data to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sample in self.sample_qualities:
                f.write(json.dumps(sample.to_dict()) + "\n")
        logger.info(f"Saved sample qualities to {path}")


class DatasetAnalyzer:
    """Comprehensive dataset quality analyzer."""

    def __init__(self, domain: str = "general"):
        """Initialize analyzer.

        Args:
            domain: Domain type (general, assembly, code)
        """
        self.domain = domain
        self.metrics = QualityMetrics(domain=domain)
        self.bias_analyzer = BiasAnalyzer()
        self.duplicate_detector = DuplicateDetector()
        self.anomaly_detector = AnomalyDetector()

    def analyze(
        self,
        samples: list[dict[str, Any]],
        dataset_name: str = "untitled",
        dataset_path: str = "",
    ) -> QualityReport:
        """Analyze a dataset comprehensively.

        Args:
            samples: List of training samples (dicts with 'instruction'/'output' fields)
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset

        Returns:
            QualityReport with complete analysis
        """
        from datetime import datetime

        report = QualityReport(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            analysis_timestamp=datetime.now().isoformat(),
        )

        if not samples:
            logger.warning("Empty dataset provided")
            return report

        # Extract texts
        instructions = []
        outputs = []
        for sample in samples:
            if isinstance(sample, dict):
                instr = sample.get("instruction") or sample.get("prompt", "")
                output = sample.get("output") or sample.get("response", "")
            else:
                instr = ""
                output = ""
            instructions.append(instr)
            outputs.append(output)

        # Compute dataset statistics
        report.statistics = self._compute_statistics(samples, instructions, outputs)

        # Analyze each sample
        sample_qualities = []
        duplicates = self.duplicate_detector.find_duplicates(samples)
        anomalies = self.anomaly_detector.find_anomalies(samples)

        for i, sample in enumerate(samples):
            quality = self._analyze_sample(
                i,
                sample,
                instructions[i] if i < len(instructions) else "",
                outputs[i] if i < len(outputs) else "",
                duplicates.get(i),
                anomalies.get(i),
            )
            sample_qualities.append(quality)

        report.sample_qualities = sample_qualities

        # Analyze bias
        report.bias_report = self.bias_analyzer.analyze(samples)

        # Compute summary metrics
        report.average_quality_score = np.mean([s.overall_quality_score for s in sample_qualities])
        report.quality_distribution = self._compute_quality_distribution(sample_qualities)
        report.improvement_opportunities = self._identify_improvements(report)

        return report

    def analyze_file(self, path: Path | str) -> QualityReport:
        """Analyze a JSONL or JSON file.

        Args:
            path: Path to dataset file

        Returns:
            QualityReport
        """
        path = Path(path)
        samples = []

        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else [data]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return self.analyze(samples, dataset_name=path.stem, dataset_path=str(path))

    def _compute_statistics(
        self,
        samples: list[dict[str, Any]],
        instructions: list[str],
        outputs: list[str],
    ) -> DatasetStatistics:
        """Compute dataset-level statistics."""
        stats = DatasetStatistics()
        stats.total_samples = len(samples)
        stats.total_size_bytes = sum(len(json.dumps(s)) for s in samples)

        # Count unique
        unique_instructions = set(instructions)
        unique_outputs = set(outputs)
        stats.unique_samples = len(unique_instructions)

        # Instruction stats
        instruction_lengths = [len(i.split()) for i in instructions if i]
        stats.instruction_count = len([i for i in instructions if i])
        if instruction_lengths:
            stats.instruction_min_length = min(instruction_lengths)
            stats.instruction_max_length = max(instruction_lengths)
            stats.instruction_avg_length = np.mean(instruction_lengths)
            stats.instruction_std_length = np.std(instruction_lengths)
        stats.instruction_vocab_size = len(set(" ".join(instructions).split()))

        # Output stats
        output_lengths = [len(o.split()) for o in outputs if o]
        stats.output_count = len([o for o in outputs if o])
        if output_lengths:
            stats.output_min_length = min(output_lengths)
            stats.output_max_length = max(output_lengths)
            stats.output_avg_length = np.mean(output_lengths)
            stats.output_std_length = np.std(output_lengths)
        stats.output_vocab_size = len(set(" ".join(outputs).split()))

        # Category distribution
        if samples and isinstance(samples[0], dict) and "category" in samples[0]:
            categories = [s.get("category", "unknown") for s in samples]
            stats.categories = list(set(categories))
            stats.category_distribution = {c: categories.count(c) for c in stats.categories}

        return stats

    def _analyze_sample(
        self,
        index: int,
        sample: dict[str, Any],
        instruction: str,
        output: str,
        duplicate_info: DuplicateInfo | None,
        anomaly_info: AnomalyInfo | None,
    ) -> SampleQuality:
        """Analyze a single sample."""
        quality = SampleQuality(index=index)

        # Compute clarity score
        clarity = self.metrics.compute_instruction_clarity(instruction)
        quality.instruction_clarity_score = clarity.overall_score()

        # Compute correctness score
        correctness = self.metrics.compute_output_correctness(output)
        quality.output_correctness_score = correctness.overall_score()

        # Overall score (weighted average)
        quality.overall_quality_score = (quality.instruction_clarity_score * 0.4 + quality.output_correctness_score * 0.6)

        # Duplicate info
        if duplicate_info:
            quality.is_duplicate = True
            quality.duplicate_info = duplicate_info

        # Anomaly info
        if anomaly_info:
            quality.is_anomaly = True
            quality.anomaly_info = anomaly_info

        # Generate recommendations
        quality.recommendations = self._generate_sample_recommendations(quality, instruction, output)

        return quality

    def _generate_sample_recommendations(
        self,
        quality: SampleQuality,
        instruction: str,
        output: str,
    ) -> list[str]:
        """Generate improvement recommendations for a sample."""
        recommendations = []

        if quality.instruction_clarity_score < 0.5:
            recommendations.append("Improve instruction clarity: be more specific about requirements")

        if quality.output_correctness_score < 0.5:
            recommendations.append("Output quality is low: check syntax and completeness")

        if quality.is_duplicate:
            recommendations.append(f"Duplicate detected: similar to sample {quality.duplicate_info.exact_duplicates}")

        if quality.is_anomaly:
            recommendations.extend([f"Anomaly: {r}" for r in quality.anomaly_info.anomaly_reasons])

        if quality.overall_quality_score > 0.8:
            recommendations.append("HIGH QUALITY: Good candidate for data augmentation")

        return recommendations

    def _compute_quality_distribution(self, samples: list[SampleQuality]) -> dict[str, int]:
        """Compute distribution of quality scores."""
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for sample in samples:
            score = sample.overall_quality_score
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        return distribution

    def _identify_improvements(self, report: QualityReport) -> list[str]:
        """Identify overall improvement opportunities."""
        opportunities = []

        # Check average quality
        if report.average_quality_score < 0.5:
            opportunities.append(
                f"Overall dataset quality is low ({report.average_quality_score:.1%}). "
                "Consider filtering out low-quality samples or improving them."
            )

        # Check duplicates
        dup_count = sum(1 for s in report.sample_qualities if s.is_duplicate)
        if dup_count > 0:
            opportunities.append(
                f"Found {dup_count} duplicate or near-duplicate samples. "
                "Remove or consolidate duplicates to improve training efficiency."
            )

        # Check anomalies
        anom_count = sum(1 for s in report.sample_qualities if s.is_anomaly)
        if anom_count > 0:
            opportunities.append(
                f"Found {anom_count} anomalous samples. "
                "Review and either fix or remove anomalies."
            )

        # Check instruction clarity
        low_clarity = sum(1 for s in report.sample_qualities if s.instruction_clarity_score < 0.5)
        if low_clarity / len(report.sample_qualities) > 0.2:
            opportunities.append(
                f"{low_clarity} samples have low instruction clarity. "
                "Improve instructions by being more specific and detailed."
            )

        # Check output correctness
        low_correctness = sum(1 for s in report.sample_qualities if s.output_correctness_score < 0.5)
        if low_correctness / len(report.sample_qualities) > 0.2:
            opportunities.append(
                f"{low_correctness} samples have low output quality. "
                "Check outputs for syntax errors, completeness, and structure."
            )

        # Check bias
        if report.bias_report.overall_bias_score > 0.3:
            opportunities.append(
                f"Dataset shows signs of bias (score: {report.bias_report.overall_bias_score:.2f}). "
                "Follow bias detection recommendations."
            )

        # High-quality candidates for augmentation
        high_quality = sum(1 for s in report.sample_qualities if s.overall_quality_score >= 0.8)
        if high_quality > 0:
            opportunities.append(
                f"Found {high_quality} high-quality samples suitable for data augmentation. "
                "Consider using these as templates for generating similar samples."
            )

        if not opportunities:
            opportunities.append("Dataset quality is good. Continue monitoring with periodic analysis.")

        return opportunities


def analyze_dataset(
    samples: list[dict[str, Any]],
    dataset_name: str = "untitled",
    domain: str = "general",
) -> QualityReport:
    """Convenience function to analyze a dataset.

    Args:
        samples: List of training samples
        dataset_name: Name of the dataset
        domain: Domain type (general, assembly, code)

    Returns:
        QualityReport with analysis
    """
    analyzer = DatasetAnalyzer(domain=domain)
    return analyzer.analyze(samples, dataset_name=dataset_name)
