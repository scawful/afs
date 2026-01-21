"""Bias detection and analysis for training datasets.

Detects and reports:
- Gender bias in examples and language
- Cultural bias in domain choices
- Technical bias in code style or design patterns
- Language bias in terminology
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenderBiasMetrics:
    """Metrics for gender bias detection."""

    pronoun_counts: dict[str, int] = field(default_factory=dict)
    gender_ratio: float = 0.0  # 0.0 = balanced, 1.0 = skewed
    biased_examples: list[str] = field(default_factory=list)
    bias_score: float = 0.0  # 0.0 = none, 1.0 = severe


@dataclass
class CulturalBiasMetrics:
    """Metrics for cultural bias detection."""

    cultural_references: list[str] = field(default_factory=list)
    language_diversity: float = 0.0  # 0.0 = monolingual, 1.0 = diverse
    regional_bias: float = 0.0
    bias_score: float = 0.0


@dataclass
class TechnicalBiasMetrics:
    """Metrics for technical bias detection."""

    code_style_consistency: float = 0.0
    framework_diversity: float = 0.0
    paradigm_diversity: float = 0.0
    accessibility_score: float = 0.0
    bias_score: float = 0.0


@dataclass
class BiasReport:
    """Comprehensive bias analysis report."""

    gender_bias: GenderBiasMetrics = field(default_factory=GenderBiasMetrics)
    cultural_bias: CulturalBiasMetrics = field(default_factory=CulturalBiasMetrics)
    technical_bias: TechnicalBiasMetrics = field(default_factory=TechnicalBiasMetrics)
    overall_bias_score: float = 0.0  # 0.0 = balanced, 1.0 = highly biased
    recommendations: list[str] = field(default_factory=list)
    high_risk_samples: list[tuple[int, str]] = field(default_factory=list)  # (idx, reason)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "gender_bias": {
                "pronouns": self.gender_bias.pronoun_counts,
                "ratio": self.gender_bias.gender_ratio,
                "examples": self.gender_bias.biased_examples[:5],
                "score": self.gender_bias.bias_score,
            },
            "cultural_bias": {
                "references": self.cultural_bias.cultural_references[:10],
                "language_diversity": self.cultural_bias.language_diversity,
                "regional_bias": self.cultural_bias.regional_bias,
                "score": self.cultural_bias.bias_score,
            },
            "technical_bias": {
                "code_style": self.technical_bias.code_style_consistency,
                "framework_diversity": self.technical_bias.framework_diversity,
                "paradigm_diversity": self.technical_bias.paradigm_diversity,
                "accessibility": self.technical_bias.accessibility_score,
                "score": self.technical_bias.bias_score,
            },
            "overall_bias_score": self.overall_bias_score,
            "recommendations": self.recommendations,
            "high_risk_samples": self.high_risk_samples[:20],
        }


class GenderBiasDetector:
    """Detect gender bias in text and examples."""

    def __init__(self):
        """Initialize gender bias detector."""
        # Pronouns
        self.male_pronouns = {"he", "him", "his", "mr", "sir", "boy", "man", "male"}
        self.female_pronouns = {"she", "her", "hers", "ms", "mrs", "madam", "girl", "woman", "female"}

        # Gendered words
        self.male_words = {
            "programmer", "engineer", "scientist", "developer", "coder",
            "leader", "manager", "boss", "executive", "chairman",
        }
        self.female_words = {
            "nurse", "secretary", "teacher", "assistant",
            "cleaner", "maid", "stewardess",
        }

        # Occupational examples
        self.gendered_occupations = {
            "nurse": "female", "doctor": "neutral", "teacher": "female",
            "engineer": "male", "programmer": "male", "designer": "neutral",
        }

    def analyze(self, texts: list[str]) -> GenderBiasMetrics:
        """Analyze gender bias in texts.

        Args:
            texts: List of text samples

        Returns:
            GenderBiasMetrics with detected bias
        """
        metrics = GenderBiasMetrics()

        # Count pronouns
        male_count = 0
        female_count = 0
        biased_examples = []

        for text in texts:
            lower_text = text.lower()
            words = re.findall(r"\b\w+\b", lower_text)

            text_male = sum(1 for w in words if w in self.male_pronouns)
            text_female = sum(1 for w in words if w in self.female_pronouns)

            male_count += text_male
            female_count += text_female

            # Check for biased examples
            if text_male > text_female:
                biased_examples.append(text[:100])
            elif text_female > text_male:
                biased_examples.append(text[:100])

        metrics.pronoun_counts = {
            "male": male_count,
            "female": female_count,
            "neutral": len(texts) * 10 - male_count - female_count,
        }

        # Calculate gender ratio
        total = male_count + female_count
        if total > 0:
            male_ratio = male_count / total
            # 0.5 = balanced, 0.0 or 1.0 = skewed
            metrics.gender_ratio = abs(0.5 - male_ratio)
        else:
            metrics.gender_ratio = 0.0

        metrics.biased_examples = biased_examples[:10]

        # Score: higher = more biased
        metrics.bias_score = min(1.0, metrics.gender_ratio * 2.0)

        return metrics


class CulturalBiasDetector:
    """Detect cultural bias in text."""

    def __init__(self):
        """Initialize cultural bias detector."""
        # English-centric bias markers
        self.english_bias_markers = {
            "american": ["dollar", "usa", "american", "english", "british", "australian"],
            "western": ["europe", "america", "western", "developed", "first world"],
        }

        # Example patterns
        self.western_names = {"john", "mary", "peter", "jane", "robert", "susan"}
        self.other_names = {"ali", "priya", "zhang", "fatima", "jamal", "amara"}

    def analyze(self, texts: list[str]) -> CulturalBiasMetrics:
        """Analyze cultural bias in texts.

        Args:
            texts: List of text samples

        Returns:
            CulturalBiasMetrics with detected bias
        """
        metrics = CulturalBiasMetrics()

        cultural_refs = []
        western_names_found = 0
        other_names_found = 0

        for text in texts:
            lower_text = text.lower()

            # Count cultural references
            for category, markers in self.english_bias_markers.items():
                for marker in markers:
                    if marker in lower_text:
                        cultural_refs.append(f"{category}: {marker}")

            # Count names
            words = re.findall(r"\b\w+\b", lower_text)
            for word in words:
                if word in self.western_names:
                    western_names_found += 1
                elif word in self.other_names:
                    other_names_found += 1

        metrics.cultural_references = list(set(cultural_refs))

        # Language diversity (simplified)
        total_refs = western_names_found + other_names_found
        if total_refs > 0:
            metrics.language_diversity = other_names_found / total_refs
        else:
            metrics.language_diversity = 0.5  # Neutral if no examples

        # Regional bias (western-centric)
        western_ratio = len([r for r in cultural_refs if "american" in r or "western" in r])
        metrics.regional_bias = western_ratio / max(1, len(cultural_refs))

        # Overall score
        bias_factors = [
            1.0 - metrics.language_diversity,  # Low diversity = high bias
            metrics.regional_bias,  # Western-centric
        ]
        metrics.bias_score = sum(bias_factors) / len(bias_factors)

        return metrics


class TechnicalBiasDetector:
    """Detect technical bias in code examples."""

    def __init__(self):
        """Initialize technical bias detector."""
        self.common_frameworks = {
            "react", "vue", "angular", "django", "flask",
            "spring", "rails", "fastapi", "express",
        }
        self.programming_paradigms = {
            "oop", "functional", "procedural", "declarative", "imperative",
        }

    def analyze(self, texts: list[str]) -> TechnicalBiasMetrics:
        """Analyze technical bias in code samples.

        Args:
            texts: List of code samples

        Returns:
            TechnicalBiasMetrics with detected bias
        """
        metrics = TechnicalBiasMetrics()

        frameworks_found = set()
        paradigms_found = set()
        style_patterns = []

        for text in texts:
            lower_text = text.lower()

            # Detect frameworks
            for fw in self.common_frameworks:
                if fw in lower_text:
                    frameworks_found.add(fw)

            # Detect paradigms (simplified)
            for paradigm in ["class ", "def ", "function", "=>", "=>", "lambda"]:
                if paradigm in lower_text:
                    paradigms_found.add(paradigm)

            # Code style (indentation, naming, etc.)
            if "    " in text:
                style_patterns.append("space_indent")
            if "\t" in text:
                style_patterns.append("tab_indent")

        # Diversity scores
        if frameworks_found:
            metrics.framework_diversity = len(frameworks_found) / len(self.common_frameworks)
        if paradigms_found:
            metrics.paradigm_diversity = len(paradigms_found) / len(self.programming_paradigms)

        # Code style consistency
        style_consistency = len(set(style_patterns)) / max(1, len(style_patterns))
        metrics.code_style_consistency = 1.0 - style_consistency  # Higher = more consistent

        # Accessibility (comments, docstrings, etc.)
        comment_ratio = sum(1 for text in texts if "#" in text or '"""' in text) / max(1, len(texts))
        metrics.accessibility_score = comment_ratio

        # Overall bias
        bias_factors = [
            1.0 - metrics.framework_diversity,
            1.0 - metrics.paradigm_diversity,
            1.0 - metrics.accessibility_score,
        ]
        metrics.bias_score = sum(bias_factors) / len(bias_factors)

        return metrics


class BiasAnalyzer:
    """Comprehensive bias analysis for datasets."""

    def __init__(self):
        """Initialize bias analyzer."""
        self.gender_detector = GenderBiasDetector()
        self.cultural_detector = CulturalBiasDetector()
        self.technical_detector = TechnicalBiasDetector()

    def analyze(self, samples: list[dict[str, Any]]) -> BiasReport:
        """Analyze bias in a dataset.

        Args:
            samples: List of training samples

        Returns:
            BiasReport with detailed analysis
        """
        report = BiasReport()

        # Extract texts
        instructions = []
        outputs = []

        for sample in samples:
            if isinstance(sample, dict):
                instr = sample.get("instruction") or sample.get("prompt", "")
                output = sample.get("output") or sample.get("response", "")
                instructions.append(instr)
                outputs.append(output)

        all_texts = instructions + outputs

        # Analyze each bias type
        report.gender_bias = self.gender_detector.analyze(all_texts)
        report.cultural_bias = self.cultural_detector.analyze(all_texts)
        report.technical_bias = self.technical_detector.analyze(all_texts)

        # Calculate overall bias score
        bias_scores = [
            report.gender_bias.bias_score,
            report.cultural_bias.bias_score,
            report.technical_bias.bias_score,
        ]
        report.overall_bias_score = sum(bias_scores) / len(bias_scores)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Identify high-risk samples
        report.high_risk_samples = self._find_high_risk_samples(samples, report)

        return report

    def _generate_recommendations(self, report: BiasReport) -> list[str]:
        """Generate improvement recommendations.

        Args:
            report: BiasReport with analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        if report.gender_bias.bias_score > 0.3:
            recommendations.append(
                "High gender bias detected. Consider adding examples with balanced pronouns and occupations."
            )

        if report.cultural_bias.bias_score > 0.3:
            recommendations.append(
                "High cultural bias detected. Diversify examples with different cultural contexts and names."
            )

        if report.technical_bias.bias_score > 0.3:
            recommendations.append(
                "High technical bias detected. Include examples from diverse frameworks and programming paradigms."
            )

        if report.gender_bias.gender_ratio > 0.4:
            recommendations.append(
                f"Pronoun distribution is skewed (male: {report.gender_bias.pronoun_counts.get('male', 0)}, "
                f"female: {report.gender_bias.pronoun_counts.get('female', 0)}). "
                "Aim for more balanced representation."
            )

        if report.cultural_bias.language_diversity < 0.3:
            recommendations.append(
                "Language diversity is low. Include examples with diverse names and cultural contexts."
            )

        if report.technical_bias.framework_diversity < 0.3:
            recommendations.append(
                f"Framework diversity is low ({report.technical_bias.framework_diversity:.1%}). "
                "Include examples from different frameworks and paradigms."
            )

        if not recommendations:
            recommendations.append("Dataset shows balanced representation across bias dimensions.")

        return recommendations

    def _find_high_risk_samples(
        self,
        samples: list[dict[str, Any]],
        report: BiasReport,
    ) -> list[tuple[int, str]]:
        """Find samples with highest bias risk.

        Args:
            samples: Training samples
            report: Bias analysis report

        Returns:
            List of (index, reason) tuples
        """
        high_risk = []

        for i, sample in enumerate(samples):
            reasons = []

            if isinstance(sample, dict):
                text = sample.get("instruction", "") + " " + sample.get("output", "")

                # Check for gender bias indicators
                if any(p in text.lower() for p in self.gender_detector.male_pronouns):
                    if "nurse" not in text.lower() and "teacher" not in text.lower():
                        reasons.append("Potential gender bias (male pronouns)")

                # Check for cultural bias
                if any(ref in text.lower() for ref in ["american", "english", "western"]):
                    if not any(name in text.lower() for name in self.cultural_detector.other_names):
                        reasons.append("Potential cultural bias (Western-centric)")

            if reasons:
                high_risk.append((i, " | ".join(reasons)))

        return high_risk[:100]


def detect_biases(samples: list[dict[str, Any]]) -> BiasReport:
    """Convenience function to detect biases in dataset.

    Args:
        samples: List of training samples

    Returns:
        BiasReport with analysis
    """
    analyzer = BiasAnalyzer()
    return analyzer.analyze(samples)
