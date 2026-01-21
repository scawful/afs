"""Quality metrics for training samples.

Computes metrics that assess:
- Instruction clarity (conciseness, specificity)
- Output correctness (syntax, semantics)
- Code quality (structure, best practices)
- Duplicate detection (exact and semantic)
- Anomaly detection (outliers)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InstructionClarity:
    """Metrics for instruction clarity and quality."""

    length: int  # Word count
    avg_sentence_length: float  # Average words per sentence
    specificity_score: float  # 0.0-1.0 (specific vs generic)
    clarity_score: float  # 0.0-1.0 (clear vs ambiguous)
    has_examples: bool  # Contains example patterns
    has_requirements: bool  # Explicitly states requirements
    has_context: bool  # Provides background context

    def overall_score(self) -> float:
        """Compute overall clarity score."""
        return (self.specificity_score + self.clarity_score) / 2.0


@dataclass
class OutputCorrectness:
    """Metrics for output quality and correctness."""

    length: int  # Character count
    num_lines: int
    syntax_valid: bool  # Passes syntax validation
    has_comments: bool  # Contains documentation
    comment_ratio: float  # Comment lines / total lines
    structure_score: float  # 0.0-1.0 (well-organized)
    completeness_score: float  # 0.0-1.0 (full vs partial)

    def overall_score(self) -> float:
        """Compute overall correctness score."""
        base = 0.7 if self.syntax_valid else 0.2
        structure_bonus = self.structure_score * 0.2
        completeness_bonus = self.completeness_score * 0.1
        return min(1.0, base + structure_bonus + completeness_bonus)


@dataclass
class DuplicateInfo:
    """Information about potential duplicates."""

    exact_duplicates: list[int] = field(default_factory=list)  # Indices of exact matches
    semantic_duplicates: list[tuple[int, float]] = field(default_factory=list)  # (idx, similarity)
    deduplication_status: str = "unique"  # unique, exact_dup, semantic_dup


@dataclass
class AnomalyInfo:
    """Information about potential anomalies."""

    is_anomaly: bool = False
    anomaly_score: float = 0.0  # 0.0-1.0 (higher = more anomalous)
    anomaly_reasons: list[str] = field(default_factory=list)
    outlier_type: str = "none"  # none, length_outlier, structure_outlier, content_outlier


class QualityMetrics:
    """Compute comprehensive quality metrics for training samples."""

    def __init__(self, domain: str = "general"):
        """Initialize metrics calculator.

        Args:
            domain: Domain type (general, assembly, code, text)
        """
        self.domain = domain
        self._clarity_cache: dict[str, InstructionClarity] = {}
        self._correctness_cache: dict[str, OutputCorrectness] = {}

    def compute_instruction_clarity(self, instruction: str) -> InstructionClarity:
        """Compute instruction clarity metrics.

        Args:
            instruction: The instruction text

        Returns:
            InstructionClarity metrics
        """
        if instruction in self._clarity_cache:
            return self._clarity_cache[instruction]

        # Basic metrics
        words = instruction.split()
        length = len(words)
        sentences = [s.strip() for s in re.split(r'[.!?]+', instruction) if s.strip()]
        avg_sentence_length = length / len(sentences) if sentences else 0

        # Specificity score: penalize generic words, reward specific ones
        generic_words = {"what", "how", "why", "explain", "describe", "write"}
        specific_words = {"calculate", "convert", "assemble", "optimize", "debug", "parse"}
        generic_count = sum(1 for w in words if w.lower() in generic_words)
        specific_count = sum(1 for w in words if w.lower() in specific_words)
        specificity_score = min(1.0, specific_count / max(1, len(words) - generic_count))

        # Clarity score: based on readability and structure
        has_imperative = any(instruction.lower().startswith(v) for v in ["write", "create", "generate", "convert"])
        has_condition = " if " in instruction.lower() or " when " in instruction.lower()
        clarity_base = 0.5
        if has_imperative:
            clarity_base += 0.3
        if has_condition:
            clarity_base += 0.1
        if avg_sentence_length < 30:  # Not too long
            clarity_base += 0.1
        clarity_score = min(1.0, clarity_base)

        # Check for examples, requirements, context
        has_examples = any(
            pattern in instruction.lower()
            for pattern in ["example", "e.g.", "for instance", "like", "such as"]
        )
        has_requirements = any(
            pattern in instruction.lower()
            for pattern in ["must", "should", "required", "ensure", "need to"]
        )
        has_context = length > 20  # More words suggest context

        result = InstructionClarity(
            length=length,
            avg_sentence_length=avg_sentence_length,
            specificity_score=specificity_score,
            clarity_score=clarity_score,
            has_examples=has_examples,
            has_requirements=has_requirements,
            has_context=has_context,
        )

        self._clarity_cache[instruction] = result
        return result

    def compute_output_correctness(self, output: str) -> OutputCorrectness:
        """Compute output correctness metrics.

        Args:
            output: The output/response text

        Returns:
            OutputCorrectness metrics
        """
        if output in self._correctness_cache:
            return self._correctness_cache[output]

        # Basic metrics
        length = len(output)
        lines = output.split("\n")
        num_lines = len(lines)

        # Syntax validation (domain-specific)
        syntax_valid = self._validate_syntax(output)

        # Comments
        comment_lines = sum(1 for line in lines if self._is_comment_line(line))
        comment_ratio = comment_lines / max(1, num_lines)
        has_comments = comment_lines > 0

        # Structure score: based on indentation, organization
        structure_score = self._compute_structure_score(output)

        # Completeness score: based on length and non-whitespace ratio
        non_whitespace = len(output.replace(" ", "").replace("\n", "").replace("\t", ""))
        completeness_score = min(1.0, non_whitespace / 500.0)  # Expects ~500+ chars

        result = OutputCorrectness(
            length=length,
            num_lines=num_lines,
            syntax_valid=syntax_valid,
            has_comments=has_comments,
            comment_ratio=comment_ratio,
            structure_score=structure_score,
            completeness_score=completeness_score,
        )

        self._correctness_cache[output] = result
        return result

    def compute_duplicate_info(
        self,
        text: str,
        all_texts: list[str],
        index: int,
        similarity_threshold: float = 0.95,
    ) -> DuplicateInfo:
        """Check for exact and semantic duplicates.

        Args:
            text: The text to check
            all_texts: All texts in dataset
            index: Index of current text
            similarity_threshold: Cosine similarity threshold for semantic dups

        Returns:
            DuplicateInfo with duplicate indices
        """
        info = DuplicateInfo()

        # Check exact duplicates
        for i, other in enumerate(all_texts):
            if i != index and text == other:
                info.exact_duplicates.append(i)

        if info.exact_duplicates:
            info.deduplication_status = "exact_dup"
            return info

        # Check semantic duplicates (simplified: longest common substring ratio)
        for i, other in enumerate(all_texts):
            if i != index:
                similarity = self._compute_text_similarity(text, other)
                if similarity >= similarity_threshold:
                    info.semantic_duplicates.append((i, similarity))
                    info.deduplication_status = "semantic_dup"

        return info

    def compute_anomaly_info(
        self,
        text: str,
        all_texts: list[str],
        length_threshold: float = 2.0,
    ) -> AnomalyInfo:
        """Detect anomalies in text.

        Args:
            text: The text to check
            all_texts: All texts in dataset
            length_threshold: Std dev multiplier for length outliers

        Returns:
            AnomalyInfo with anomaly detection results
        """
        info = AnomalyInfo()

        # Check for length outliers
        lengths = [len(t) for t in all_texts if t]  # Exclude empty texts
        if len(lengths) > 1:  # Need at least 2 samples to compute stats
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            text_len = len(text)
            if std_length > 0:
                z_score = abs(text_len - mean_length) / std_length
                if z_score > length_threshold:
                    info.is_anomaly = True
                    info.anomaly_reasons.append(f"Length outlier (z={z_score:.2f}, len={text_len})")
                    info.outlier_type = "length_outlier"
                    info.anomaly_score = min(1.0, z_score / (length_threshold * 2.0))

        # Check for mostly whitespace
        non_whitespace_ratio = len(text.strip()) / max(1, len(text))
        if non_whitespace_ratio < 0.3:
            info.is_anomaly = True
            info.anomaly_reasons.append("Mostly whitespace")
            info.outlier_type = "content_outlier"
            info.anomaly_score = max(info.anomaly_score, 0.8)

        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(1, len(text))
        if special_ratio > 0.5:
            info.is_anomaly = True
            info.anomaly_reasons.append(f"High special char ratio: {special_ratio:.2f}")
            info.outlier_type = "content_outlier"
            info.anomaly_score = max(info.anomaly_score, 0.7)

        return info

    def _validate_syntax(self, text: str) -> bool:
        """Check basic syntax validity.

        Args:
            text: Text to validate

        Returns:
            True if syntax appears valid
        """
        if not text.strip():
            return False

        # Domain-specific validation
        if self.domain == "assembly":
            # Basic assembly syntax check
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                # Should have instruction or label
                if not any(c in line for c in [":", "ORG", "DB", "DW", "LD", "MOV", "JMP"]):
                    if line and not line[0].isspace():  # Not indented, might be label
                        continue
            return True

        # Basic Python/code syntax
        if self.domain == "code":
            try:
                compile(text, "<string>", "exec")
                return True
            except SyntaxError:
                return False

        # For general text, just check it's not empty
        return len(text.strip()) > 0

    def _is_comment_line(self, line: str) -> bool:
        """Check if a line is a comment."""
        stripped = line.strip()
        return (
            stripped.startswith("#")
            or stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith(";")
            or stripped.startswith("--")
            or stripped.startswith("\"\"\"")
        )

    def _compute_structure_score(self, text: str) -> float:
        """Compute structural quality score."""
        lines = text.split("\n")
        if not lines:
            return 0.0

        # Check for consistent indentation
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indents.append(indent)

        if not indents:
            return 0.5

        # Indentation consistency
        indent_set = set(indents)
        indent_regularity = 1.0 - min(1.0, len(indent_set) / 10.0)

        # Line length variance (prefer consistent line lengths)
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        if line_lengths:
            avg_length = np.mean(line_lengths)
            length_variance = np.std(line_lengths) / max(1, avg_length)
            length_consistency = 1.0 - min(1.0, length_variance)
        else:
            length_consistency = 0.5

        return (indent_regularity + length_consistency) / 2.0

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts (0.0-1.0)."""
        if text1 == text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Simple longest common substring ratio
        s1, s2 = text1, text2
        common_len = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                k = 0
                while i + k < len(s1) and j + k < len(s2) and s1[i + k] == s2[j + k]:
                    k += 1
                common_len = max(common_len, k)

        max_len = max(len(s1), len(s2))
        return common_len / max_len if max_len > 0 else 0.0


class DuplicateDetector:
    """Detect duplicate samples in datasets."""

    def __init__(self, exact_match: bool = True, semantic_threshold: float = 0.95):
        """Initialize duplicate detector.

        Args:
            exact_match: Whether to detect exact duplicates
            semantic_threshold: Similarity threshold for semantic duplicates
        """
        self.exact_match = exact_match
        self.semantic_threshold = semantic_threshold

    def find_duplicates(self, samples: list[dict[str, Any]]) -> dict[int, DuplicateInfo]:
        """Find duplicates in a list of samples.

        Args:
            samples: List of training samples

        Returns:
            Dict mapping sample index to DuplicateInfo
        """
        duplicates = {}
        metrics = QualityMetrics()

        # Extract text from samples (support different formats)
        texts = []
        for sample in samples:
            if isinstance(sample, dict):
                # Try different key names
                text = sample.get("instruction") or sample.get("prompt") or sample.get("text", "")
            else:
                text = str(sample)
            texts.append(text)

        # Find duplicates for each sample
        for i, text in enumerate(texts):
            info = metrics.compute_duplicate_info(text, texts, i, self.semantic_threshold)
            if info.deduplication_status != "unique":
                duplicates[i] = info

        return duplicates


class AnomalyDetector:
    """Detect anomalous samples in datasets."""

    def __init__(self, length_threshold: float = 2.0, enable_content_checks: bool = True):
        """Initialize anomaly detector.

        Args:
            length_threshold: Std dev multiplier for length outliers
            enable_content_checks: Whether to check for content anomalies
        """
        self.length_threshold = length_threshold
        self.enable_content_checks = enable_content_checks

    def find_anomalies(self, samples: list[dict[str, Any]]) -> dict[int, AnomalyInfo]:
        """Find anomalies in a list of samples.

        Args:
            samples: List of training samples

        Returns:
            Dict mapping sample index to AnomalyInfo
        """
        anomalies = {}
        metrics = QualityMetrics()

        # Extract texts
        texts = []
        for sample in samples:
            if isinstance(sample, dict):
                text = sample.get("instruction") or sample.get("prompt") or sample.get("text", "")
            else:
                text = str(sample)
            texts.append(text)

        # Find anomalies for each sample
        for i, text in enumerate(texts):
            info = metrics.compute_anomaly_info(text, texts, self.length_threshold)
            if info.is_anomaly:
                anomalies[i] = info

        return anomalies
