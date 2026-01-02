"""Encoder-based utilities for improving decoder training data.

Uses the ASM tokenizer and encoder models to:
1. Semantic deduplication - Remove near-duplicate samples
2. Diversity sampling - Select representative diverse samples
3. Quality clustering - Group samples by semantic similarity
4. Anomaly detection - Find outlier/malformed samples

These help improve decoder (LLM) fine-tuning by ensuring
high-quality, diverse training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class EncoderConfig:
    """Configuration for encoder-based utilities."""

    # Model paths
    tokenizer_path: Path | None = None
    encoder_path: Path | None = None

    # Deduplication
    similarity_threshold: float = 0.95  # Cosine similarity for duplicates

    # Diversity sampling
    num_clusters: int = 100  # For k-means clustering
    samples_per_cluster: int = 10  # Samples to keep per cluster

    # Quality
    min_instruction_tokens: int = 5
    min_output_tokens: int = 10
    max_unk_ratio: float = 0.1  # Max unknown token ratio


class EncoderDataProcessor:
    """Process training data using encoder embeddings."""

    def __init__(
        self,
        config: EncoderConfig | None = None,
        tokenizer: Any | None = None,
        encoder: Any | None = None,
    ):
        """Initialize processor.

        Args:
            config: Processing configuration
            tokenizer: ASMTokenizer instance (loads from config if None)
            encoder: Encoder model (loads from config if None)
        """
        self.config = config or EncoderConfig()
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._embeddings_cache: dict[str, np.ndarray] = {}

    @property
    def tokenizer(self) -> Any:
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from afs.tokenizer import ASMTokenizer

            if self.config.tokenizer_path:
                self._tokenizer = ASMTokenizer.load(self.config.tokenizer_path)
            else:
                self._tokenizer = ASMTokenizer()
        return self._tokenizer

    @property
    def encoder(self) -> Any:
        """Lazy load encoder model."""
        if self._encoder is None and self.config.encoder_path:
            import torch
            from transformers import BertModel

            self._encoder = BertModel.from_pretrained(self.config.encoder_path)
            self._encoder.eval()
        return self._encoder

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using encoder or fallback to token counts."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]

        if self.encoder is not None:
            embedding = self._get_encoder_embedding(text)
        else:
            # Fallback: use token frequency vector
            embedding = self._get_token_embedding(text)

        self._embeddings_cache[text] = embedding
        return embedding

    def _get_encoder_embedding(self, text: str) -> np.ndarray:
        """Get embedding from encoder model."""
        import torch

        encoded = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.encoder(**encoded)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()

        return embedding.flatten()

    def _get_token_embedding(self, text: str) -> np.ndarray:
        """Get embedding from token frequency (fallback when no encoder)."""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]

        # Create frequency vector
        vocab_size = len(self.tokenizer)
        freq = np.zeros(vocab_size)
        for token_id in input_ids:
            if token_id < vocab_size:
                freq[token_id] += 1

        # Normalize
        norm = np.linalg.norm(freq)
        if norm > 0:
            freq = freq / norm

        return freq

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # =========================================================================
    # Quality Analysis
    # =========================================================================

    def analyze_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Analyze a single sample for quality metrics.

        Note: The ASM tokenizer is designed for assembly code (output field).
        Instructions are natural language and will have high UNK ratios - this
        is expected and not flagged as an issue.

        Args:
            sample: Training sample with 'instruction' and 'output' fields

        Returns:
            Analysis dict with quality metrics
        """
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")

        # Tokenize output (assembly code) with ASM tokenizer
        output_tokens = self.tokenizer.tokenize(output)
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)

        unk_id = self.tokenizer.unk_token_id
        output_unk = sum(1 for i in output_ids if i == unk_id)
        output_unk_ratio = output_unk / len(output_ids) if output_ids else 0

        # Count instruction words (natural language, not ASM tokenized)
        instr_words = instruction.split()

        # Quality flags
        issues = []

        # Instruction checks (natural language)
        if len(instr_words) < self.config.min_instruction_tokens:
            issues.append("instruction_too_short")
        if "TODO" in instruction:
            issues.append("incomplete_instruction_todo")
        if instruction.strip().endswith("that"):
            issues.append("incomplete_instruction_trailing")
        if instruction.strip().endswith("that ."):
            issues.append("incomplete_instruction_trailing")

        # Output checks (assembly code)
        if len(output_tokens) < self.config.min_output_tokens:
            issues.append("output_too_short")
        if output_unk_ratio > self.config.max_unk_ratio:
            issues.append("output_high_unk")
        if not output.strip():
            issues.append("output_empty")

        # Check for assembly-specific quality
        has_opcode = any(
            t in output.upper()
            for t in ["LDA", "STA", "JSR", "JSL", "RTS", "RTL", "BRA", "BEQ", "BNE"]
        )
        if len(output) > 50 and not has_opcode:
            issues.append("output_no_opcodes")

        return {
            "instruction_words": len(instr_words),
            "output_tokens": len(output_tokens),
            "output_unk_ratio": output_unk_ratio,
            "issues": issues,
            "is_valid": len(issues) == 0,
        }

    def filter_by_quality(
        self,
        samples: list[dict[str, Any]],
        verbose: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Filter samples by quality metrics.

        Returns:
            (passed, failed) tuple of sample lists
        """
        passed = []
        failed = []

        for sample in samples:
            analysis = self.analyze_sample(sample)
            if analysis["is_valid"]:
                passed.append(sample)
            else:
                if verbose:
                    sample["_quality_issues"] = analysis["issues"]
                failed.append(sample)

        return passed, failed

    # =========================================================================
    # Deduplication
    # =========================================================================

    def find_duplicates(
        self,
        samples: list[dict[str, Any]],
        field: str = "output",
        threshold: float | None = None,
    ) -> list[set[int]]:
        """Find groups of duplicate samples.

        Args:
            samples: List of training samples
            field: Field to compare ('output' or 'instruction')
            threshold: Similarity threshold (default from config)

        Returns:
            List of sets, each containing indices of duplicate samples
        """
        threshold = threshold or self.config.similarity_threshold

        # Get embeddings
        embeddings = []
        for sample in samples:
            text = sample.get(field, "")
            embeddings.append(self.get_embedding(text))

        # Find duplicates
        n = len(samples)
        visited = set()
        duplicate_groups = []

        for i in range(n):
            if i in visited:
                continue

            group = {i}
            for j in range(i + 1, n):
                if j in visited:
                    continue

                sim = self.cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    group.add(j)
                    visited.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
            visited.add(i)

        return duplicate_groups

    def deduplicate(
        self,
        samples: list[dict[str, Any]],
        field: str = "output",
        keep: str = "first",  # first, longest, shortest
    ) -> list[dict[str, Any]]:
        """Remove duplicate samples.

        Args:
            samples: List of training samples
            field: Field to compare
            keep: Which duplicate to keep

        Returns:
            Deduplicated sample list
        """
        duplicate_groups = self.find_duplicates(samples, field)

        # Mark indices to remove
        to_remove = set()
        for group in duplicate_groups:
            group_list = sorted(group)

            if keep == "first":
                # Keep first, remove rest
                to_remove.update(group_list[1:])
            elif keep == "longest":
                # Keep longest output
                lengths = [(i, len(samples[i].get("output", ""))) for i in group_list]
                lengths.sort(key=lambda x: -x[1])
                to_remove.update(i for i, _ in lengths[1:])
            elif keep == "shortest":
                lengths = [(i, len(samples[i].get("output", ""))) for i in group_list]
                lengths.sort(key=lambda x: x[1])
                to_remove.update(i for i, _ in lengths[1:])

        return [s for i, s in enumerate(samples) if i not in to_remove]

    # =========================================================================
    # Diversity Sampling
    # =========================================================================

    def cluster_samples(
        self,
        samples: list[dict[str, Any]],
        n_clusters: int | None = None,
        field: str = "output",
    ) -> list[list[int]]:
        """Cluster samples by semantic similarity.

        Args:
            samples: List of training samples
            n_clusters: Number of clusters (default from config)
            field: Field to embed

        Returns:
            List of clusters, each containing sample indices
        """
        from sklearn.cluster import KMeans

        n_clusters = n_clusters or self.config.num_clusters
        n_clusters = min(n_clusters, len(samples))

        # Get embeddings
        embeddings = []
        for sample in samples:
            text = sample.get(field, "")
            embeddings.append(self.get_embedding(text))

        X = np.array(embeddings)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group by cluster
        clusters: list[list[int]] = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        return clusters

    def sample_diverse(
        self,
        samples: list[dict[str, Any]],
        n_samples: int | None = None,
        field: str = "output",
    ) -> list[dict[str, Any]]:
        """Sample diverse subset using clustering.

        Ensures representation from different semantic regions.

        Args:
            samples: List of training samples
            n_samples: Target number of samples
            field: Field to embed

        Returns:
            Diverse sample subset
        """
        if n_samples is None:
            n_samples = self.config.num_clusters * self.config.samples_per_cluster

        n_clusters = min(self.config.num_clusters, len(samples) // 2)
        samples_per = max(1, n_samples // n_clusters)

        clusters = self.cluster_samples(samples, n_clusters, field)

        # Sample from each cluster
        selected_indices = set()
        for cluster in clusters:
            # Take up to samples_per from each cluster
            for idx in cluster[:samples_per]:
                selected_indices.add(idx)
                if len(selected_indices) >= n_samples:
                    break
            if len(selected_indices) >= n_samples:
                break

        return [samples[i] for i in sorted(selected_indices)]


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_dataset(
    input_path: Path,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Analyze a training dataset for quality metrics.

    Args:
        input_path: Path to JSONL file
        tokenizer_path: Path to saved tokenizer

    Returns:
        Analysis summary
    """
    config = EncoderConfig(tokenizer_path=tokenizer_path)
    processor = EncoderDataProcessor(config=config)

    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Analyze
    issues_count: dict[str, int] = {}
    valid_count = 0

    for sample in samples:
        analysis = processor.analyze_sample(sample)
        if analysis["is_valid"]:
            valid_count += 1
        for issue in analysis["issues"]:
            issues_count[issue] = issues_count.get(issue, 0) + 1

    return {
        "total_samples": len(samples),
        "valid_samples": valid_count,
        "valid_ratio": valid_count / len(samples) if samples else 0,
        "issues": issues_count,
    }


def deduplicate_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path | None = None,
    threshold: float = 0.95,
) -> dict[str, int]:
    """Deduplicate a training dataset.

    Args:
        input_path: Input JSONL path
        output_path: Output JSONL path
        tokenizer_path: Path to saved tokenizer
        threshold: Similarity threshold

    Returns:
        Stats dict
    """
    config = EncoderConfig(
        tokenizer_path=tokenizer_path,
        similarity_threshold=threshold,
    )
    processor = EncoderDataProcessor(config=config)

    # Load
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    original_count = len(samples)

    # Deduplicate
    deduped = processor.deduplicate(samples, keep="longest")

    # Save
    with open(output_path, "w") as f:
        for sample in deduped:
            f.write(json.dumps(sample) + "\n")

    return {
        "original": original_count,
        "deduplicated": len(deduped),
        "removed": original_count - len(deduped),
    }


def sample_diverse_dataset(
    input_path: Path,
    output_path: Path,
    n_samples: int,
    tokenizer_path: Path | None = None,
) -> dict[str, int]:
    """Sample diverse subset from dataset.

    Args:
        input_path: Input JSONL path
        output_path: Output JSONL path
        n_samples: Target sample count
        tokenizer_path: Path to saved tokenizer

    Returns:
        Stats dict
    """
    config = EncoderConfig(tokenizer_path=tokenizer_path)
    processor = EncoderDataProcessor(config=config)

    # Load
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Sample
    diverse = processor.sample_diverse(samples, n_samples)

    # Save
    with open(output_path, "w") as f:
        for sample in diverse:
            f.write(json.dumps(sample) + "\n")

    return {
        "original": len(samples),
        "sampled": len(diverse),
    }
