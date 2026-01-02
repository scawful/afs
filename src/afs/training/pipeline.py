"""Full data processing pipeline for training data preparation.

Combines all preprocessing steps into a single pipeline:
1. Load raw data from JSONL files
2. Expand tokenizer vocabulary
3. Extract entities (populate kg_entities)
4. Score quality (populate quality_score)
5. Filter low-quality samples
6. Apply augmentation (Phase 1 & 2)
7. Deduplicate
8. Split into train/val/test
9. Export in multiple formats
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .encoder_utils import EncoderConfig, EncoderDataProcessor
from .scoring import QualityScorer, ScoringConfig, ScoringWeights
from .splitter import DatasetSplitter


@dataclass
class PipelineConfig:
    """Configuration for the full data pipeline."""

    # Input
    input_paths: list[Path] = field(default_factory=list)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./pipeline_output"))

    # Tokenizer
    tokenizer_path: Path | None = None
    expand_vocab: bool = True
    min_token_frequency: int = 2
    max_vocab_size: int = 50000

    # Entity extraction
    extract_entities: bool = True
    include_hardware_entities: bool = True

    # Quality scoring
    score_quality: bool = True
    min_quality_score: float = 0.5
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    electra_model_path: Path | None = None

    # Augmentation
    apply_phase1_augment: bool = True
    phase1_paraphrase_count: int = 3
    apply_phase2_augment: bool = True

    # Deduplication
    deduplicate: bool = True
    dedupe_threshold: float = 0.95

    # Splitting
    split_data: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Export formats
    export_formats: list[str] = field(default_factory=lambda: ["jsonl"])

    # Processing
    batch_size: int = 100
    verbose: bool = True


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    # Counts
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    augmented_count: int = 0
    dedupe_removed: int = 0

    # Quality stats
    mean_quality_score: float = 0.0
    min_quality_score: float = 0.0
    max_quality_score: float = 0.0

    # Entity stats
    total_entities: int = 0
    known_entities: int = 0
    entity_coverage: float = 0.0

    # Output paths
    output_paths: dict[str, Path] = field(default_factory=dict)

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "filtered_count": self.filtered_count,
            "augmented_count": self.augmented_count,
            "dedupe_removed": self.dedupe_removed,
            "mean_quality_score": self.mean_quality_score,
            "min_quality_score": self.min_quality_score,
            "max_quality_score": self.max_quality_score,
            "total_entities": self.total_entities,
            "known_entities": self.known_entities,
            "entity_coverage": self.entity_coverage,
            "output_paths": {k: str(v) for k, v in self.output_paths.items()},
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class DataPipeline:
    """Full data processing pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._tokenizer = None
        self._quality_scorer = None
        self._entity_extractor = None

    def run(self) -> PipelineResult:
        """Execute the full pipeline.

        Returns:
            PipelineResult with statistics and output paths
        """
        from afs.generators.base import TrainingSample

        result = PipelineResult()
        result.start_time = datetime.now().isoformat()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self._log("=" * 60)
        self._log("AFS Data Pipeline")
        self._log("=" * 60)

        # Stage 1: Load raw data
        self._log("\n[1/9] Loading raw data...")
        samples = self._load_samples()
        result.input_count = len(samples)
        self._log(f"  Loaded {len(samples)} samples from {len(self.config.input_paths)} files")

        if not samples:
            result.errors.append("No samples loaded")
            return result

        # Stage 2: Expand tokenizer vocabulary
        if self.config.expand_vocab:
            self._log("\n[2/9] Expanding tokenizer vocabulary...")
            self._expand_tokenizer(samples)
        else:
            self._log("\n[2/9] Skipping vocabulary expansion")

        # Stage 3: Extract entities
        if self.config.extract_entities:
            self._log("\n[3/9] Extracting entities...")
            entity_stats = self._extract_entities(samples)
            result.total_entities = entity_stats["total"]
            result.known_entities = entity_stats["known"]
            result.entity_coverage = entity_stats["coverage"]
            self._log(f"  Found {result.total_entities} entities, {result.known_entities} known ({100*result.entity_coverage:.1f}%)")
        else:
            self._log("\n[3/9] Skipping entity extraction")

        # Stage 4: Score quality
        if self.config.score_quality:
            self._log("\n[4/9] Scoring quality...")
            quality_stats = self._score_quality(samples)
            result.mean_quality_score = quality_stats["mean"]
            result.min_quality_score = quality_stats["min"]
            result.max_quality_score = quality_stats["max"]
            self._log(f"  Mean score: {result.mean_quality_score:.3f} (range: {result.min_quality_score:.3f} - {result.max_quality_score:.3f})")
        else:
            self._log("\n[4/9] Skipping quality scoring")

        # Stage 5: Filter low-quality
        if self.config.score_quality and self.config.min_quality_score > 0:
            self._log("\n[5/9] Filtering low-quality samples...")
            before = len(samples)
            samples = [s for s in samples if s.quality_score >= self.config.min_quality_score]
            result.filtered_count = before - len(samples)
            self._log(f"  Filtered {result.filtered_count} samples below {self.config.min_quality_score}")
            self._log(f"  Remaining: {len(samples)} samples")
        else:
            self._log("\n[5/9] Skipping quality filtering")

        # Stage 6: Apply augmentation
        if self.config.apply_phase1_augment or self.config.apply_phase2_augment:
            self._log("\n[6/9] Applying augmentation...")
            before = len(samples)
            samples = self._augment_samples(samples)
            result.augmented_count = len(samples) - before
            self._log(f"  Generated {result.augmented_count} augmented samples")
            self._log(f"  Total: {len(samples)} samples")
        else:
            self._log("\n[6/9] Skipping augmentation")

        # Stage 7: Deduplicate
        if self.config.deduplicate:
            self._log("\n[7/9] Deduplicating...")
            before = len(samples)
            samples = self._deduplicate(samples)
            result.dedupe_removed = before - len(samples)
            self._log(f"  Removed {result.dedupe_removed} duplicates")
            self._log(f"  Remaining: {len(samples)} samples")
        else:
            self._log("\n[7/9] Skipping deduplication")

        # Stage 8: Split into train/val/test
        if self.config.split_data:
            self._log("\n[8/9] Splitting data...")
            splits = self._split_data(samples)
            self._log(f"  Train: {len(splits['train'])} samples")
            self._log(f"  Val: {len(splits['val'])} samples")
            self._log(f"  Test: {len(splits['test'])} samples")
        else:
            self._log("\n[8/9] Skipping data split")
            splits = {"all": samples}

        # Stage 9: Export
        self._log("\n[9/9] Exporting...")
        result.output_paths = self._export(splits)
        for name, path in result.output_paths.items():
            self._log(f"  {name}: {path}")

        # Finalize
        result.output_count = len(samples)
        result.end_time = datetime.now().isoformat()

        # Calculate duration
        start = datetime.fromisoformat(result.start_time)
        end = datetime.fromisoformat(result.end_time)
        result.duration_seconds = (end - start).total_seconds()

        # Save pipeline result
        result_path = self.config.output_dir / "pipeline_result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self._log("\n" + "=" * 60)
        self._log("Pipeline Complete")
        self._log("=" * 60)
        self._log(f"  Input: {result.input_count} samples")
        self._log(f"  Output: {result.output_count} samples")
        self._log(f"  Duration: {result.duration_seconds:.1f} seconds")

        return result

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)

    def _load_samples(self) -> list:
        """Load samples from input files."""
        from afs.generators.base import TrainingSample

        samples = []
        for path in self.config.input_paths:
            path = Path(path)
            if not path.exists():
                self._log(f"  Warning: {path} not found, skipping")
                continue

            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            samples.append(TrainingSample.from_dict(data))
                        except json.JSONDecodeError as e:
                            self._log(f"  Warning: Invalid JSON in {path}: {e}")

        return samples

    def _expand_tokenizer(self, samples: list) -> None:
        """Expand tokenizer vocabulary from samples."""
        from afs.tokenizer import ASMTokenizer

        # Load or create tokenizer
        if self.config.tokenizer_path and self.config.tokenizer_path.exists():
            tokenizer = ASMTokenizer.load(self.config.tokenizer_path)
            self._log(f"  Loaded tokenizer with {len(tokenizer)} tokens")
        else:
            tokenizer = ASMTokenizer()
            self._log(f"  Created new tokenizer with {len(tokenizer)} base tokens")

        # Extract texts
        texts = [s.output for s in samples]

        # Train
        before = len(tokenizer)
        tokenizer.train_on_corpus(
            texts,
            min_frequency=self.config.min_token_frequency,
            max_vocab_size=self.config.max_vocab_size,
        )
        added = len(tokenizer) - before
        self._log(f"  Added {added} tokens (total: {len(tokenizer)})")

        # Save
        tokenizer_path = self.config.output_dir / "tokenizer"
        tokenizer.save(tokenizer_path)
        self._log(f"  Saved tokenizer to {tokenizer_path}")

        self._tokenizer = tokenizer

    def _extract_entities(self, samples: list) -> dict[str, Any]:
        """Extract entities from all samples."""
        from afs.knowledge import EntityExtractor

        extractor = EntityExtractor(
            include_hardware=self.config.include_hardware_entities
        )

        total = 0
        known = 0

        for sample in samples:
            sample.populate_kg_entities(extractor)
            result = extractor.extract(sample.output)
            total += len(result.entities)
            known += result.known_count

        return {
            "total": total,
            "known": known,
            "coverage": known / total if total > 0 else 0.0,
        }

    def _score_quality(self, samples: list) -> dict[str, float]:
        """Score quality of all samples."""
        config = ScoringConfig(
            weights=self.config.scoring_weights,
            electra_model_path=self.config.electra_model_path,
        )
        scorer = QualityScorer(config=config)
        scores = scorer.score_batch(samples, update_samples=True)

        score_values = [s.overall for s in scores]
        return {
            "mean": sum(score_values) / len(score_values) if score_values else 0.0,
            "min": min(score_values) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
        }

    def _augment_samples(self, samples: list) -> list:
        """Apply augmentation to samples."""
        from afs.generators.asm_augment import (
            AsmAugmentGenerator,
            AsmAugmentConfig,
            Phase2Augmenter,
            Phase2AugmentConfig,
        )

        result = list(samples)  # Keep originals

        # Phase 1: Instruction paraphrasing
        if self.config.apply_phase1_augment:
            aug_config = AsmAugmentConfig(
                paraphrase_count=self.config.phase1_paraphrase_count,
                include_original=False,  # We already have originals
            )
            generator = AsmAugmentGenerator(input_path=None, config=aug_config)

            for sample in samples:
                augmented = generator._augment_sample(sample)
                result.extend(augmented)

        # Phase 2: Register swap, address, style variations
        if self.config.apply_phase2_augment:
            augmenter = Phase2Augmenter(Phase2AugmentConfig())

            for sample in samples:
                augmented = augmenter.augment(sample)
                result.extend(augmented)

        return result

    def _deduplicate(self, samples: list) -> list:
        """Deduplicate samples."""
        config = EncoderConfig(
            tokenizer_path=self.config.tokenizer_path or (self.config.output_dir / "tokenizer"),
            similarity_threshold=self.config.dedupe_threshold,
        )
        processor = EncoderDataProcessor(config=config, tokenizer=self._tokenizer)
        return processor.deduplicate(samples, keep="longest")

    def _split_data(self, samples: list) -> dict[str, list]:
        """Split samples into train/val/test."""
        splitter = DatasetSplitter(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )
        result = splitter.split(samples)
        return {
            "train": result.train,
            "val": result.val,
            "test": result.test,
        }

    def _export(self, splits: dict[str, list]) -> dict[str, Path]:
        """Export splits to files."""
        output_paths = {}

        for split_name, samples in splits.items():
            if not samples:
                continue

            for fmt in self.config.export_formats:
                if fmt == "jsonl":
                    path = self.config.output_dir / f"{split_name}.jsonl"
                    with open(path, "w") as f:
                        for sample in samples:
                            f.write(json.dumps(sample.to_dict()) + "\n")
                    output_paths[f"{split_name}_jsonl"] = path

        return output_paths


def run_pipeline(
    input_paths: list[Path | str],
    output_dir: Path | str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Convenience function to run the pipeline.

    Args:
        input_paths: List of input JSONL files
        output_dir: Output directory
        config: Optional pipeline configuration

    Returns:
        PipelineResult with statistics
    """
    if config is None:
        config = PipelineConfig()

    config.input_paths = [Path(p) for p in input_paths]
    config.output_dir = Path(output_dir)

    pipeline = DataPipeline(config)
    return pipeline.run()
