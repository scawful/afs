"""Training utilities for AFS domain-specific models.

Part of the AFS (Agentic File System) framework, this module provides:
- Training data preparation and format conversion
- Model experiment tracking and A/B testing
- Dataset splitting with stratification
- ASM encoder training with custom tokenizers

Used by AFS agents for training domain-specific models for
ALTTP/SNES assembly understanding and generation tasks.

See also:
- afs.tokenizer: Custom assembly tokenizer
- afs.generators: Training data generation
- afs.discriminator: Quality filtering models
"""

from .config import (
    Framework,
    CotInclusionMode,
    LoRAConfig,
    DatasetConfig,
    TrainingConfig,
    MODEL_PRESETS,
    get_model_preset,
)
from .splitter import DatasetSplitter, SplitResult, split_dataset
from .registry import (
    ExperimentMetrics,
    ModelExperiment,
    ModelRegistry,
)
from .converters import (
    BaseConverter,
    MLXConverter,
    MLXCompletionConverter,
    AlpacaConverter,
    ChatMLConverter,
    ShareGPTConverter,
    UnslothThinkingConverter,
    LlamaCppConverter,
    GGUFTrainConverter,
    get_converter,
)
from .asm_trainer import (
    ASMTrainerConfig,
    ASMDataset,
    ASMTrainer,
    train_asm_encoder,
)
from .encoder_utils import (
    EncoderConfig,
    EncoderDataProcessor,
    analyze_dataset,
    deduplicate_dataset,
    sample_diverse_dataset,
)
from .scoring import (
    ScoringWeights,
    ScoringConfig,
    QualityScore,
    QualityScorer,
    build_scoring_config,
    score_samples,
    score_jsonl,
    analyze_scores,
)
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    DataPipeline,
    run_pipeline,
)
from .memory_export import (
    MemoryExportResult,
    export_memory_to_dataset,
)
from .history_export import (
    HistoryExportResult,
    export_history_to_dataset,
)
from .antigravity_export import (
    AntigravityExportResult,
    export_antigravity_to_dataset,
)
from .gemini_export import (
    GeminiExportResult,
    export_gemini_logs_to_dataset,
)
from .claude_export import (
    ClaudeExportResult,
    export_claude_logs_to_dataset,
)
from .rebalance import (
    RebalanceResult,
    rebalance_dataset,
)
from .codex_export import (
    CodexExportResult,
    CodexHistoryImportResult,
    export_codex_logs_to_dataset,
    import_codex_logs_to_history,
)

__all__ = [
    # Config
    "Framework",
    "CotInclusionMode",
    "LoRAConfig",
    "DatasetConfig",
    "TrainingConfig",
    "MODEL_PRESETS",
    "get_model_preset",
    # Splitting
    "DatasetSplitter",
    "SplitResult",
    "split_dataset",
    # Registry
    "ExperimentMetrics",
    "ModelExperiment",
    "ModelRegistry",
    # Converters
    "BaseConverter",
    "MLXConverter",
    "MLXCompletionConverter",
    "AlpacaConverter",
    "ChatMLConverter",
    "ShareGPTConverter",
    "UnslothThinkingConverter",
    "LlamaCppConverter",
    "GGUFTrainConverter",
    "get_converter",
    # ASM Encoder Training
    "ASMTrainerConfig",
    "ASMDataset",
    "ASMTrainer",
    "train_asm_encoder",
    # Encoder-based Data Utils
    "EncoderConfig",
    "EncoderDataProcessor",
    "analyze_dataset",
    "deduplicate_dataset",
    "sample_diverse_dataset",
    # Quality Scoring
    "ScoringWeights",
    "ScoringConfig",
    "QualityScore",
    "QualityScorer",
    "build_scoring_config",
    "score_samples",
    "score_jsonl",
    "analyze_scores",
    # Pipeline
    "PipelineConfig",
    "PipelineResult",
    "DataPipeline",
    "run_pipeline",
    # Memory export
    "MemoryExportResult",
    "export_memory_to_dataset",
    # History export
    "HistoryExportResult",
    "export_history_to_dataset",
    # Antigravity export
    "AntigravityExportResult",
    "export_antigravity_to_dataset",
    # Gemini export
    "GeminiExportResult",
    "export_gemini_logs_to_dataset",
    # Claude export
    "ClaudeExportResult",
    "export_claude_logs_to_dataset",
    # Rebalance
    "RebalanceResult",
    "rebalance_dataset",
    # Codex export / history import
    "CodexExportResult",
    "CodexHistoryImportResult",
    "export_codex_logs_to_dataset",
    "import_codex_logs_to_history",
]
