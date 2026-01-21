"""Training data generators for AFS."""

from .asar_validator import (
    AsarValidator,
    AsarValidatorConfig,
    ValidationResult,
    ValidationStats,
    check_asar_available,
    validate_training_data,
)
from .asm_augment import (
    AsmAugmentConfig,
    AsmAugmentGenerator,
    detect_category,
    generate_paraphrases,
)
from .base import (
    BaseGenerator,
    GenerationResult,
    TrainingSample,
    clean_instruction,
    clean_sample_instruction,
    is_malformed_output,
    read_jsonl,
    write_jsonl,
)
from .cot import (
    CotConfig,
    CotFormat,
    CotGenerator,
    GeminiClient,
    LLMClient,
    format_cot_sample,
)
from .curriculum_generator import (
    CurriculumGenerator,
    CurriculumTemplate,
    Difficulty,
    ExpertDomain,
    GenerationProgress,
    ScaleConfig,
    create_curriculum_generator,
)
from .data_cleaner import (
    CleaningStats,
    clean_dataset,
    clean_sample,
)
from .knowledge_generator import (
    KnowledgeAwareGenerator,
    KnowledgeGeneratorConfig,
    create_knowledge_generator,
)
from .model_generator import (
    APIBackend,
    HuggingFaceBackend,
    LlamaCppBackend,
    MLXBackend,
    ModelBackend,
    ModelGenerator,
    ModelGeneratorConfig,
    ModelType,
    available_backends,
    create_generator,
    register_backend,
)

__all__ = [
    # Base classes
    "BaseGenerator",
    "GenerationResult",
    "TrainingSample",
    # I/O utilities
    "read_jsonl",
    "write_jsonl",
    # Cleaning utilities (from base.py)
    "clean_instruction",
    "clean_sample_instruction",
    "is_malformed_output",
    # ASM augmentation
    "AsmAugmentConfig",
    "AsmAugmentGenerator",
    "detect_category",
    "generate_paraphrases",
    # Chain of Thought
    "CotConfig",
    "CotGenerator",
    "CotFormat",
    "format_cot_sample",
    "GeminiClient",
    "LLMClient",
    # Data cleaning (batch operations)
    "CleaningStats",
    "clean_dataset",
    "clean_sample",
    # Asar validation
    "AsarValidator",
    "AsarValidatorConfig",
    "ValidationResult",
    "ValidationStats",
    "check_asar_available",
    "validate_training_data",
    # Model-based generation
    "ModelGenerator",
    "ModelGeneratorConfig",
    "ModelType",
    "ModelBackend",
    "MLXBackend",
    "LlamaCppBackend",
    "HuggingFaceBackend",
    "APIBackend",
    "create_generator",
    "available_backends",
    "register_backend",
    # Knowledge-aware generation
    "KnowledgeAwareGenerator",
    "KnowledgeGeneratorConfig",
    "create_knowledge_generator",
    # Curriculum-based generation (scaled)
    "CurriculumGenerator",
    "CurriculumTemplate",
    "Difficulty",
    "ExpertDomain",
    "GenerationProgress",
    "ScaleConfig",
    "create_curriculum_generator",
]
