"""Training data generators for AFS.

Core provides generic generator infrastructure (BaseGenerator, TrainingSample, CoT, model backends).
Domain-specific generators (ASM augment, ASAR, curriculum, knowledge) are extension-owned
and available when a companion extension repo is installed.
"""

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
from .data_cleaner import (
    CleaningStats,
    clean_dataset,
    clean_sample,
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
]

# Domain-specific generators (extension-owned, available with a companion extension repo)
try:
    from .asar_validator import (
        AsarValidator,
        AsarValidatorConfig,
        ValidationResult,
        ValidationStats,
        check_asar_available,
        validate_training_data,
    )

    __all__.extend([
        "AsarValidator", "AsarValidatorConfig", "ValidationResult",
        "ValidationStats", "check_asar_available", "validate_training_data",
    ])
except Exception:
    pass

try:
    from .asm_augment import (
        AsmAugmentConfig,
        AsmAugmentGenerator,
        detect_category,
        generate_paraphrases,
    )

    __all__.extend([
        "AsmAugmentConfig", "AsmAugmentGenerator",
        "detect_category", "generate_paraphrases",
    ])
except Exception:
    pass

try:
    from .curriculum_generator import (
        CurriculumGenerator,
        CurriculumTemplate,
        Difficulty,
        ExpertDomain,
        GenerationProgress,
        ScaleConfig,
        create_curriculum_generator,
    )

    __all__.extend([
        "CurriculumGenerator", "CurriculumTemplate", "Difficulty",
        "ExpertDomain", "GenerationProgress", "ScaleConfig",
        "create_curriculum_generator",
    ])
except Exception:
    pass

try:
    from .knowledge_generator import (
        KnowledgeAwareGenerator,
        KnowledgeGeneratorConfig,
        create_knowledge_generator,
    )

    __all__.extend([
        "KnowledgeAwareGenerator", "KnowledgeGeneratorConfig",
        "create_knowledge_generator",
    ])
except Exception:
    pass
