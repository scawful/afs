"""Distillation pipeline for training student models from teacher ensemble.

This module provides:
- Multi-provider teacher model wrappers (OpenAI, Google, Anthropic)
- Distillation data generation with provider rotation
- Student training infrastructure
- Capability evaluation
"""

from .teacher import (
    TeacherModel,
    OpenAITeacher,
    GoogleTeacher,
    AnthropicTeacher,
    TeacherEnsemble,
    TeacherResponse,
    ProviderConfig,
    Provider,
)
from .data_gen import (
    DistillationDataGenerator,
    DistillationSample,
    DistillationConfig,
    GenerationProgress,
)

__all__ = [
    # Teacher models
    "TeacherModel",
    "OpenAITeacher",
    "GoogleTeacher",
    "AnthropicTeacher",
    "TeacherEnsemble",
    "TeacherResponse",
    "ProviderConfig",
    "Provider",
    # Data generation
    "DistillationDataGenerator",
    "DistillationSample",
    "DistillationConfig",
    "GenerationProgress",
]
