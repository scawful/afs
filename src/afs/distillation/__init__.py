"""Distillation pipeline for training student models from teacher ensemble.

This module provides:
- Multi-provider teacher model wrappers (OpenAI, Google, Anthropic)
- Distillation data generation with provider rotation
- Student training infrastructure
- Capability evaluation
"""

from .data_gen import (
    DistillationConfig,
    DistillationDataGenerator,
    DistillationSample,
    GenerationProgress,
)
from .teacher import (
    AnthropicTeacher,
    GoogleTeacher,
    OpenAITeacher,
    Provider,
    ProviderConfig,
    TeacherEnsemble,
    TeacherModel,
    TeacherResponse,
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
