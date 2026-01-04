"""ASM-ELECTRA: Discriminator for 65816 assembly quality filtering.

This module provides tools for training an ELECTRA-based discriminator
to distinguish real assembly code from LLM-generated errors, enabling:
- Pre-training data filtering
- Reward modeling for RLHF
- Inference-time rejection sampling
"""

from .electra import ASMElectra, ElectraConfig
from .fake_generators import (
    FakeGenerator,
    SyntaxErrorGenerator,
    AddressingErrorGenerator,
    OpcodeSwapGenerator,
    CompositeGenerator,
)
from .data import ElectraDataset, ElectraSample, create_training_data
from .filter import SampleFilter, FilterConfig

__all__ = [
    # Core
    "ASMElectra",
    "ElectraConfig",
    # Generators
    "FakeGenerator",
    "SyntaxErrorGenerator",
    "AddressingErrorGenerator",
    "OpcodeSwapGenerator",
    "CompositeGenerator",
    # Data
    "ElectraDataset",
    "ElectraSample",
    "create_training_data",
    # Filtering
    "SampleFilter",
    "FilterConfig",
]
