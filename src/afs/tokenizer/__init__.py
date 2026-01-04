"""65816 Assembly Tokenizer for AFS domain capabilities.

Part of the AFS (Agentic File System) domain-specific tooling for
ALTTP/SNES assembly tasks. This module provides semantic tokenization
that preserves opcodes, addresses, and indexed addressing modes.

Used by AFS agents for:
- Assembly code understanding and generation
- Training domain-specific encoder models
- Integration with the afs.knowledge and afs.training modules

Usage:
    from afs.tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer()
    encoded = tokenizer.encode("LDA $7F00,X")
    decoded = tokenizer.decode(encoded["input_ids"])
"""

from .asm_tokenizer import ASMTokenizer, create_tokenizer
from .vocab import (
    BASE_VOCAB,
    OPCODES,
    REGISTERS,
    DIRECTIVES,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)
from .pretokenizer import (
    AssemblyPreTokenizer,
    Token,
    normalize_token,
    split_address,
)

__all__ = [
    # Main tokenizer
    "ASMTokenizer",
    "create_tokenizer",
    # Pre-tokenizer
    "AssemblyPreTokenizer",
    "Token",
    "normalize_token",
    "split_address",
    # Vocabulary
    "BASE_VOCAB",
    "OPCODES",
    "REGISTERS",
    "DIRECTIVES",
    "SPECIAL_TOKENS",
    "VOCAB_SIZE",
]
