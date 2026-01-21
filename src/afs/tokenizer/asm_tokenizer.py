"""65816 Assembly Tokenizer for AFS domain capabilities.

A custom tokenizer optimized for SNES/65816 assembly code, designed
for use within the AFS (Agentic File System) framework. Preserves
semantic units like opcodes, addresses, and labels.

This tokenizer is part of AFS's domain-specific tooling for ALTTP/SNES
assembly tasks, providing:
- Semantic preservation (opcodes, addresses as single tokens)
- HuggingFace-compatible interface for transformer training
- Integration with afs.training for encoder model training
- Vocabulary expansion via corpus training

See also:
- afs.knowledge: ALTTP address tables and domain knowledge
- afs.training: Model training utilities
- afs.generators: Training data generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .pretokenizer import (
    AssemblyPreTokenizer,
    Token,
    normalize_token,
    split_address,
)
from .vocab import BASE_VOCAB, SPECIAL_TOKENS

# Type aliases for HuggingFace compatibility
TextInput = str | list[str]
BatchEncoding = dict[str, Any]


class ASMTokenizer:
    """Tokenizer for 65816 assembly code.

    Features:
    - Semantic tokenization (opcodes, addresses as units)
    - Fixed vocabulary for known assembly constructs
    - BPE fallback for unknown tokens (labels, comments)
    - HuggingFace-compatible interface
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        split_addresses: bool = False,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ):
        """Initialize tokenizer.

        Args:
            vocab: Custom vocabulary (uses default if None).
            split_addresses: If True, split addresses into prefix + digits.
            max_length: Maximum sequence length.
            add_special_tokens: Add [CLS] and [SEP] tokens.
        """
        self.vocab = vocab or BASE_VOCAB.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.split_addresses = split_addresses
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        # Pre-tokenizer
        self.pretokenizer = AssemblyPreTokenizer(preserve_whitespace=False)

        # Special token IDs
        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.unk_token_id = self.vocab.get("[UNK]", 1)
        self.cls_token_id = self.vocab.get("[CLS]", 2)
        self.sep_token_id = self.vocab.get("[SEP]", 3)
        self.mask_token_id = self.vocab.get("[MASK]", 4)

        # Track unknown tokens for vocabulary expansion
        self._unknown_tokens: dict[str, int] = {}

        # HuggingFace-compatible attributes
        self.model_max_length = max_length
        self.padding_side = "right"
        self.truncation_side = "right"

    # =========================================================================
    # HuggingFace-compatible properties
    # =========================================================================

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def __len__(self) -> int:
        """Return vocabulary size (HuggingFace compatibility)."""
        return len(self.vocab)

    @property
    def is_fast(self) -> bool:
        """Return False - this is not a Rust-backed tokenizer."""
        return False

    @property
    def pad_token(self) -> str:
        return "[PAD]"

    @property
    def unk_token(self) -> str:
        return "[UNK]"

    @property
    def cls_token(self) -> str:
        return "[CLS]"

    @property
    def sep_token(self) -> str:
        return "[SEP]"

    @property
    def mask_token(self) -> str:
        return "[MASK]"

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary dictionary."""
        return self.vocab.copy()

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """Convert token(s) to ID(s)."""
        if isinstance(tokens, str):
            return self._token_to_id(tokens)
        return [self._token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        """Convert ID(s) to token(s)."""
        if isinstance(ids, int):
            return self.id_to_token.get(ids, "[UNK]")
        return [self.id_to_token.get(i, "[UNK]") for i in ids]

    def _token_to_id(self, token_text: str) -> int:
        """Convert token text to ID."""
        # Try direct lookup
        if token_text in self.vocab:
            return self.vocab[token_text]

        # Try uppercase (for opcodes)
        upper = token_text.upper()
        if upper in self.vocab:
            return self.vocab[upper]

        # Try with normalized suffix (LDA.B -> LDA.b)
        if '.' in token_text:
            base, suffix = token_text.rsplit('.', 1)
            normalized = base.upper() + '.' + suffix.lower()
            if normalized in self.vocab:
                return self.vocab[normalized]

        # Try lowercase (for directives)
        lower = token_text.lower()
        if lower in self.vocab:
            return self.vocab[lower]

        # Track unknown
        self._unknown_tokens[token_text] = self._unknown_tokens.get(token_text, 0) + 1

        return self.unk_token_id

    def _tokenize_one(self, token: Token) -> list[int]:
        """Convert a pre-tokenized Token to ID(s)."""
        normalized = normalize_token(token)

        # Optionally split addresses for finer granularity
        if self.split_addresses and token.type in (
            'address_long', 'address_abs', 'address_dp', 'immediate', 'hex'
        ):
            parts = split_address(token)
            return [self._token_to_id(p) for p in parts]

        return [self._token_to_id(normalized)]

    def _tokenize_one_to_strings(self, token: Token) -> list[str]:
        """Convert a pre-tokenized Token to string token(s)."""
        normalized = normalize_token(token)

        if self.split_addresses and token.type in (
            'address_long', 'address_abs', 'address_dp', 'immediate', 'hex'
        ):
            return split_address(token)

        return [normalized]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into string tokens (HuggingFace compatibility).

        Args:
            text: Assembly code to tokenize.

        Returns:
            List of string tokens.
        """
        tokens = self.pretokenizer.tokenize(text)
        result = []
        for token in tokens:
            result.extend(self._tokenize_one_to_strings(token))
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool | None = None,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Encode text to token IDs.

        Args:
            text: Assembly code to encode.
            add_special_tokens: Add [CLS] and [SEP].
            max_length: Maximum length (uses default if None).
            padding: Pad to max_length.
            truncation: Truncate to max_length.
            return_tensors: "pt" for PyTorch, None for lists.

        Returns:
            Dictionary with input_ids and attention_mask.
        """
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        if max_length is None:
            max_length = self.max_length

        # Pre-tokenize
        tokens = self.pretokenizer.tokenize(text)

        # Convert to IDs
        input_ids = []
        for token in tokens:
            input_ids.extend(self._tokenize_one(token))

        # Add special tokens
        if add_special_tokens:
            input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]

        # Truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            # Ensure SEP at end if we had special tokens
            if add_special_tokens:
                input_ids[-1] = self.sep_token_id

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Padding
        if padding:
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            result = {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }

        return result

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Skip [PAD], [CLS], [SEP], etc.

        Returns:
            Decoded assembly code.
        """
        tokens = []

        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]

                # Skip special tokens if requested
                if skip_special_tokens and token in SPECIAL_TOKENS:
                    continue

                tokens.append(token)
            else:
                tokens.append("[UNK]")

        # Join with appropriate spacing
        return self._join_tokens(tokens)

    def _join_tokens(self, tokens: list[str]) -> str:
        """Join tokens back into assembly code with proper spacing."""
        if not tokens:
            return ""

        result = []
        prev_token = ""

        for token in tokens:
            # No space before punctuation/brackets
            if token in ",:;)]":
                result.append(token)
            # No space after open brackets or $/#/%
            elif prev_token in "([#$%":
                result.append(token)
            # No space before/after index markers
            elif token.startswith(",") or prev_token.endswith(","):
                result.append(token)
            # No space after $ or # (for addresses)
            elif token in "$#%":
                if result:
                    result.append(" " + token)
                else:
                    result.append(token)
            # Hex digits after $ should not have space
            elif prev_token == "$" or (len(prev_token) == 1 and prev_token in "0123456789ABCDEFabcdef"):
                # Check if current is also a hex digit or we're building an address
                if len(token) == 1 and token in "0123456789ABCDEFabcdef":
                    result.append(token)
                else:
                    result.append(" " + token) if result else result.append(token)
            # Space between most tokens
            elif result:
                result.append(" " + token)
            else:
                result.append(token)

            prev_token = token

        return "".join(result)

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | list[str] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Make tokenizer callable like HuggingFace tokenizers.

        Args:
            text: Text or list of texts to encode.
            text_pair: Second text for sequence pairs (not used for assembly).
            add_special_tokens: Add [CLS] and [SEP] tokens.
            padding: Pad sequences to max_length.
            truncation: Truncate sequences to max_length.
            max_length: Maximum sequence length.
            return_tensors: "pt" for PyTorch tensors.
            return_attention_mask: Include attention mask in output.

        Returns:
            Dictionary with input_ids and attention_mask.
        """
        # Normalize padding argument
        do_padding = padding is True or padding == "max_length" or padding == "longest"

        if isinstance(text, str):
            return self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=do_padding,
                truncation=bool(truncation),
                return_tensors=return_tensors,
            )

        # Batch encoding
        encoded_batch = []
        for t in text:
            encoded = self.encode(
                t,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,  # We'll pad the batch together
                truncation=bool(truncation),
            )
            encoded_batch.append(encoded)

        # Find max length for batch padding
        if do_padding:
            batch_max = max(len(e["input_ids"]) for e in encoded_batch)
            target_length = max_length if max_length and padding == "max_length" else batch_max

            for encoded in encoded_batch:
                pad_length = target_length - len(encoded["input_ids"])
                if pad_length > 0:
                    encoded["input_ids"] = encoded["input_ids"] + [self.pad_token_id] * pad_length
                    encoded["attention_mask"] = encoded["attention_mask"] + [0] * pad_length

        results = {
            "input_ids": [e["input_ids"] for e in encoded_batch],
            "attention_mask": [e["attention_mask"] for e in encoded_batch],
        }

        if return_tensors == "pt":
            import torch
            results["input_ids"] = torch.tensor(results["input_ids"])
            results["attention_mask"] = torch.tensor(results["attention_mask"])

        return results

    @property
    def special_tokens_map(self) -> dict[str, str]:
        """Return map of special token names to values."""
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
        }

    @property
    def all_special_tokens(self) -> list[str]:
        """Return list of all special tokens."""
        return list(SPECIAL_TOKENS.keys())

    @property
    def all_special_ids(self) -> list[int]:
        """Return list of all special token IDs."""
        return list(SPECIAL_TOKENS.values())

    def get_unknown_tokens(self) -> dict[str, int]:
        """Get tokens that weren't in vocabulary (for expansion)."""
        return dict(sorted(
            self._unknown_tokens.items(),
            key=lambda x: -x[1]  # Sort by frequency descending
        ))

    def add_tokens(self, tokens: list[str]) -> int:
        """Add tokens to vocabulary.

        Args:
            tokens: Tokens to add.

        Returns:
            Number of tokens added.
        """
        added = 0
        next_id = max(self.vocab.values()) + 1

        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
                added += 1

        return added

    def save(self, path: str | Path) -> None:
        """Save tokenizer to directory.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=2)

        # Save config
        config = {
            "split_addresses": self.split_addresses,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(path / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ASMTokenizer:
        """Load tokenizer from directory.

        Args:
            path: Directory to load from.

        Returns:
            Loaded tokenizer.
        """
        path = Path(path)

        # Load vocabulary
        with open(path / "vocab.json") as f:
            vocab = json.load(f)

        # Load config
        with open(path / "tokenizer_config.json") as f:
            config = json.load(f)

        return cls(
            vocab=vocab,
            split_addresses=config.get("split_addresses", False),
            max_length=config.get("max_length", 512),
            add_special_tokens=config.get("add_special_tokens", True),
        )

    def train_on_corpus(
        self,
        texts: list[str],
        min_frequency: int = 2,
        max_vocab_size: int | None = None,
    ) -> int:
        """Expand vocabulary based on corpus.

        Adds frequently occurring unknown tokens to vocabulary.

        Args:
            texts: Assembly code samples.
            min_frequency: Minimum occurrences to add token.
            max_vocab_size: Maximum vocabulary size.

        Returns:
            Number of tokens added.
        """
        # Reset unknown tracking
        self._unknown_tokens = {}

        # Tokenize all texts to collect unknowns
        for text in texts:
            self.encode(text, add_special_tokens=False)

        # Filter by frequency
        candidates = [
            token for token, count in self._unknown_tokens.items()
            if count >= min_frequency
        ]

        # Sort by frequency
        candidates.sort(key=lambda t: -self._unknown_tokens[t])

        # Limit if max_vocab_size specified
        if max_vocab_size:
            available = max_vocab_size - self.vocab_size
            candidates = candidates[:available]

        return self.add_tokens(candidates)


def create_tokenizer(
    split_addresses: bool = False,
    max_length: int = 512,
) -> ASMTokenizer:
    """Create a new ASM tokenizer with default settings.

    Args:
        split_addresses: Split addresses into prefix + digits.
        max_length: Maximum sequence length.

    Returns:
        Configured tokenizer.
    """
    return ASMTokenizer(
        split_addresses=split_addresses,
        max_length=max_length,
    )
