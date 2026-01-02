"""Tokenizer CLI commands: create, train, analyze, info."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tokenizer_create_command(args: argparse.Namespace) -> int:
    """Create a new ASM tokenizer."""
    from ..tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer(
        split_addresses=args.split_addresses,
        max_length=args.max_length,
    )

    output = Path(args.output)
    tokenizer.save(output)
    print(f"Created tokenizer with {len(tokenizer)} tokens")
    print(f"Saved to {output}")
    return 0


def tokenizer_train_command(args: argparse.Namespace) -> int:
    """Train tokenizer on corpus to expand vocabulary."""
    from ..tokenizer import ASMTokenizer

    # Load existing tokenizer or create new
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
        print(f"Loaded tokenizer with {len(tokenizer)} tokens")
    else:
        tokenizer = ASMTokenizer()
        print(f"Created new tokenizer with {len(tokenizer)} base tokens")

    # Load training texts
    texts = []
    input_path = Path(args.input)
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSONL
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    # Support various field names
                    for field in ["output", "text", "code", "asm"]:
                        if field in data:
                            texts.append(data[field])
                            break
                    continue
                except json.JSONDecodeError:
                    pass
            texts.append(line)

    print(f"Training on {len(texts)} samples...")
    added = tokenizer.train_on_corpus(
        texts,
        min_frequency=args.min_frequency,
        max_vocab_size=args.max_vocab_size,
    )

    print(f"Added {added} tokens. New vocab size: {len(tokenizer)}")

    # Save
    output = Path(args.output)
    tokenizer.save(output)
    print(f"Saved to {output}")

    # Show unknown tokens if requested
    if args.show_unknowns:
        unknowns = tokenizer.get_unknown_tokens()
        print(f"\nTop unknown tokens (not added):")
        for token, count in list(unknowns.items())[:20]:
            print(f"  {token}: {count}")

    return 0


def tokenizer_analyze_command(args: argparse.Namespace) -> int:
    """Analyze text with tokenizer."""
    from ..tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer.load(args.tokenizer)

    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        print("Error: --text or --file required")
        return 1

    # Tokenize
    tokens = tokenizer.tokenize(text)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]

    # Count unknowns
    unk_id = tokenizer.unk_token_id
    unk_count = sum(1 for i in input_ids if i == unk_id)
    unk_ratio = unk_count / len(input_ids) if input_ids else 0

    print(f"Tokens: {len(tokens)}")
    print(f"Unknown: {unk_count} ({100*unk_ratio:.1f}%)")
    print(f"\nTokens: {tokens[:50]}{'...' if len(tokens) > 50 else ''}")

    if args.verbose:
        print(f"\nIDs: {input_ids[:50]}{'...' if len(input_ids) > 50 else ''}")

        # Decode back
        decoded = tokenizer.decode(input_ids)
        print(f"\nDecoded: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")

    return 0


def tokenizer_info_command(args: argparse.Namespace) -> int:
    """Show tokenizer info."""
    from ..tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer.load(args.tokenizer)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Max length: {tokenizer.max_length}")
    print(f"Split addresses: {tokenizer.split_addresses}")
    print(f"\nSpecial tokens:")
    for name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {name}: {token} (id={token_id})")

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register tokenizer command parsers."""
    tok_parser = subparsers.add_parser(
        "tokenizer", help="ASM tokenizer utilities for 65816 assembly."
    )
    tok_sub = tok_parser.add_subparsers(dest="tokenizer_command")

    # tokenizer create
    tok_create = tok_sub.add_parser("create", help="Create a new ASM tokenizer.")
    tok_create.add_argument(
        "--output", required=True, help="Output directory for tokenizer."
    )
    tok_create.add_argument(
        "--split-addresses",
        action="store_true",
        help="Split addresses into prefix and value tokens.",
    )
    tok_create.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048).",
    )
    tok_create.set_defaults(func=tokenizer_create_command)

    # tokenizer train
    tok_train = tok_sub.add_parser(
        "train", help="Train tokenizer on corpus to expand vocabulary."
    )
    tok_train.add_argument(
        "--input", required=True, help="Input corpus (JSONL or plain text)."
    )
    tok_train.add_argument(
        "--output", required=True, help="Output directory for tokenizer."
    )
    tok_train.add_argument(
        "--tokenizer", help="Existing tokenizer to expand (creates new if not specified)."
    )
    tok_train.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency to add token (default: 2).",
    )
    tok_train.add_argument(
        "--max-vocab-size",
        type=int,
        default=32000,
        help="Maximum vocabulary size (default: 32000).",
    )
    tok_train.add_argument(
        "--show-unknowns",
        action="store_true",
        help="Show top unknown tokens not added.",
    )
    tok_train.set_defaults(func=tokenizer_train_command)

    # tokenizer analyze
    tok_analyze = tok_sub.add_parser("analyze", help="Analyze text with tokenizer.")
    tok_analyze.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer."
    )
    tok_analyze.add_argument("--text", help="Text to analyze.")
    tok_analyze.add_argument("--file", help="File to analyze.")
    tok_analyze.add_argument(
        "--verbose", action="store_true", help="Show detailed output."
    )
    tok_analyze.set_defaults(func=tokenizer_analyze_command)

    # tokenizer info
    tok_info = tok_sub.add_parser("info", help="Show tokenizer info.")
    tok_info.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer."
    )
    tok_info.set_defaults(func=tokenizer_info_command)
