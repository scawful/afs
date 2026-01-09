#!/usr/bin/env python3
"""
Convert tool usage examples to training format

Converts from custom JSON format to OpenAI function calling format
compatible with Qwen 2.5 Coder and other models.

Usage:
    python3 convert_to_training_format.py \
        --input ../examples/ \
        --output ../training_formatted/ \
        --format openai
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import random


def convert_to_openai_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert custom format to OpenAI function calling format

    Input format:
    {
      "instruction": "Read OAM table...",
      "tool_calls": [{
        "tool": "yaze_debugger.read_memory",
        "parameters": {"address": "0x0300", ...}
      }]
    }

    Output format:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Read OAM table..."},
        {"role": "assistant", "content": "", "tool_calls": [...]}
      ]
    }
    """

    # System prompt (can be customized per specialist model)
    system_prompt = (
        "You are an expert SNES ROM development assistant with access to MCP tools. "
        "You help developers debug, analyze, and modify SNES ROM files using "
        "yaze-debugger, mesen2 emulator, and z3ed-cli tools. "
        "Always use the appropriate MCP tool for each task."
    )

    # Convert tool calls to OpenAI format
    tool_calls = []
    for i, tc in enumerate(example.get('tool_calls', [])):
        tool_name = tc['tool']
        parameters = tc.get('parameters', {})

        # Convert parameters to JSON string
        arguments_json = json.dumps(parameters)

        tool_calls.append({
            "id": f"call_{i+1}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments_json
            }
        })

    # Build messages array
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": example['instruction']
        },
        {
            "role": "assistant",
            "content": "",  # Empty content when using tool calls
            "tool_calls": tool_calls
        }
    ]

    return {
        "messages": messages,
        "metadata": {
            "example_id": example.get('id', 'unknown'),
            "difficulty": example.get('difficulty', 'unknown'),
            "source": example.get('source', 'unknown')
        }
    }


def load_examples(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all example JSON files from directory tree"""
    examples = []

    # Search all subdirectories
    for json_file in input_dir.rglob('*.json'):
        # Skip summary files
        if 'summary' in json_file.name.lower():
            continue

        try:
            with open(json_file) as f:
                example = json.load(f)
                examples.append(example)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    return examples


def create_splits(examples: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create stratified train/val/test splits

    Stratifies by:
    - Tool type (maintain tool distribution)
    - Difficulty (maintain difficulty balance)
    """
    random.seed(seed)

    # Group examples by (first_tool, difficulty)
    groups = {}
    for example in examples:
        # Get metadata from the example
        metadata = example.get('metadata', {})

        # Try to get tool from messages if metadata not available
        messages = example.get('messages', [])
        first_tool = 'unknown'
        for msg in messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    first_tool = tool_calls[0].get('function', {}).get('name', 'unknown')
                    break

        difficulty = metadata.get('difficulty', 'unknown')
        key = (first_tool, difficulty)

        if key not in groups:
            groups[key] = []
        groups[key].append(example)

    # Split each group
    train_examples = []
    val_examples = []
    test_examples = []

    for key, group in groups.items():
        # Shuffle within group
        random.shuffle(group)

        # Calculate split points
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_examples.extend(group[:train_end])
        val_examples.extend(group[train_end:val_end])
        test_examples.extend(group[val_end:])

    # Final shuffle
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    return train_examples, val_examples, test_examples


def save_jsonl(examples: List[Dict], output_file: Path):
    """Save examples in JSONL format (one JSON object per line)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for example in examples:
            json_line = json.dumps(example)
            f.write(json_line + '\n')


def main():
    parser = argparse.ArgumentParser(description="Convert examples to training format")
    parser.add_argument('--input', type=str, required=True, help='Input directory with examples')
    parser.add_argument('--output', type=str, required=True, help='Output directory for formatted data')
    parser.add_argument('--format', type=str, default='openai', choices=['openai'], help='Output format')
    parser.add_argument('--split', action='store_true', help='Create train/val/test splits')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splits')

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    print(f"Loading examples from {input_dir}...")
    examples = load_examples(input_dir)
    print(f"Loaded {len(examples)} examples")

    print(f"\nConverting to {args.format} format...")
    converted = []
    for example in examples:
        if args.format == 'openai':
            converted.append(convert_to_openai_format(example))
        else:
            raise ValueError(f"Unknown format: {args.format}")

    print(f"Converted {len(converted)} examples")

    if args.split:
        print(f"\nCreating splits ({args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
        train, val, test = create_splits(
            converted,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        print(f"  Train: {len(train)} examples")
        print(f"  Val:   {len(val)} examples")
        print(f"  Test:  {len(test)} examples")

        # Save splits
        save_jsonl(train, output_dir / 'train.jsonl')
        save_jsonl(val, output_dir / 'val.jsonl')
        save_jsonl(test, output_dir / 'test.jsonl')

        print(f"\n✅ Saved splits to {output_dir}/")

        # Save split statistics
        stats = {
            "total_examples": len(converted),
            "train_examples": len(train),
            "val_examples": len(val),
            "test_examples": len(test),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "random_seed": args.seed,
            "format": args.format
        }

        with open(output_dir / 'split_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    else:
        # Save all examples to single file
        save_jsonl(converted, output_dir / 'all_examples.jsonl')
        print(f"\n✅ Saved {len(converted)} examples to {output_dir}/all_examples.jsonl")


if __name__ == '__main__':
    main()
