"""ToolBench dataset converter.

Converts ToolBench (HuggingFace: tuandunghcmut/toolbench-v1) format to AFS TrainingSample.

ToolBench format:
- id: Task description
- conversations: Dict with 'from' and 'value' arrays showing multi-turn tool use
  - system: Sets up task and available tools
  - user: User query
  - assistant: Thought, Action, Action Input
  - function: Tool execution result
  - assistant: Final answer with Finish function

Output TrainingSample:
- instruction: User's task/query
- output: Final answer (from last assistant message)
- thinking: Chain of thought (all Thought/Action pairs)
- domain: "tool_use"
- source: "toolbench"
- metadata: Tools used, number of steps, task type
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ...generators.base import TrainingSample


def parse_toolbench_parquet(parquet_path: Path) -> list[TrainingSample]:
    """Parse ToolBench parquet file into TrainingSample objects.

    Args:
        parquet_path: Path to ToolBench parquet file

    Returns:
        List of TrainingSample objects
    """
    from ...generators.base import TrainingSample

    df = pd.read_parquet(parquet_path)
    samples = []

    for _, row in df.iterrows():
        task_id = row['id']
        conversations = row['conversations']

        # Extract from/value arrays
        from_array = conversations['from']
        value_array = conversations['value']

        # Build conversation turns
        turns = list(zip(from_array, value_array, strict=False))

        # Parse conversation
        user_query = None
        assistant_messages = []
        function_results = []
        system_prompt = None
        final_answer = None

        for role, content in turns:
            if role == 'system':
                system_prompt = content
            elif role == 'user':
                user_query = content
            elif role == 'assistant':
                assistant_messages.append(content)
                # Check if this is the final answer
                if 'Finish' in content and 'give_answer' in content:
                    final_answer = _extract_final_answer(content)
            elif role == 'function':
                function_results.append(content)

        # Skip if no user query or final answer
        if not user_query or not final_answer:
            continue

        # Build thinking from assistant messages (exclude final answer)
        thinking_parts = []
        for msg in assistant_messages[:-1]:  # Exclude last (final answer)
            thought = _extract_thought(msg)
            action = _extract_action(msg)
            if thought:
                thinking_parts.append(f"Thought: {thought}")
            if action:
                thinking_parts.append(f"Action: {action['name']}")
                if action['input']:
                    thinking_parts.append(f"Input: {json.dumps(action['input'], indent=2)}")

        thinking = "\n".join(thinking_parts) if thinking_parts else None

        # Extract metadata
        tools_used = _extract_tools_used(assistant_messages)
        num_steps = len([msg for msg in assistant_messages if 'Action:' in msg])

        # Determine task type from system prompt
        task_type = _determine_task_type(system_prompt or "")

        # Create TrainingSample
        sample = TrainingSample(
            instruction=user_query.strip(),
            output=final_answer.strip(),
            thinking=thinking,
            domain="tool_use",
            source="toolbench",
            _metadata={
                "task_id": task_id[:100],  # Truncate if too long
                "tools_used": tools_used,
                "num_steps": num_steps,
                "task_type": task_type,
                "has_system_prompt": bool(system_prompt),
            }
        )

        samples.append(sample)

    return samples


def _extract_thought(assistant_message: str) -> str | None:
    """Extract Thought from assistant message."""
    match = re.search(r'Thought:\s*(.+?)(?:\nAction:|$)', assistant_message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_action(assistant_message: str) -> dict[str, Any] | None:
    """Extract Action and Action Input from assistant message."""
    action_match = re.search(r'Action:\s*(\w+)', assistant_message)
    if not action_match:
        return None

    action_name = action_match.group(1).strip()

    # Extract Action Input (JSON)
    input_match = re.search(r'Action Input:\s*({.+?})', assistant_message, re.DOTALL)
    action_input = {}
    if input_match:
        try:
            action_input = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            pass  # Keep empty dict

    return {
        "name": action_name,
        "input": action_input
    }


def _extract_final_answer(assistant_message: str) -> str:
    """Extract final answer from Finish function call."""
    # Look for final_answer in JSON
    match = re.search(r'"final_answer":\s*"(.+?)"(?=[,}])', assistant_message, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: extract everything after "final_answer"
    match = re.search(r'final_answer.*?:\s*"(.+)"', assistant_message, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return whole message
    return assistant_message


def _extract_tools_used(assistant_messages: list[str]) -> list[str]:
    """Extract unique list of tools used."""
    tools = []
    for msg in assistant_messages:
        action = _extract_action(msg)
        if action and action['name'] not in ['Finish', 'finish']:
            if action['name'] not in tools:
                tools.append(action['name'])
    return tools


def _determine_task_type(system_prompt: str) -> str:
    """Determine task type from system prompt."""
    system_lower = system_prompt.lower()

    if 'instagram' in system_lower or 'social media' in system_lower:
        return 'social_media'
    elif 'search' in system_lower or 'web' in system_lower:
        return 'web_search'
    elif 'api' in system_lower or 'data' in system_lower:
        return 'api_call'
    elif 'analyze' in system_lower or 'process' in system_lower:
        return 'analysis'
    else:
        return 'general'


def load_toolbench_dataset(
    dataset_dir: Path,
    split: str = "train",
    max_samples: int | None = None
) -> list[TrainingSample]:
    """Load ToolBench dataset from directory.

    Args:
        dataset_dir: Directory containing ToolBench parquet files
        split: Dataset split ('train' or 'validation')
        max_samples: Maximum samples to load (None = all)

    Returns:
        List of TrainingSample objects
    """
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find parquet files for split
    pattern = f"{split}-*.parquet" if split == "train" else f"{split}-*.parquet"
    parquet_files = sorted(data_dir.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(f"No {split} parquet files found in {data_dir}")

    all_samples = []
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}...")
        samples = parse_toolbench_parquet(parquet_file)
        all_samples.extend(samples)

        if max_samples and len(all_samples) >= max_samples:
            all_samples = all_samples[:max_samples]
            break

    print(f"Loaded {len(all_samples)} samples from ToolBench {split} split")
    return all_samples


def export_toolbench_to_jsonl(
    dataset_dir: Path,
    output_path: Path,
    split: str = "train",
    max_samples: int | None = None
) -> int:
    """Export ToolBench dataset to JSONL format.

    Args:
        dataset_dir: Directory containing ToolBench parquet files
        output_path: Output JSONL file path
        split: Dataset split ('train' or 'validation')
        max_samples: Maximum samples to export (None = all)

    Returns:
        Number of samples exported
    """
    samples = load_toolbench_dataset(dataset_dir, split, max_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + '\n')

    print(f"Exported {len(samples)} samples to {output_path}")
    return len(samples)
