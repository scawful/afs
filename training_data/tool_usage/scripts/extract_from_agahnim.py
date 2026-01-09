#!/usr/bin/env python3
"""
Extract Tool Usage Examples from Agahnim Corpus

Parses 54 Agahnim workflow examples to identify implicit tool usage
and generates training examples for yaze/z3ed/mesen2.

Usage:
    python3 extract_from_agahnim.py \
        --input ../agahnim/examples/ \
        --output ../tool_usage/examples/ \
        --schema ../tool_usage/schemas/mcp_tools_schema.json
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from dataclasses import dataclass, asdict


@dataclass
class ToolCall:
    """Represents a single MCP tool call"""
    tool: str  # e.g., "yaze_debugger.read_memory"
    parameters: Dict[str, Any]
    rationale: str
    expected_output: str


@dataclass
class TrainingExample:
    """Single training example for tool usage"""
    id: str
    source: str  # "agahnim_example_017"
    context: Dict[str, Any]
    instruction: str
    tool_calls: List[ToolCall]
    success_criteria: str
    difficulty: str  # simple, medium, complex


class ToolUsageExtractor:
    """Extracts tool usage patterns from Agahnim examples"""

    def __init__(self, schema_path: Path):
        """Load tool schema for validation"""
        with open(schema_path) as f:
            self.schema = json.load(f)

        # Patterns that indicate tool usage in phase descriptions
        self.patterns = {
            "read_memory": [
                r"read.*(?:OAM|WRAM|memory|RAM|register|data)",
                r"inspect.*(?:sprite|property|bytes|value)",
                r"check.*(?:memory|value|state|flag)",
            ],
            "write_memory": [
                r"write.*(?:byte|data|value|patch)",
                r"apply.*patch",
                r"modify.*(?:ROM|data|bytes)",
                r"change.*(?:value|byte)",
            ],
            "set_breakpoint": [
                r"set.*breakpoint",
                r"break.*(?:at|on)",
                r"stop.*(?:at|when)",
            ],
            "read_memory_runtime": [
                r"read.*(?:during|runtime|gameplay)",
                r"inspect.*(?:OAM table|sprite state|PPU)",
                r"check.*(?:while running|in emulator)",
            ],
            "screenshot": [
                r"visual.*(?:verify|test|check)",
                r"take.*screenshot",
                r"capture.*frame",
                r"compare.*graphics",
            ],
            "assemble": [
                r"assemble.*(?:code|instruction|asm)",
                r"generate.*(?:patch|code)",
                r"calculate.*(?:instruction size|bytes)",
            ]
        }

    def extract_address_from_text(self, text: str) -> Optional[str]:
        """Extract hex address from text (e.g., $0300, 0x808000)"""
        # Match $HHHH or 0xHHHH formats
        match = re.search(r'(?:\$|0x)([0-9A-Fa-f]{4,6})', text)
        if match:
            hex_val = match.group(1)
            # Convert to 0xHHHHHH format
            return f"0x{hex_val.upper()}"
        return None

    def identify_tool_from_phase(self, phase: Dict[str, Any]) -> Optional[str]:
        """Identify which tool would be used based on phase description"""
        description = str(phase.get("description", "")).lower()
        name = str(phase.get("name", "")).lower()
        text = description + " " + name

        # Check for emulator usage (mesen2)
        if any(word in text for word in ["emulator", "runtime", "gameplay", "test in game"]):
            if any(re.search(p, text, re.I) for p in self.patterns["screenshot"]):
                return "mesen2.screenshot"
            if any(re.search(p, text, re.I) for p in self.patterns["read_memory_runtime"]):
                return "mesen2.read_memory"
            if "run" in text or "test" in text:
                return "mesen2.run"

        # Check for ROM operations (yaze_debugger)
        if any(re.search(p, text, re.I) for p in self.patterns["read_memory"]):
            return "yaze_debugger.read_memory"
        if any(re.search(p, text, re.I) for p in self.patterns["write_memory"]):
            return "yaze_debugger.write_memory"
        if any(re.search(p, text, re.I) for p in self.patterns["set_breakpoint"]):
            return "yaze_debugger.set_breakpoint"
        if any(re.search(p, text, re.I) for p in self.patterns["assemble"]):
            return "yaze_debugger.assemble"

        # Check for z3ed CLI operations
        if any(word in text for word in ["extract", "export", "dump"]):
            return "z3ed_cli.extract"
        if any(word in text for word in ["import", "apply", "insert"]):
            return "z3ed_cli.import"
        if "inspect" in text and "ROM" in text:
            return "z3ed_cli.inspect"

        return None

    def generate_tool_call(self, tool_name: str, phase: Dict[str, Any], context: Dict[str, Any]) -> Optional[ToolCall]:
        """Generate a ToolCall from phase and context"""
        description = phase.get("description", "")

        # Extract address if present
        address = self.extract_address_from_text(description)

        # Generate tool call based on tool type
        if tool_name == "yaze_debugger.read_memory":
            if not address:
                # Default to common addresses based on context
                if "OAM" in description:
                    address = "0x0300"
                elif "sprite" in description.lower():
                    address = "0x008000"
                else:
                    address = "0x000000"  # Fallback

            return ToolCall(
                tool="yaze_debugger.read_memory",
                parameters={
                    "address": address,
                    "length": 64 if "OAM" in description else 16,
                    "format": "hex"
                },
                rationale=f"Read ROM data to {phase.get('name', 'analyze')}",
                expected_output="Hex bytes showing ROM content"
            )

        elif tool_name == "yaze_debugger.write_memory":
            return ToolCall(
                tool="yaze_debugger.write_memory",
                parameters={
                    "address": address or "0x008000",
                    "data": "EA EA EA"  # NOP example
                },
                rationale=f"Apply patch to {phase.get('name', 'fix issue')}",
                expected_output="True if write successful"
            )

        elif tool_name == "mesen2.read_memory":
            return ToolCall(
                tool="mesen2.read_memory",
                parameters={
                    "address": address or "0x0300",
                    "length": 64,
                    "memory_type": "work_ram"
                },
                rationale=f"Read runtime memory state during {phase.get('name', 'testing')}",
                expected_output="Runtime memory contents"
            )

        elif tool_name == "mesen2.screenshot":
            return ToolCall(
                tool="mesen2.screenshot",
                parameters={
                    "format": "png"
                },
                rationale=f"Capture visual state to verify {phase.get('name', 'changes')}",
                expected_output="Path to screenshot PNG"
            )

        elif tool_name == "mesen2.run":
            return ToolCall(
                tool="mesen2.run",
                parameters={
                    "speed": 1.0,
                    "frames": 300  # 5 seconds at 60fps
                },
                rationale=f"Test changes in emulator for {phase.get('name', 'verification')}",
                expected_output="Emulation running"
            )

        elif tool_name == "yaze_debugger.assemble":
            code = "LDA #$00\nSTA $0300"  # Example code
            return ToolCall(
                tool="yaze_debugger.assemble",
                parameters={
                    "code": code,
                    "origin": "0x008000"
                },
                rationale=f"Assemble code for {phase.get('name', 'patch')}",
                expected_output="Assembled bytes: A9 00 8D 00 03"
            )

        elif tool_name == "z3ed_cli.inspect":
            return ToolCall(
                tool="z3ed_cli.inspect",
                parameters={
                    "rom_path": context.get("rom_path", "~/roms/zelda3.sfc"),
                    "what": "header"
                },
                rationale=f"Inspect ROM structure for {phase.get('name', 'analysis')}",
                expected_output="ROM header data"
            )

        return None

    def extract_from_example(self, example: Dict[str, Any]) -> List[TrainingExample]:
        """Extract training examples from a single Agahnim example"""
        examples = []
        example_id = example.get("id", "unknown")

        # Context from task
        task = example.get("task", {})

        # Safely extract file list from workflow
        files = []
        for phase in example.get("workflow", []):
            if isinstance(phase, dict) and phase.get("file_modified"):
                files.append(phase.get("file_modified"))

        context = {
            "task_description": task.get("description", ""),
            "bug_symptoms": task.get("bug_symptoms", []),
            "category": example.get("category", ""),
            "difficulty": example.get("difficulty", "medium"),
            "files": files
        }

        # Process each workflow phase
        workflow = example.get("workflow", [])
        for i, phase in enumerate(workflow):
            # Skip if phase is not a dict (malformed data)
            if not isinstance(phase, dict):
                continue

            tool_name = self.identify_tool_from_phase(phase)
            if not tool_name:
                continue  # No tool usage in this phase

            tool_call = self.generate_tool_call(tool_name, phase, context)
            if not tool_call:
                continue

            # Create training example
            training_ex = TrainingExample(
                id=f"{example_id}_phase_{i+1}",
                source=example_id,
                context=context,
                instruction=phase.get("description", phase.get("name", "Unknown task")),
                tool_calls=[tool_call],  # Single tool call per example for now
                success_criteria=phase.get("outcome", "Tool call completes successfully"),
                difficulty="simple" if len([tool_call]) == 1 else "medium"
            )

            examples.append(training_ex)

        return examples

    def process_corpus(self, input_dir: Path, output_dir: Path):
        """Process entire Agahnim corpus"""
        output_dir.mkdir(parents=True, exist_ok=True)

        examples_generated = 0
        examples_by_tool = {}

        # Process each example file
        for json_file in sorted(input_dir.glob("example_*.json")):
            print(f"Processing {json_file.name}...")

            with open(json_file) as f:
                example = json.load(f)

            # Extract training examples
            training_examples = self.extract_from_example(example)

            # Save each training example
            for tex in training_examples:
                output_file = output_dir / f"{tex.id}.json"
                with open(output_file, 'w') as f:
                    json.dump(asdict(tex), f, indent=2)

                examples_generated += 1

                # Track by tool
                tool = tex.tool_calls[0].tool if tex.tool_calls else "unknown"
                examples_by_tool[tool] = examples_by_tool.get(tool, 0) + 1

        # Generate summary
        summary = {
            "total_examples": examples_generated,
            "by_tool": examples_by_tool,
            "source": "agahnim_corpus_extraction",
            "extraction_date": "2026-01-08"
        }

        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Extraction complete!")
        print(f"Generated {examples_generated} training examples")
        print(f"\nBreakdown by tool:")
        for tool, count in sorted(examples_by_tool.items()):
            print(f"  {tool}: {count}")
        print(f"\nSummary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract tool usage from Agahnim corpus")
    parser.add_argument("--input", type=str, required=True, help="Agahnim examples directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for training examples")
    parser.add_argument("--schema", type=str, required=True, help="Path to MCP tools schema JSON")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    schema_path = Path(args.schema).expanduser()

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return

    extractor = ToolUsageExtractor(schema_path)
    extractor.process_corpus(input_dir, output_dir)


if __name__ == "__main__":
    main()
