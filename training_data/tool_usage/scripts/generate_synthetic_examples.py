#!/usr/bin/env python3
"""
Generate Synthetic Tool Usage Examples

Creates comprehensive training examples covering all MCP tool operations
for yaze-debugger, mesen2, and z3ed-cli.

Usage:
    python3 generate_synthetic_examples.py \
        --output ../tool_usage/examples/synthetic/ \
        --schema ../tool_usage/schemas/mcp_tools_schema.json \
        --count 200
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
from dataclasses import dataclass, asdict
import random


@dataclass
class ToolCall:
    """Represents a single MCP tool call"""
    tool: str
    parameters: Dict[str, Any]
    rationale: str
    expected_output: str


@dataclass
class TrainingExample:
    """Single training example for tool usage"""
    id: str
    source: str  # "synthetic"
    context: Dict[str, Any]
    instruction: str
    tool_calls: List[ToolCall]
    success_criteria: str
    difficulty: str


class SyntheticExampleGenerator:
    """Generates synthetic training examples for tool usage"""

    def __init__(self, schema_path: Path):
        """Load tool schema"""
        with open(schema_path) as f:
            self.schema = json.load(f)

        # Common SNES/Zelda addresses for realistic examples
        self.common_addresses = {
            "oam_table": "0x0300",
            "oam_end": "0x033F",
            "link_state": "0x7E0000",
            "sprite_data": "0x7E0E00",
            "ppu_control": "0x2100",
            "dma_control": "0x4300",
            "rom_header": "0x00FFC0",
            "code_start": "0x008000",
            "graphics_bank": "0x1C8000",
            "dungeon_data": "0x0F8000"
        }

        # Common sprites for examples
        self.sprites = ["Octorok", "Link", "Moldorm", "Armos", "Keese", "Stalfos"]

        # Common graphics sheets
        self.graphics_sheets = list(range(0, 50))

        # Common ROM operations
        self.rom_operations = [
            "Read sprite graphics data",
            "Read dungeon room configuration",
            "Read overworld map data",
            "Read palette data",
            "Read text dialogue",
            "Apply code patch",
            "Fix sprite priority",
            "Modify level tileset"
        ]

    def generate_yaze_read_memory_examples(self, count: int) -> List[TrainingExample]:
        """Generate yaze_debugger.read_memory examples"""
        examples = []

        templates = [
            {
                "instruction": "Read OAM table to inspect sprite properties",
                "address": self.common_addresses["oam_table"],
                "length": 64,
                "context": "Debugging sprite rendering issues",
                "success": "Successfully read OAM property bytes",
                "difficulty": "simple"
            },
            {
                "instruction": "Read ROM header to verify version",
                "address": self.common_addresses["rom_header"],
                "length": 64,
                "context": "ROM identification and validation",
                "success": "ROM header data retrieved",
                "difficulty": "simple"
            },
            {
                "instruction": "Read graphics bank data for sprite analysis",
                "address": self.common_addresses["graphics_bank"],
                "length": 256,
                "context": "Analyzing sprite graphics structure",
                "success": "Graphics data extracted for analysis",
                "difficulty": "medium"
            },
            {
                "instruction": f"Read {random.choice(self.sprites)} sprite definition from ROM",
                "address": "0x008000",
                "length": 32,
                "context": "Sprite behavior analysis",
                "success": "Sprite data structure retrieved",
                "difficulty": "simple"
            }
        ]

        for i in range(min(count, len(templates))):
            template = templates[i % len(templates)]

            example = TrainingExample(
                id=f"synthetic_yaze_read_{i+1:03d}",
                source="synthetic",
                context={
                    "scenario": template["context"],
                    "rom_path": "~/roms/zelda3.sfc",
                    "tool_category": "rom_analysis"
                },
                instruction=template["instruction"],
                tool_calls=[
                    ToolCall(
                        tool="yaze_debugger.read_memory",
                        parameters={
                            "address": template["address"],
                            "length": template["length"],
                            "format": "hex"
                        },
                        rationale=f"Read ROM data to {template['instruction'].lower()}",
                        expected_output="Hex bytes showing ROM content at specified address"
                    )
                ],
                success_criteria=template["success"],
                difficulty=template["difficulty"]
            )
            examples.append(example)

        return examples

    def generate_yaze_write_memory_examples(self, count: int) -> List[TrainingExample]:
        """Generate yaze_debugger.write_memory examples"""
        examples = []

        patches = [
            ("EA EA EA", "NOP out problematic code", "simple"),
            ("A9 00 8D 00 03", "Clear OAM entry", "medium"),
            ("20 80 80", "JSR to subroutine", "medium"),
            ("80", "Set priority bit", "simple"),
            ("4C 00 80", "JMP to code location", "medium")
        ]

        for i in range(count):
            data, purpose, difficulty = patches[i % len(patches)]
            sprite = random.choice(self.sprites)

            example = TrainingExample(
                id=f"synthetic_yaze_write_{i+1:03d}",
                source="synthetic",
                context={
                    "scenario": f"Fix {sprite} behavior bug",
                    "rom_path": "~/roms/zelda3.sfc",
                    "tool_category": "rom_patching"
                },
                instruction=f"Apply patch to {purpose.lower()}",
                tool_calls=[
                    ToolCall(
                        tool="yaze_debugger.write_memory",
                        parameters={
                            "address": "0x008000",
                            "data": data
                        },
                        rationale=purpose,
                        expected_output="True if write successful"
                    )
                ],
                success_criteria=f"Patch applied successfully: {data}",
                difficulty=difficulty
            )
            examples.append(example)

        return examples

    def generate_mesen2_examples(self, count: int) -> List[TrainingExample]:
        """Generate mesen2 tool examples"""
        examples = []

        scenarios = [
            {
                "tool": "mesen2.load_rom",
                "instruction": "Load ROM into emulator for testing",
                "params": {"path": "~/roms/zelda3.sfc", "auto_save_state": True},
                "success": "ROM loaded successfully",
                "difficulty": "simple"
            },
            {
                "tool": "mesen2.read_memory",
                "instruction": "Read OAM table during gameplay to debug sprite rendering",
                "params": {"address": "0x0300", "length": 64, "memory_type": "work_ram"},
                "success": "Runtime OAM data retrieved",
                "difficulty": "medium"
            },
            {
                "tool": "mesen2.screenshot",
                "instruction": "Capture screenshot for visual regression testing",
                "params": {"format": "png"},
                "success": "Screenshot saved",
                "difficulty": "simple"
            },
            {
                "tool": "mesen2.run",
                "instruction": "Run emulation for 5 seconds to test changes",
                "params": {"speed": 1.0, "frames": 300},
                "success": "Emulation completed without errors",
                "difficulty": "simple"
            },
            {
                "tool": "mesen2.write_memory",
                "instruction": "Set debug flag in memory for testing",
                "params": {"address": "0x7E0200", "data": "FF", "memory_type": "work_ram"},
                "success": "Debug flag set successfully",
                "difficulty": "medium"
            }
        ]

        for i in range(count):
            scenario = scenarios[i % len(scenarios)]

            example = TrainingExample(
                id=f"synthetic_mesen2_{i+1:03d}",
                source="synthetic",
                context={
                    "scenario": "Emulator testing and debugging",
                    "rom_loaded": True if scenario["tool"] != "mesen2.load_rom" else False,
                    "tool_category": "emulation"
                },
                instruction=scenario["instruction"],
                tool_calls=[
                    ToolCall(
                        tool=scenario["tool"],
                        parameters=scenario["params"],
                        rationale=scenario["instruction"],
                        expected_output=scenario["success"]
                    )
                ],
                success_criteria=scenario["success"],
                difficulty=scenario["difficulty"]
            )
            examples.append(example)

        return examples

    def generate_z3ed_examples(self, count: int) -> List[TrainingExample]:
        """Generate z3ed CLI examples"""
        examples = []

        scenarios = [
            {
                "tool": "z3ed_cli.inspect",
                "instruction": "Inspect ROM header information",
                "params": {"rom_path": "~/roms/zelda3.sfc", "what": "header"},
                "success": "ROM metadata retrieved",
                "difficulty": "simple"
            },
            {
                "tool": "z3ed_cli.extract",
                "instruction": "Extract all graphics sheets for external editing",
                "params": {"rom_path": "~/roms/zelda3.sfc", "what": "graphics", "output_dir": "~/extracted/", "format": "png"},
                "success": "Graphics extracted to PNG files",
                "difficulty": "medium"
            },
            {
                "tool": "z3ed_cli.validate",
                "instruction": "Validate ROM integrity after patching",
                "params": {"rom_path": "~/roms/zelda3.sfc", "checks": ["all"]},
                "success": "ROM validation passed",
                "difficulty": "simple"
            },
            {
                "tool": "z3ed_cli.import",
                "instruction": "Import modified graphics back into ROM",
                "params": {"rom_path": "~/roms/zelda3.sfc", "data_type": "graphics", "input_path": "~/edited/sprite.png", "target_id": 10},
                "success": "Graphics imported successfully",
                "difficulty": "medium"
            }
        ]

        for i in range(count):
            scenario = scenarios[i % len(scenarios)]

            example = TrainingExample(
                id=f"synthetic_z3ed_{i+1:03d}",
                source="synthetic",
                context={
                    "scenario": "ROM editing and validation",
                    "rom_path": "~/roms/zelda3.sfc",
                    "tool_category": "rom_editing"
                },
                instruction=scenario["instruction"],
                tool_calls=[
                    ToolCall(
                        tool=scenario["tool"],
                        parameters=scenario["params"],
                        rationale=scenario["instruction"],
                        expected_output=scenario["success"]
                    )
                ],
                success_criteria=scenario["success"],
                difficulty=scenario["difficulty"]
            )
            examples.append(example)

        return examples

    def generate_complex_workflows(self, count: int) -> List[TrainingExample]:
        """Generate complex multi-tool workflows"""
        examples = []

        workflows = [
            {
                "instruction": "Debug sprite priority issue using yaze + mesen2",
                "tools": [
                    ("yaze_debugger.read_memory", {"address": "0x0300", "length": 64, "format": "hex"}),
                    ("mesen2.load_rom", {"path": "~/roms/zelda3.sfc", "auto_save_state": True}),
                    ("mesen2.read_memory", {"address": "0x0300", "length": 64, "memory_type": "work_ram"}),
                    ("mesen2.screenshot", {"format": "png"})
                ],
                "success": "Priority issue identified and documented",
                "difficulty": "complex"
            },
            {
                "instruction": "Extract, edit, and re-import graphics",
                "tools": [
                    ("z3ed_cli.extract", {"rom_path": "~/roms/zelda3.sfc", "what": "graphics", "output_dir": "~/extracted/", "format": "png"}),
                    ("z3ed_cli.import", {"rom_path": "~/roms/zelda3.sfc", "data_type": "graphics", "input_path": "~/edited/sprite.png", "target_id": 10}),
                    ("z3ed_cli.validate", {"rom_path": "~/roms/zelda3.sfc", "checks": ["all"]}),
                    ("mesen2.load_rom", {"path": "~/roms/zelda3.sfc", "auto_save_state": True}),
                    ("mesen2.screenshot", {"format": "png"})
                ],
                "success": "Graphics updated and verified visually",
                "difficulty": "complex"
            }
        ]

        for i in range(count):
            workflow = workflows[i % len(workflows)]

            tool_calls = []
            for tool_name, params in workflow["tools"]:
                tool_calls.append(
                    ToolCall(
                        tool=tool_name,
                        parameters=params,
                        rationale=f"Step in workflow: {workflow['instruction']}",
                        expected_output="Step completed"
                    )
                )

            example = TrainingExample(
                id=f"synthetic_complex_{i+1:03d}",
                source="synthetic",
                context={
                    "scenario": "Complex multi-tool workflow",
                    "requires_multiple_tools": True,
                    "tool_category": "workflow"
                },
                instruction=workflow["instruction"],
                tool_calls=tool_calls,
                success_criteria=workflow["success"],
                difficulty=workflow["difficulty"]
            )
            examples.append(example)

        return examples

    def generate_all(self, total_count: int, output_dir: Path):
        """Generate all synthetic examples"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Distribution of examples across tools
        yaze_read_count = int(total_count * 0.20)  # 20%
        yaze_write_count = int(total_count * 0.15)  # 15%
        mesen2_count = int(total_count * 0.40)  # 40%
        z3ed_count = int(total_count * 0.15)  # 15%
        complex_count = int(total_count * 0.10)  # 10%

        print(f"Generating {total_count} synthetic examples:")
        print(f"  yaze read_memory: {yaze_read_count}")
        print(f"  yaze write_memory: {yaze_write_count}")
        print(f"  mesen2: {mesen2_count}")
        print(f"  z3ed: {z3ed_count}")
        print(f"  complex workflows: {complex_count}")

        all_examples = []
        all_examples.extend(self.generate_yaze_read_memory_examples(yaze_read_count))
        all_examples.extend(self.generate_yaze_write_memory_examples(yaze_write_count))
        all_examples.extend(self.generate_mesen2_examples(mesen2_count))
        all_examples.extend(self.generate_z3ed_examples(z3ed_count))
        all_examples.extend(self.generate_complex_workflows(complex_count))

        # Save each example
        for example in all_examples:
            output_file = output_dir / f"{example.id}.json"
            with open(output_file, 'w') as f:
                json.dump(asdict(example), f, indent=2)

        # Generate summary
        by_tool = {}
        by_difficulty = {}
        for ex in all_examples:
            by_difficulty[ex.difficulty] = by_difficulty.get(ex.difficulty, 0) + 1
            for tc in ex.tool_calls:
                by_tool[tc.tool] = by_tool.get(tc.tool, 0) + 1

        summary = {
            "total_examples": len(all_examples),
            "by_tool": by_tool,
            "by_difficulty": by_difficulty,
            "source": "synthetic_generation",
            "generation_date": "2026-01-08"
        }

        summary_file = output_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Generation complete!")
        print(f"Generated {len(all_examples)} training examples")
        print(f"\nBy tool:")
        for tool, count in sorted(by_tool.items()):
            print(f"  {tool}: {count}")
        print(f"\nBy difficulty:")
        for difficulty, count in sorted(by_difficulty.items()):
            print(f"  {difficulty}: {count}")
        print(f"\nSummary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tool usage examples")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--schema", type=str, required=True, help="MCP tools schema JSON")
    parser.add_argument("--count", type=int, default=200, help="Total examples to generate")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    schema_path = Path(args.schema).expanduser()

    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return

    generator = SyntheticExampleGenerator(schema_path)
    generator.generate_all(args.count, output_dir)


if __name__ == "__main__":
    main()
