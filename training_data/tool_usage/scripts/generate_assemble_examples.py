#!/usr/bin/env python3
"""
Generate focused examples for yaze_debugger.assemble

Creates 20 high-quality examples covering different assembly scenarios
for SNES 65816 programming.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class ToolCall:
    tool: str
    parameters: Dict[str, Any]
    rationale: str
    expected_output: str


@dataclass
class TrainingExample:
    id: str
    source: str
    context: Dict[str, Any]
    instruction: str
    tool_calls: List[ToolCall]
    success_criteria: str
    difficulty: str


# 65816 assembly scenarios with realistic code snippets
ASSEMBLY_SCENARIOS = [
    # Basic instructions
    {
        "instruction": "Assemble NOP instruction at bank 0x80",
        "code": "NOP",
        "origin": "0x808000",
        "context": "Testing code injection point",
        "success": "Single NOP byte generated (0xEA)",
        "difficulty": "simple"
    },
    {
        "instruction": "Assemble LDA immediate to load sprite ID",
        "code": "LDA #$05",
        "origin": "0x808000",
        "context": "Loading sprite ID for new enemy",
        "success": "LDA immediate assembled (A9 05)",
        "difficulty": "simple"
    },
    {
        "instruction": "Assemble STA absolute to write to RAM",
        "code": "STA $0300",
        "origin": "0x808000",
        "context": "Writing to OAM table",
        "success": "STA absolute assembled (8D 00 03)",
        "difficulty": "simple"
    },
    
    # Branching
    {
        "instruction": "Assemble BNE branch for loop control",
        "code": "BNE $8080",
        "origin": "0x808000",
        "context": "Loop until counter reaches zero",
        "success": "Branch instruction generated",
        "difficulty": "medium"
    },
    {
        "instruction": "Assemble BEQ for conditional skip",
        "code": "BEQ $8090",
        "origin": "0x808000",
        "context": "Skip code if zero flag set",
        "success": "BEQ with relative offset",
        "difficulty": "medium"
    },
    
    # Subroutine calls
    {
        "instruction": "Assemble JSR to call subroutine",
        "code": "JSR $9000",
        "origin": "0x808000",
        "context": "Call sprite drawing routine",
        "success": "JSR absolute assembled (20 00 90)",
        "difficulty": "medium"
    },
    {
        "instruction": "Assemble RTS to return from subroutine",
        "code": "RTS",
        "origin": "0x808000",
        "context": "Return from custom routine",
        "success": "RTS byte generated (60)",
        "difficulty": "simple"
    },
    {
        "instruction": "Assemble JSL for long subroutine call",
        "code": "JSL $80F000",
        "origin": "0x808000",
        "context": "Call routine in different bank",
        "success": "JSL long assembled (22 00 F0 80)",
        "difficulty": "medium"
    },
    
    # Multi-line snippets
    {
        "instruction": "Assemble sprite initialization sequence",
        "code": "LDA #$10\\nSTA $0300\\nLDA #$20\\nSTA $0301",
        "origin": "0x808000",
        "context": "Initialize sprite X and Y positions",
        "success": "Multi-line assembly successful",
        "difficulty": "medium"
    },
    {
        "instruction": "Assemble loop counter decrement",
        "code": "LDA $7E0200\\nDEC A\\nSTA $7E0200\\nBNE $8000",
        "origin": "0x808000",
        "context": "Decrement RAM counter and loop",
        "success": "Loop code assembled",
        "difficulty": "complex"
    },
    
    # Stack operations
    {
        "instruction": "Assemble PHA to push accumulator",
        "code": "PHA",
        "origin": "0x808000",
        "context": "Save accumulator before subroutine",
        "success": "PHA byte generated (48)",
        "difficulty": "simple"
    },
    {
        "instruction": "Assemble PLA to pull accumulator",
        "code": "PLA",
        "origin": "0x808000",
        "context": "Restore accumulator after subroutine",
        "success": "PLA byte generated (68)",
        "difficulty": "simple"
    },
    {
        "instruction": "Assemble register preservation sequence",
        "code": "PHA\\nPHX\\nPHY",
        "origin": "0x808000",
        "context": "Save all registers before complex operation",
        "success": "Stack push sequence assembled",
        "difficulty": "medium"
    },
    
    # DMA setup
    {
        "instruction": "Assemble DMA register setup for VRAM transfer",
        "code": "LDA #$01\\nSTA $4300\\nLDA #$18\\nSTA $4301",
        "origin": "0x808000",
        "context": "Configure DMA for graphics upload",
        "success": "DMA setup code assembled",
        "difficulty": "complex"
    },
    
    # Index register operations
    {
        "instruction": "Assemble LDX immediate for loop counter",
        "code": "LDX #$08",
        "origin": "0x808000",
        "context": "Set X register as loop counter",
        "success": "LDX immediate assembled (A2 08)",
        "difficulty": "simple"
    },
    {
        "instruction": "Assemble DEX for loop decrement",
        "code": "DEX\\nBNE $8000",
        "origin": "0x808000",
        "context": "Decrement X and branch if not zero",
        "success": "DEX+BNE loop assembled",
        "difficulty": "medium"
    },
    
    # Addressing modes
    {
        "instruction": "Assemble LDA indirect indexed for sprite data",
        "code": "LDA ($10),Y",
        "origin": "0x808000",
        "context": "Read sprite data using pointer",
        "success": "Indirect indexed mode assembled",
        "difficulty": "complex"
    },
    {
        "instruction": "Assemble STA long absolute for different bank",
        "code": "STA $7E0200",
        "origin": "0x808000",
        "context": "Write to work RAM in bank 7E",
        "success": "Long absolute STA assembled",
        "difficulty": "medium"
    },
    
    # Bit manipulation
    {
        "instruction": "Assemble AND immediate to mask bits",
        "code": "AND #$0F",
        "origin": "0x808000",
        "context": "Mask lower nibble of value",
        "success": "AND immediate assembled (29 0F)",
        "difficulty": "medium"
    },
    {
        "instruction": "Assemble ORA immediate to set bits",
        "code": "ORA #$80",
        "origin": "0x808000",
        "context": "Set high bit for sprite priority",
        "success": "ORA immediate assembled (09 80)",
        "difficulty": "medium"
    }
]


def generate_assemble_examples(output_dir: Path) -> List[TrainingExample]:
    """Generate focused yaze_debugger.assemble examples"""
    examples = []
    
    for i, scenario in enumerate(ASSEMBLY_SCENARIOS, 1):
        example = TrainingExample(
            id=f"synthetic_assemble_{i:03d}",
            source="synthetic_focused",
            context={
                "scenario": scenario["context"],
                "rom_path": "~/roms/zelda3.sfc",
                "tool_category": "assembly_generation",
                "instruction_set": "65816"
            },
            instruction=scenario["instruction"],
            tool_calls=[
                ToolCall(
                    tool="yaze_debugger.assemble",
                    parameters={
                        "code": scenario["code"],
                        "origin": scenario["origin"]
                    },
                    rationale=f"Assemble 65816 code: {scenario['instruction']}",
                    expected_output="Assembled bytes in hexadecimal format"
                )
            ],
            success_criteria=scenario["success"],
            difficulty=scenario["difficulty"]
        )
        examples.append(example)
    
    return examples


def main():
    output_dir = Path("../examples/synthetic_assemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating 20 yaze_debugger.assemble examples...")
    examples = generate_assemble_examples(output_dir)
    
    # Save each example
    for example in examples:
        output_file = output_dir / f"{example.id}.json"
        with open(output_file, 'w') as f:
            # Convert dataclasses to dict
            example_dict = asdict(example)
            json.dump(example_dict, f, indent=2)
    
    # Generate summary
    summary = {
        "total_examples": len(examples),
        "tool": "yaze_debugger.assemble",
        "by_difficulty": {
            "simple": sum(1 for e in examples if e.difficulty == "simple"),
            "medium": sum(1 for e in examples if e.difficulty == "medium"),
            "complex": sum(1 for e in examples if e.difficulty == "complex")
        },
        "source": "synthetic_focused_generation",
        "generation_date": "2026-01-08",
        "instruction_types": [
            "Basic loads/stores",
            "Branching",
            "Subroutine calls",
            "Stack operations",
            "DMA setup",
            "Index operations",
            "Addressing modes",
            "Bit manipulation"
        ]
    }
    
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Generated {len(examples)} examples!")
    print(f"\nBy difficulty:")
    for diff, count in summary["by_difficulty"].items():
        print(f"  {diff}: {count}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
