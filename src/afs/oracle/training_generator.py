"""Oracle of Secrets training data generator for Majora v1.

Extracts training samples from the Oracle codebase (SNES ROM hack):
- Documentation → Q&A pairs and explanations
- Assembly code → code understanding and patterns
- Memory maps → variable lookups and system state
- Quest flow → progression knowledge
- Architecture → system design patterns

Expected output: ~2,845 raw samples → ~2,156 after quality filtering (>0.6)

Usage:
    python3 -m afs.oracle.training_generator \
        --oracle-path ~/src/hobby/oracle-of-secrets \
        --output ~/.context/training/oracle/majora_v1_raw.jsonl \
        --include-unity-patterns
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..generators.base import BaseGenerator, TrainingSample


@dataclass
class MarkdownSection:
    """Parsed section from markdown documentation."""
    title: str
    content: str
    level: int
    file_path: Path


@dataclass
class AsmCodeBlock:
    """Parsed assembly code block."""
    label: str
    code: str
    comments: list[str]
    file_path: Path


class MarkdownDocParser:
    """Parse Oracle markdown documentation."""

    def __init__(self, doc_path: Path):
        self.doc_path = doc_path
        self.content = doc_path.read_text()

    def extract_sections(self) -> list[MarkdownSection]:
        """Extract hierarchical sections from markdown."""
        sections = []
        lines = self.content.split('\n')

        current_section = None
        current_content = []

        for line in lines:
            # Check for headers (# Header, ## Header, etc.)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = MarkdownSection(
                    title=title,
                    content="",
                    level=level,
                    file_path=self.doc_path
                )
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        return sections

    def extract_code_blocks(self) -> list[str]:
        """Extract code blocks from markdown (```...```)."""
        code_blocks = []
        in_block = False
        current_block = []

        for line in self.content.split('\n'):
            if line.strip().startswith('```'):
                if in_block:
                    # End of block
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_block = False
                else:
                    # Start of block
                    in_block = True
            elif in_block:
                current_block.append(line)

        return code_blocks

    def extract_tables(self) -> list[dict[str, list[str]]]:
        """Extract markdown tables."""
        tables = []
        lines = self.content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]
            # Check for table header (| Col1 | Col2 |)
            if '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
                # Parse header
                header = [col.strip() for col in line.split('|')[1:-1]]

                # Skip separator line
                i += 2

                # Parse rows
                rows = []
                while i < len(lines) and '|' in lines[i]:
                    row = [col.strip() for col in lines[i].split('|')[1:-1]]
                    if len(row) == len(header):
                        rows.append(dict(zip(header, row, strict=False)))
                    i += 1

                if rows:
                    tables.append({"header": header, "rows": rows})
            else:
                i += 1

        return tables


class AsmCodeParser:
    """Parse Oracle 65816 assembly code."""

    def __init__(self, asm_path: Path):
        self.asm_path = asm_path
        self.content = asm_path.read_text()

    def extract_subroutines(self) -> list[AsmCodeBlock]:
        """Extract labeled subroutines with their code and comments."""
        subroutines = []
        lines = self.content.split('\n')

        current_label = None
        current_code = []
        current_comments = []

        for line in lines:
            # Check for label (starts at column 0, ends with :)
            if line and not line[0].isspace() and ':' in line:
                # Save previous subroutine
                if current_label:
                    subroutines.append(AsmCodeBlock(
                        label=current_label,
                        code='\n'.join(current_code),
                        comments=current_comments,
                        file_path=self.asm_path
                    ))

                # Start new subroutine
                current_label = line.split(':')[0].strip()
                current_code = [line]
                current_comments = []
            elif current_label:
                current_code.append(line)

                # Extract comments
                if ';' in line:
                    comment = line.split(';', 1)[1].strip()
                    if comment:
                        current_comments.append(comment)

        # Save last subroutine
        if current_label:
            subroutines.append(AsmCodeBlock(
                label=current_label,
                code='\n'.join(current_code),
                comments=current_comments,
                file_path=self.asm_path
            ))

        return subroutines

    def extract_constants(self) -> dict[str, str]:
        """Extract !define constants and their values."""
        constants = {}

        for line in self.content.split('\n'):
            # Match !define NAME = VALUE or !define NAME VALUE
            match = re.match(r'^\s*!define\s+(\w+)\s*[=\s]\s*(.+)$', line)
            if match:
                name = match.group(1)
                value = match.group(2).strip()
                # Remove comments
                if ';' in value:
                    value = value.split(';')[0].strip()
                constants[name] = value

        return constants


class OracleTrainingGenerator(BaseGenerator):
    """Generate training data from Oracle of Secrets codebase.

    Produces samples for:
    - Variable lookups (WRAM/SRAM addresses)
    - Quest progression knowledge
    - Code explanations (assembly patterns)
    - Architecture understanding (system design)
    - Sprite creation and behavior
    """

    def __init__(self, oracle_path: Path):
        """Initialize generator with Oracle codebase path."""
        super().__init__(name="oracle", domain="oracle")
        self.oracle_path = Path(oracle_path).expanduser().resolve()

        # Validate paths
        if not self.oracle_path.exists():
            raise FileNotFoundError(f"Oracle path not found: {self.oracle_path}")

        self.docs_path = self.oracle_path / "Docs"
        self.core_path = self.oracle_path / "Core"

        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_path}")

    def generate(self) -> Iterator[TrainingSample]:
        """Generate all training samples."""
        yield from self.generate_doc_samples()
        yield from self.generate_asm_samples()
        yield from self.generate_memory_samples()
        yield from self.generate_quest_samples()
        yield from self.generate_architecture_samples()

    def generate_doc_samples(self) -> Iterator[TrainingSample]:
        """Generate samples from documentation files.

        Patterns:
        - Section explanations: "What is [topic]?" → section content
        - Table lookups: "What is the value of [row]?" → table cell
        - Code examples: "How do I [task]?" → code snippet with explanation
        """
        doc_files = list(self.docs_path.rglob("*.md"))

        for doc_file in doc_files:
            try:
                parser = MarkdownDocParser(doc_file)
                sections = parser.extract_sections()
                tables = parser.extract_tables()
                code_blocks = parser.extract_code_blocks()

                # Generate from sections
                for section in sections:
                    if len(section.content) < 50:
                        continue  # Skip trivial sections

                    # Q&A: "What is [section title]?"
                    instruction = f"What is {section.title} in the Oracle of Secrets codebase?"
                    output = section.content

                    # Extract thinking from subsections
                    thinking = f"This is documented in {doc_file.relative_to(self.oracle_path)}. "
                    thinking += f"The section covers {section.title.lower()} with detailed information."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_docs",
                        source=str(doc_file.relative_to(self.oracle_path)),
                        _metadata={
                            "doc_section": section.title,
                            "section_level": section.level
                        }
                    )

                # Generate from tables
                for table in tables:
                    for row in table["rows"]:
                        # Use first column as key, others as values
                        if not row:
                            continue

                        key_col = table["header"][0]
                        key_val = row.get(key_col, "")

                        for col in table["header"][1:]:
                            val = row.get(col, "")
                            if not val or val == "-":
                                continue

                            instruction = f"In the Oracle of Secrets codebase, what is the {col.lower()} for {key_val}?"
                            output = val
                            thinking = f"Looking up {key_val} in the {doc_file.stem} documentation table."

                            yield TrainingSample(
                                instruction=instruction,
                                output=output,
                                thinking=thinking,
                                domain="oracle_lookup",
                                source=str(doc_file.relative_to(self.oracle_path)),
                                _metadata={
                                    "table_key": key_val,
                                    "table_column": col
                                }
                            )

            except Exception as e:
                print(f"Warning: Failed to parse {doc_file}: {e}")
                continue

    def generate_asm_samples(self) -> Iterator[TrainingSample]:
        """Generate samples from assembly code.

        Patterns:
        - Code explanations: "What does [subroutine] do?" → explanation from comments
        - Pattern recognition: "How is [pattern] implemented?" → code example
        - Constant lookups: "What is the value of [constant]?" → !define value
        """
        asm_files = list(self.core_path.rglob("*.asm"))

        for asm_file in asm_files:
            try:
                parser = AsmCodeParser(asm_file)
                subroutines = parser.extract_subroutines()
                constants = parser.extract_constants()

                # Generate from subroutines
                for sub in subroutines:
                    if len(sub.comments) < 2:
                        continue  # Skip undocumented subroutines

                    instruction = f"What does the {sub.label} subroutine do in Oracle of Secrets?"

                    # Build output from comments and code structure
                    output = f"The {sub.label} subroutine "
                    output += ' '.join(sub.comments[:3])  # First few comments

                    thinking = f"This subroutine is defined in {asm_file.relative_to(self.oracle_path)}. "
                    thinking += f"It has {len(sub.code.split(chr(10)))} lines of code with {len(sub.comments)} inline comments."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_asm",
                        source=str(asm_file.relative_to(self.oracle_path)),
                        _metadata={
                            "subroutine": sub.label,
                            "lines_of_code": len(sub.code.split('\n'))
                        }
                    )

                # Generate from constants
                for const_name, const_value in constants.items():
                    instruction = f"What is the value of {const_name} in Oracle of Secrets?"
                    output = f"The constant {const_name} is defined as {const_value}."
                    thinking = f"This is a !define constant in {asm_file.relative_to(self.oracle_path)}."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_constants",
                        source=str(asm_file.relative_to(self.oracle_path)),
                        _metadata={
                            "constant_name": const_name,
                            "constant_value": const_value
                        }
                    )

            except Exception as e:
                print(f"Warning: Failed to parse {asm_file}: {e}")
                continue

    def generate_memory_samples(self) -> Iterator[TrainingSample]:
        """Generate samples from memory map documentation.

        Patterns:
        - Address lookups: "What address stores [variable]?" → $7EXXXX
        - Variable explanations: "What does address $7EXXXX store?" → variable explanation
        - Memory range queries: "What is stored in WRAM/SRAM?" → overview
        """
        memory_doc = self.docs_path / "Core" / "MemoryMap.md"

        if not memory_doc.exists():
            return

        try:
            parser = MarkdownDocParser(memory_doc)
            tables = parser.extract_tables()

            for table in tables:
                for row in table["rows"]:
                    address = row.get("Address", row.get("Offset", ""))
                    name = row.get("Name", row.get("Variable", ""))
                    description = row.get("Description", row.get("Purpose", ""))

                    if not address or not name:
                        continue

                    # Lookup by variable name
                    instruction = f"What address stores {name} in Oracle of Secrets?"
                    output = f"{name} is stored at address {address}. {description}"
                    thinking = f"Checking the memory map documentation for {name}."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_memory",
                        source="Docs/Core/MemoryMap.md",
                        _metadata={
                            "address": address,
                            "variable_name": name
                        }
                    )

                    # Reverse lookup by address
                    instruction = f"What does address {address} store in Oracle of Secrets?"
                    output = f"Address {address} stores {name}. {description}"
                    thinking = f"Looking up address {address} in the memory map."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_memory",
                        source="Docs/Core/MemoryMap.md",
                        _metadata={
                            "address": address,
                            "variable_name": name
                        }
                    )

        except Exception as e:
            print(f"Warning: Failed to parse memory map: {e}")

    def generate_quest_samples(self) -> Iterator[TrainingSample]:
        """Generate samples from quest flow documentation.

        Patterns:
        - Quest progression: "How do I progress [quest]?" → steps
        - Flag checks: "What flag unlocks [event]?" → flag name and value
        - Sequence queries: "What comes after [event]?" → next step
        """
        quest_doc = self.docs_path / "Guides" / "QuestFlow.md"

        if not quest_doc.exists():
            return

        try:
            parser = MarkdownDocParser(quest_doc)
            sections = parser.extract_sections()

            for section in sections:
                if section.level > 2:  # Focus on main quest sections
                    continue

                # Extract quest steps from numbered lists
                steps = []
                for line in section.content.split('\n'):
                    if re.match(r'^\s*\d+\.', line):
                        steps.append(line.strip())

                if len(steps) < 2:
                    continue

                # Generate progression query
                instruction = f"How do I progress through {section.title} in Oracle of Secrets?"
                output = f"To complete {section.title}:\n" + '\n'.join(steps)
                thinking = f"Consulting the quest flow documentation for {section.title}."

                yield TrainingSample(
                    instruction=instruction,
                    output=output,
                    thinking=thinking,
                    domain="oracle_quest",
                    source="Docs/Guides/QuestFlow.md",
                    _metadata={
                        "quest_name": section.title,
                        "num_steps": len(steps)
                    }
                )

                # Generate step sequence queries
                for i in range(len(steps) - 1):
                    instruction = f"In Oracle of Secrets {section.title}, what comes after '{steps[i][:50]}...'?"
                    output = steps[i + 1]
                    thinking = f"Following the quest progression for {section.title}."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_quest",
                        source="Docs/Guides/QuestFlow.md",
                        _metadata={
                            "quest_name": section.title,
                            "step_index": i
                        }
                    )

        except Exception as e:
            print(f"Warning: Failed to parse quest flow: {e}")

    def generate_architecture_samples(self) -> Iterator[TrainingSample]:
        """Generate samples from architecture documentation.

        Patterns:
        - System queries: "How does [system] work?" → architecture explanation
        - Component interaction: "How does [A] interact with [B]?" → interaction flow
        - Design patterns: "What pattern is used for [feature]?" → pattern explanation
        """
        arch_docs = [
            self.docs_path / "Core" / "SystemArchitecture.md",
            self.docs_path / "Core" / "SystemInteractions.md",
        ]

        for arch_doc in arch_docs:
            if not arch_doc.exists():
                continue

            try:
                parser = MarkdownDocParser(arch_doc)
                sections = parser.extract_sections()

                for section in sections:
                    if len(section.content) < 100:
                        continue

                    instruction = f"How does {section.title} work in Oracle of Secrets?"
                    output = section.content
                    thinking = f"Referencing the system architecture documentation for {section.title}."

                    yield TrainingSample(
                        instruction=instruction,
                        output=output,
                        thinking=thinking,
                        domain="oracle_architecture",
                        source=str(arch_doc.relative_to(self.oracle_path)),
                        _metadata={
                            "architecture_topic": section.title,
                            "doc_type": arch_doc.stem
                        }
                    )

            except Exception as e:
                print(f"Warning: Failed to parse {arch_doc}: {e}")
                continue


def main():
    """CLI for generating Oracle training data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training data from Oracle of Secrets codebase"
    )
    parser.add_argument(
        "--oracle-path",
        type=Path,
        default=Path("~/src/hobby/oracle-of-secrets"),
        help="Path to Oracle of Secrets repository"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples (for testing)"
    )

    args = parser.parse_args()

    # Expand paths
    oracle_path = args.oracle_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Oracle Training Data Generator")
    print("=" * 60)
    print(f"\nOracle path: {oracle_path}")
    print(f"Output: {output_path}")

    # Generate samples
    generator = OracleTrainingGenerator(oracle_path)

    count = 0
    with open(output_path, 'w') as f:
        for sample in generator.generate():
            f.write(json.dumps(sample.to_dict()) + '\n')
            count += 1

            if count % 100 == 0:
                print(f"  Generated {count} samples...")

            if args.limit and count >= args.limit:
                break

    print(f"\n✓ Generated {count} training samples")
    print(f"  Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
