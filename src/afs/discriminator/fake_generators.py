"""Fake assembly generators for ELECTRA training.

These generators introduce intentional errors into real 65816 assembly
to create "fake" examples for discriminator training.
"""

import random
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

# Valid 65816 opcodes
VALID_OPCODES = {
    # Load/Store
    "LDA", "LDX", "LDY", "STA", "STX", "STY", "STZ",
    # Transfer
    "TAX", "TAY", "TXA", "TYA", "TXS", "TSX", "TCD", "TDC", "TCS", "TSC", "XBA",
    # Stack
    "PHA", "PHX", "PHY", "PHP", "PHB", "PHD", "PHK",
    "PLA", "PLX", "PLY", "PLP", "PLB", "PLD",
    "PEA", "PEI", "PER",
    # Arithmetic
    "ADC", "SBC", "INC", "INX", "INY", "DEC", "DEX", "DEY",
    # Logical
    "AND", "ORA", "EOR", "BIT",
    # Shift/Rotate
    "ASL", "LSR", "ROL", "ROR",
    # Compare
    "CMP", "CPX", "CPY",
    # Branch
    "BRA", "BEQ", "BNE", "BCC", "BCS", "BMI", "BPL", "BVC", "BVS", "BRL",
    # Jump
    "JMP", "JML", "JSR", "JSL", "RTS", "RTL", "RTI",
    # Flag
    "SEC", "CLC", "SEI", "CLI", "SED", "CLD", "CLV",
    "SEP", "REP", "XCE",
    # Misc
    "NOP", "WDM", "STP", "WAI", "BRK", "COP",
    # Block Move
    "MVN", "MVP",
}

# Fake opcodes that look plausible but don't exist
FAKE_OPCODES = {
    "LDB", "STB", "LDZ", "TAB", "TBA", "PHZ", "PLZ",
    "ADX", "SBX", "INZ", "DEZ", "ANX", "ORX", "EOR",
    "CPA", "CPB", "JMS", "RET", "CAL", "PSH", "POP",
    "MOV", "ADD", "SUB", "MUL", "DIV", "NOT", "XOR",
}

# Common addressing mode patterns
ADDR_PATTERNS = {
    "immediate": r"#\$[0-9A-Fa-f]+",
    "direct": r"\$[0-9A-Fa-f]{2}(?![0-9A-Fa-f])",
    "absolute": r"\$[0-9A-Fa-f]{4}(?![0-9A-Fa-f])",
    "long": r"\$[0-9A-Fa-f]{6}",
    "indexed_x": r"\$[0-9A-Fa-f]+,\s*[Xx]",
    "indexed_y": r"\$[0-9A-Fa-f]+,\s*[Yy]",
    "indirect": r"\(\$[0-9A-Fa-f]+\)",
    "indirect_x": r"\(\$[0-9A-Fa-f]+,\s*[Xx]\)",
    "indirect_y": r"\(\$[0-9A-Fa-f]+\),\s*[Yy]",
}


@dataclass
class FakeResult:
    """Result of fake generation."""
    original: str
    modified: str
    error_type: str
    error_positions: list[int]  # Token indices that were modified


class FakeGenerator(ABC):
    """Base class for fake assembly generators."""

    @abstractmethod
    def generate(self, code: str) -> FakeResult | None:
        """Generate a fake version of the code.

        Returns None if no modification was possible.
        """
        pass

    def _tokenize_line(self, line: str) -> list[str]:
        """Split a line into tokens (label, opcode, operand, comment)."""
        # Remove leading/trailing whitespace
        line = line.strip()
        if not line:
            return []

        # Check for comment-only line
        if line.startswith(";"):
            return [line]

        tokens = []

        # Check for label (ends with :)
        if ":" in line:
            label_end = line.index(":")
            tokens.append(line[:label_end + 1])
            line = line[label_end + 1:].strip()

        if not line:
            return tokens

        # Split remaining into parts
        parts = line.split(None, 1)
        if parts:
            tokens.append(parts[0])  # opcode
            if len(parts) > 1:
                # Operand and possible comment
                rest = parts[1]
                if ";" in rest:
                    operand, comment = rest.split(";", 1)
                    tokens.append(operand.strip())
                    tokens.append(";" + comment)
                else:
                    tokens.append(rest.strip())

        return tokens


class SyntaxErrorGenerator(FakeGenerator):
    """Generate assembly with syntax errors."""

    def __init__(self, error_rate: float = 0.15):
        self.error_rate = error_rate

    def generate(self, code: str) -> FakeResult | None:
        lines = code.split("\n")
        modified_lines = []
        error_positions = []

        for i, line in enumerate(lines):
            if random.random() < self.error_rate:
                modified, error_type = self._introduce_syntax_error(line)
                if modified != line:
                    modified_lines.append(modified)
                    error_positions.append(i)
                    continue
            modified_lines.append(line)

        if not error_positions:
            return None

        return FakeResult(
            original=code,
            modified="\n".join(modified_lines),
            error_type="syntax",
            error_positions=error_positions,
        )

    def _introduce_syntax_error(self, line: str) -> tuple[str, str]:
        """Introduce a syntax error into a line."""
        errors = [
            self._missing_hash,
            self._wrong_brackets,
            self._missing_comma,
            self._double_comma,
            self._wrong_hex_prefix,
        ]

        error_func = random.choice(errors)
        return error_func(line)

    def _missing_hash(self, line: str) -> tuple[str, str]:
        """Remove # from immediate addressing."""
        if "#$" in line:
            return line.replace("#$", "$", 1), "missing_hash"
        return line, ""

    def _wrong_brackets(self, line: str) -> tuple[str, str]:
        """Use wrong bracket types."""
        if "(" in line and ")" in line:
            return line.replace("(", "[").replace(")", "]"), "wrong_brackets"
        return line, ""

    def _missing_comma(self, line: str) -> tuple[str, str]:
        """Remove comma from indexed addressing."""
        if ",X" in line.upper():
            return re.sub(r",\s*[Xx]", "X", line), "missing_comma"
        if ",Y" in line.upper():
            return re.sub(r",\s*[Yy]", "Y", line), "missing_comma"
        return line, ""

    def _double_comma(self, line: str) -> tuple[str, str]:
        """Add extra comma."""
        if "," in line:
            return line.replace(",", ",,", 1), "double_comma"
        return line, ""

    def _wrong_hex_prefix(self, line: str) -> tuple[str, str]:
        """Use wrong hex prefix (0x instead of $)."""
        if "$" in line:
            return re.sub(r"\$([0-9A-Fa-f]+)", r"0x\1", line), "wrong_hex_prefix"
        return line, ""


class OpcodeSwapGenerator(FakeGenerator):
    """Swap valid opcodes with fake/wrong ones."""

    def __init__(self, error_rate: float = 0.1):
        self.error_rate = error_rate

    def generate(self, code: str) -> FakeResult | None:
        lines = code.split("\n")
        modified_lines = []
        error_positions = []

        for i, line in enumerate(lines):
            tokens = self._tokenize_line(line)

            # Find opcode token
            for _j, token in enumerate(tokens):
                if token.upper() in VALID_OPCODES and random.random() < self.error_rate:
                    # Swap with fake opcode
                    fake = random.choice(list(FAKE_OPCODES))
                    modified_line = line.replace(token, fake, 1)
                    modified_lines.append(modified_line)
                    error_positions.append(i)
                    break
            else:
                modified_lines.append(line)

        if not error_positions:
            return None

        return FakeResult(
            original=code,
            modified="\n".join(modified_lines),
            error_type="opcode_swap",
            error_positions=error_positions,
        )


class AddressingErrorGenerator(FakeGenerator):
    """Generate assembly with wrong addressing modes."""

    def __init__(self, error_rate: float = 0.1):
        self.error_rate = error_rate

    def generate(self, code: str) -> FakeResult | None:
        lines = code.split("\n")
        modified_lines = []
        error_positions = []

        for i, line in enumerate(lines):
            if random.random() < self.error_rate:
                modified = self._modify_addressing(line)
                if modified != line:
                    modified_lines.append(modified)
                    error_positions.append(i)
                    continue
            modified_lines.append(line)

        if not error_positions:
            return None

        return FakeResult(
            original=code,
            modified="\n".join(modified_lines),
            error_type="addressing",
            error_positions=error_positions,
        )

    def _modify_addressing(self, line: str) -> str:
        """Modify addressing mode in a line."""
        # Try various addressing mode corruptions

        # Truncate long address to short
        match = re.search(r"\$([0-9A-Fa-f]{4,6})", line)
        if match and random.random() < 0.5:
            addr = match.group(1)
            if len(addr) >= 4:
                # Truncate to 2 digits (wrong for absolute addressing)
                return line.replace(f"${addr}", f"${addr[-2:]}")

        # Change direct page to absolute unnecessarily
        match = re.search(r"\$([0-9A-Fa-f]{2})(?![0-9A-Fa-f])", line)
        if match and random.random() < 0.5:
            addr = match.group(1)
            # Pad to 4 digits with wrong bank
            return line.replace(f"${addr}", f"$00{addr}")

        # Swap X and Y indexing
        if ",X" in line.upper():
            return re.sub(r",\s*[Xx]", ",Y", line)
        if ",Y" in line.upper():
            return re.sub(r",\s*[Yy]", ",X", line)

        return line


class CompositeGenerator(FakeGenerator):
    """Combine multiple generators for varied fake output."""

    def __init__(
        self,
        generators: list[FakeGenerator] | None = None,
        error_rate: float = 0.15,
    ):
        if generators is None:
            generators = [
                SyntaxErrorGenerator(error_rate),
                OpcodeSwapGenerator(error_rate),
                AddressingErrorGenerator(error_rate),
            ]
        self.generators = generators

    def generate(self, code: str) -> FakeResult | None:
        # Pick a random generator
        generator = random.choice(self.generators)
        return generator.generate(code)

    def generate_batch(
        self,
        codes: list[str],
        target_ratio: float = 0.5,
    ) -> Iterator[tuple[str, int]]:
        """Generate a batch with specified real/fake ratio.

        Yields (code, label) pairs where label is 0 for real, 1 for fake.
        """
        for code in codes:
            # Randomly decide if this should be real or fake
            if random.random() < target_ratio:
                # Generate fake
                result = self.generate(code)
                if result:
                    yield result.modified, 1
                else:
                    # Couldn't generate fake, use original
                    yield code, 0
            else:
                # Keep real
                yield code, 0
