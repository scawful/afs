"""Din (Optimization) benchmark runner.

Evaluates optimization quality:
- Cycle count reduction
- Code size reduction
- Correctness (semantic equivalence)
"""

from __future__ import annotations

import re
from typing import Any

from .base import BenchmarkConfig, BenchmarkItem, BenchmarkRunner


class DinBenchmark(BenchmarkRunner):
    """Benchmark runner for Din (optimization) expert."""

    domain = "din"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._compiler = None

    def _get_compiler(self):
        """Lazy-load ASAR compiler."""
        if self._compiler is None:
            from ..evaluation.semantic_eval import SemanticEvaluator
            evaluator = SemanticEvaluator()
            self._compiler = evaluator.compile_code
        return self._compiler

    def _format_instruction(self, item: BenchmarkItem) -> str:
        """Format optimization instruction."""
        return f"Optimize this 65816 assembly code for fewer cycles:\n```\n{item.code}\n```"

    def _extract_code(self, output: str) -> str:
        """Extract assembly code from model output."""
        # Try to find code blocks
        code_block_match = re.search(r"```(?:asm|assembly|65816)?\n(.*?)```", output, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find code after "Optimized:" or similar
        optimized_match = re.search(r"(?:Optimized|Result|Output):\s*\n(.*?)(?:\n\n|$)", output, re.DOTALL)
        if optimized_match:
            return optimized_match.group(1).strip()

        # Return first lines that look like assembly
        lines = []
        for line in output.split("\n"):
            line = line.strip()
            if line and not line.startswith(("*", "-", "1.", "2.", "#")):
                # Looks like assembly if it starts with label or opcode
                if re.match(r"^[A-Za-z_\.@]", line) or line.startswith(";"):
                    lines.append(line)
        return "\n".join(lines)

    def _count_cycles(self, code: str) -> int:
        """Estimate cycle count for 65816 code.

        This is a simplified estimation - real cycle counting requires
        considering addressing modes, page crossings, etc.
        """
        # Simplified cycle counts for common instructions
        CYCLE_COUNTS = {
            # Load/Store
            "LDA": 3, "LDX": 3, "LDY": 3,
            "STA": 4, "STX": 4, "STY": 4,
            "STZ": 4,
            # Arithmetic
            "ADC": 3, "SBC": 3,
            "INC": 5, "DEC": 5,
            "INX": 2, "INY": 2, "DEX": 2, "DEY": 2,
            # Logic
            "AND": 3, "ORA": 3, "EOR": 3,
            "ASL": 2, "LSR": 2, "ROL": 2, "ROR": 2,
            # Compare/Branch
            "CMP": 3, "CPX": 3, "CPY": 3,
            "BEQ": 2, "BNE": 2, "BCC": 2, "BCS": 2,
            "BPL": 2, "BMI": 2, "BRA": 3, "BRL": 4,
            # Transfer
            "TAX": 2, "TAY": 2, "TXA": 2, "TYA": 2,
            "TXS": 2, "TSX": 2,
            # Stack
            "PHA": 3, "PLA": 4, "PHX": 3, "PLX": 4,
            "PHY": 3, "PLY": 4, "PHP": 3, "PLP": 4,
            # Mode
            "REP": 3, "SEP": 3,
            # Subroutine
            "JSR": 6, "JSL": 8, "RTS": 6, "RTL": 6, "RTI": 7,
            # Jump
            "JMP": 3, "JML": 4,
            # Flags
            "SEC": 2, "CLC": 2, "SEI": 2, "CLI": 2,
            # NOP
            "NOP": 2,
        }

        total = 0
        for line in code.split("\n"):
            line = line.strip()
            if not line or line.startswith(";") or line.endswith(":"):
                continue

            # Extract opcode
            match = re.match(r"([A-Z]{3})", line.upper())
            if match:
                opcode = match.group(1)
                cycles = CYCLE_COUNTS.get(opcode, 4)  # Default 4 cycles

                # Adjust for addressing modes
                if "#$" in line:  # Immediate
                    pass
                elif "$7E" in line or "$7F" in line:  # Long address
                    cycles += 1
                elif ",X" in line or ",Y" in line:  # Indexed
                    cycles += 1

                total += cycles

        return total

    def _count_bytes(self, code: str) -> int:
        """Estimate byte count for 65816 code."""
        BYTE_COUNTS = {
            # Implied (1 byte)
            "INX": 1, "INY": 1, "DEX": 1, "DEY": 1,
            "TAX": 1, "TAY": 1, "TXA": 1, "TYA": 1,
            "TXS": 1, "TSX": 1,
            "PHA": 1, "PLA": 1, "PHX": 1, "PLX": 1,
            "PHY": 1, "PLY": 1, "PHP": 1, "PLP": 1,
            "SEC": 1, "CLC": 1, "SEI": 1, "CLI": 1,
            "NOP": 1, "RTS": 1, "RTL": 1, "RTI": 1,
            # Immediate/Zero Page (2 bytes)
            "LDA": 2, "LDX": 2, "LDY": 2,
            "REP": 2, "SEP": 2,
            "ADC": 2, "SBC": 2, "AND": 2, "ORA": 2, "EOR": 2,
            "CMP": 2, "CPX": 2, "CPY": 2,
            "ASL": 1, "LSR": 1, "ROL": 1, "ROR": 1,  # Implied mode
            # Store (2-4 bytes depending on address)
            "STA": 3, "STX": 3, "STY": 3, "STZ": 3,
            "INC": 3, "DEC": 3,
            # Branch (2 bytes)
            "BEQ": 2, "BNE": 2, "BCC": 2, "BCS": 2,
            "BPL": 2, "BMI": 2, "BRA": 2, "BRL": 3,
            # Subroutine (3 bytes)
            "JSR": 3, "JMP": 3,
            # Long (4 bytes)
            "JSL": 4, "JML": 4,
        }

        total = 0
        for line in code.split("\n"):
            line = line.strip()
            if not line or line.startswith(";") or line.endswith(":"):
                continue

            match = re.match(r"([A-Z]{3})", line.upper())
            if match:
                opcode = match.group(1)
                bytes_count = BYTE_COUNTS.get(opcode, 3)

                # Adjust for long addresses
                if "$7E" in line or "$7F" in line:
                    bytes_count += 1
                elif "#$" in line and len(re.findall(r"\$([0-9A-Fa-f]+)", line)[0] if re.findall(r"\$([0-9A-Fa-f]+)", line) else "") > 2:
                    bytes_count += 1  # 16-bit immediate

                total += bytes_count

        return total

    def evaluate_item(self, item: BenchmarkItem, output: str) -> dict[str, Any]:
        """Evaluate an optimization result."""
        # Extract optimized code
        optimized_code = self._extract_code(output)

        if not optimized_code:
            return {
                "passed": False,
                "score": 0.0,
                "metrics": {},
                "details": {"error": "No code extracted from output"},
            }

        # Calculate metrics
        baseline_cycles = self._count_cycles(item.code)
        optimized_cycles = self._count_cycles(optimized_code)
        baseline_bytes = self._count_bytes(item.code)
        optimized_bytes = self._count_bytes(optimized_code)

        # Calculate reductions
        cycle_reduction = 0.0
        if baseline_cycles > 0:
            cycle_reduction = (baseline_cycles - optimized_cycles) / baseline_cycles

        size_reduction = 0.0
        if baseline_bytes > 0:
            size_reduction = (baseline_bytes - optimized_bytes) / baseline_bytes

        # Try to compile both (if semantic eval enabled)
        baseline_compiles = True
        optimized_compiles = True

        if self.config.enable_semantic_eval:
            try:
                compile_fn = self._get_compiler()
                baseline_compiles, _, _, _ = compile_fn(f"org $008000\n{item.code}")
                optimized_compiles, _, _, _ = compile_fn(f"org $008000\n{optimized_code}")
            except Exception:
                pass

        # Calculate overall score
        # Positive cycle reduction is good, negative is bad
        # Bonus for size reduction, penalty for compilation failure
        score = 0.0
        if optimized_compiles:
            score += 0.3  # Base score for compiling
            score += max(0, cycle_reduction) * 0.4  # Up to 0.4 for cycle reduction
            score += max(0, size_reduction) * 0.3  # Up to 0.3 for size reduction
        else:
            score = 0.1 if cycle_reduction > 0 else 0.0

        # Passed if we have positive improvement and code compiles
        passed = optimized_compiles and (cycle_reduction > 0 or size_reduction > 0)

        return {
            "passed": passed,
            "score": min(1.0, score),
            "metrics": {
                "baseline_cycles": baseline_cycles,
                "optimized_cycles": optimized_cycles,
                "cycle_reduction": cycle_reduction,
                "baseline_bytes": baseline_bytes,
                "optimized_bytes": optimized_bytes,
                "size_reduction": size_reduction,
                "compiles": 1.0 if optimized_compiles else 0.0,
            },
            "details": {
                "baseline_code": item.code,
                "optimized_code": optimized_code,
            },
        }
