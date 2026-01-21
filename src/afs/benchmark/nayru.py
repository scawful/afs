"""Nayru (Generation) benchmark runner.

Evaluates code generation quality:
- ASAR compilation validity
- Entity coverage (correct ALTTP addresses)
- Code structure quality
- Semantic correctness
"""

from __future__ import annotations

import re
from typing import Any

from .base import BenchmarkConfig, BenchmarkItem, BenchmarkRunner


class NayruBenchmark(BenchmarkRunner):
    """Benchmark runner for Nayru (generation) expert."""

    domain = "nayru"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._evaluator = None

    def _get_evaluator(self):
        """Lazy-load semantic evaluator."""
        if self._evaluator is None:
            from ..evaluation.semantic_eval import SemanticEvaluator
            self._evaluator = SemanticEvaluator()
        return self._evaluator

    def _format_instruction(self, item: BenchmarkItem) -> str:
        """Format generation instruction."""
        task = item.metadata.get("task", item.code)
        context = item.metadata.get("context", "")

        if context:
            return f"Write a 65816 assembly routine to {task}\n\nContext:\n{context}"
        return f"Write a 65816 assembly routine to {task}"

    def _extract_code(self, output: str) -> str:
        """Extract assembly code from model output."""
        # Try to find code blocks
        code_block_match = re.search(r"```(?:asm|assembly|65816)?\n(.*?)```", output, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Return lines that look like assembly
        lines = []
        in_code = False
        for line in output.split("\n"):
            stripped = line.strip()

            # Start of code section
            if re.match(r"^[A-Za-z_\.@][A-Za-z0-9_]*:", stripped):
                in_code = True

            if in_code or re.match(r"^\s*[A-Z]{3}", stripped):
                in_code = True
                lines.append(line.rstrip())

            # End conditions
            if stripped.startswith("RTS") or stripped.startswith("RTL") or stripped.startswith("RTI"):
                lines.append(line.rstrip())
                break

        return "\n".join(lines)

    def _check_alttp_entities(self, code: str) -> dict[str, Any]:
        """Check for correct ALTTP entity usage."""
        # Common ALTTP WRAM addresses
        ALTTP_ENTITIES = {
            # Link state
            "$7E0020": "link_y_lo",
            "$7E0021": "link_y_hi",
            "$7E0022": "link_x_lo",
            "$7E0023": "link_x_hi",
            "$7E002F": "link_direction",
            "$7E0031": "link_speed",
            "$7E036C": "link_health",
            "$7E036D": "link_max_health",
            "$7E005D": "link_state",
            # Input
            "$7E00F0": "joypad_pressed",
            "$7E00F2": "joypad_held",
            "$4218": "joypad1_lo",
            "$4219": "joypad1_hi",
            # PPU
            "$2100": "inidisp",
            "$2115": "vmain",
            "$2116": "vmaddl",
            "$2117": "vmaddh",
            "$2118": "vmdatal",
            "$2119": "vmdatah",
            # DMA
            "$420B": "mdmaen",
            "$4300": "dmap0",
            "$4301": "bbad0",
            "$4302": "a1t0l",
            "$4303": "a1t0h",
            "$4304": "a1b0",
            "$4305": "das0l",
            "$4306": "das0h",
        }

        found_entities = []
        code_upper = code.upper()

        for addr, name in ALTTP_ENTITIES.items():
            if addr.upper() in code_upper:
                found_entities.append(name)

        # Check for common patterns
        has_proper_structure = bool(re.search(r"(RTS|RTL|RTI)\s*$", code, re.MULTILINE))
        has_labels = bool(re.search(r"^[A-Za-z_][A-Za-z0-9_]*:", code, re.MULTILINE))
        has_comments = ";" in code

        return {
            "entities_found": found_entities,
            "entity_count": len(found_entities),
            "has_structure": has_proper_structure,
            "has_labels": has_labels,
            "has_comments": has_comments,
        }

    def _count_instructions(self, code: str) -> int:
        """Count assembly instructions."""
        count = 0
        for line in code.split("\n"):
            line = line.strip()
            if not line or line.startswith(";") or line.endswith(":"):
                continue
            if re.match(r"^[A-Z]{3}", line.upper()):
                count += 1
        return count

    def evaluate_item(self, item: BenchmarkItem, output: str) -> dict[str, Any]:
        """Evaluate a generation result."""
        # Extract code
        generated_code = self._extract_code(output)

        if not generated_code:
            return {
                "passed": False,
                "score": 0.0,
                "metrics": {},
                "details": {"error": "No code extracted from output"},
            }

        # Try to compile
        compiles = False
        compile_errors = []
        compile_warnings = []

        try:
            evaluator = self._get_evaluator()
            compiles, bytecode, errors, warnings = evaluator.compile_code(
                f"org $008000\n{generated_code}"
            )
            compile_errors = errors
            compile_warnings = warnings
        except Exception as e:
            compile_errors = [str(e)]

        # Check entity usage
        entity_info = self._check_alttp_entities(generated_code)

        # Count instructions
        instruction_count = self._count_instructions(generated_code)

        # Calculate quality score
        score = 0.0

        if compiles:
            score += 0.4  # Base for compilation

        # Structure quality
        if entity_info["has_structure"]:
            score += 0.1
        if entity_info["has_labels"]:
            score += 0.1
        if entity_info["has_comments"]:
            score += 0.1

        # Entity coverage
        expected_entities = item.metadata.get("expected_entities", [])
        if expected_entities:
            matched = len(set(expected_entities) & set(entity_info["entities_found"]))
            entity_score = matched / len(expected_entities) if expected_entities else 0
            score += entity_score * 0.3
        else:
            # Reward any entity usage
            score += min(0.3, entity_info["entity_count"] * 0.05)

        # Passed if compiles and has reasonable structure
        passed = compiles and instruction_count >= 3 and entity_info["has_structure"]

        return {
            "passed": passed,
            "score": min(1.0, score),
            "metrics": {
                "compiles": 1.0 if compiles else 0.0,
                "entity_count": entity_info["entity_count"],
                "instruction_count": instruction_count,
                "has_structure": 1.0 if entity_info["has_structure"] else 0.0,
                "has_labels": 1.0 if entity_info["has_labels"] else 0.0,
                "has_comments": 1.0 if entity_info["has_comments"] else 0.0,
            },
            "details": {
                "generated_code": generated_code,
                "entities_found": entity_info["entities_found"],
                "compile_errors": compile_errors,
                "compile_warnings": compile_warnings,
            },
        }


class FaroreBenchmark(BenchmarkRunner):
    """Benchmark runner for Farore (debugging) expert."""

    domain = "farore"

    def _format_instruction(self, item: BenchmarkItem) -> str:
        """Format debugging instruction."""
        symptom = item.metadata.get("symptom", "incorrect behavior")
        return f"Debug this 65816 code and fix the bug:\n```\n{item.code}\n```\n\nSymptom: {symptom}"

    def _extract_fix(self, output: str) -> tuple[str, str]:
        """Extract bug identification and fixed code."""
        # Find bug description
        bug_match = re.search(r"(?:Bug|Issue|Problem|Error):\s*(.+?)(?:\n\n|Fix:|Corrected)", output, re.IGNORECASE | re.DOTALL)
        bug_desc = bug_match.group(1).strip() if bug_match else ""

        # Find fixed code
        code_match = re.search(r"```(?:asm|assembly|65816)?\n(.*?)```", output, re.DOTALL)
        fixed_code = code_match.group(1).strip() if code_match else ""

        # Fallback: look for "Fix:" or "Corrected:" section
        if not fixed_code:
            fix_match = re.search(r"(?:Fix|Corrected|Fixed)[:\s]*\n(.*?)(?:\n\n|$)", output, re.DOTALL)
            if fix_match:
                fixed_code = fix_match.group(1).strip()

        return bug_desc, fixed_code

    def evaluate_item(self, item: BenchmarkItem, output: str) -> dict[str, Any]:
        """Evaluate a debugging result."""
        bug_desc, fixed_code = self._extract_fix(output)

        # Check if bug was identified
        expected_issue = item.metadata.get("issue", "")
        bug_identified = False
        if expected_issue:
            # Check if any key terms from expected issue are mentioned
            issue_terms = set(expected_issue.lower().split())
            output_terms = set(output.lower().split())
            overlap = len(issue_terms & output_terms)
            bug_identified = overlap >= min(3, len(issue_terms) // 2)

        # Check if fix compiles
        fix_compiles = False
        if fixed_code:
            try:
                from ..evaluation.semantic_eval import SemanticEvaluator
                evaluator = SemanticEvaluator()
                fix_compiles, _, _, _ = evaluator.compile_code(f"org $008000\n{fixed_code}")
            except Exception:
                pass

        # Check if fix matches expected
        expected_fix = item.expected_output or ""
        fix_matches = False
        if fixed_code and expected_fix:
            # Normalize and compare
            def normalize(s: str) -> str:
                return re.sub(r'\s+', ' ', s.strip().lower())

            fix_matches = normalize(fixed_code) == normalize(expected_fix)

        # Calculate score
        score = 0.0
        if bug_identified:
            score += 0.3
        if fixed_code:
            score += 0.2
        if fix_compiles:
            score += 0.3
        if fix_matches:
            score += 0.2

        passed = bug_identified and fix_compiles

        return {
            "passed": passed,
            "score": min(1.0, score),
            "metrics": {
                "bug_identified": 1.0 if bug_identified else 0.0,
                "fix_provided": 1.0 if fixed_code else 0.0,
                "fix_compiles": 1.0 if fix_compiles else 0.0,
                "fix_matches": 1.0 if fix_matches else 0.0,
            },
            "details": {
                "bug_description": bug_desc,
                "fixed_code": fixed_code,
            },
        }


class VeranBenchmark(BenchmarkRunner):
    """Benchmark runner for Veran (explanation) expert."""

    domain = "veran"

    def _format_instruction(self, item: BenchmarkItem) -> str:
        """Format explanation instruction."""
        return f"Explain what this 65816 assembly code does:\n```\n{item.code}\n```"

    def _extract_concepts(self, output: str) -> list[str]:
        """Extract technical concepts from explanation."""
        # Common 65816/SNES concepts
        CONCEPT_PATTERNS = [
            r"\baccumulator\b",
            r"\bindex register\b",
            r"\bstack\b",
            r"\bzero page\b",
            r"\baddressing mode\b",
            r"\bimmediate\b",
            r"\bdirect\b",
            r"\bindexed\b",
            r"\bindirect\b",
            r"\bbranch\b",
            r"\bloop\b",
            r"\bsubroutine\b",
            r"\binterrupt\b",
            r"\bDMA\b",
            r"\bVRAM\b",
            r"\bWRAM\b",
            r"\bOAM\b",
            r"\bpalette\b",
            r"\bsprite\b",
            r"\btilemap\b",
            r"\bVBLANK\b",
            r"\bNMI\b",
            r"\bcarry\b",
            r"\boverflow\b",
            r"\bzero flag\b",
            r"\bnegative flag\b",
            r"\b16-bit\b",
            r"\b8-bit\b",
            r"\bmode switch\b",
            r"\bREP\b",
            r"\bSEP\b",
        ]

        found = []
        output_lower = output.lower()
        for pattern in CONCEPT_PATTERNS:
            if re.search(pattern, output_lower):
                found.append(pattern.replace(r"\b", "").replace("\\", ""))

        return found

    def evaluate_item(self, item: BenchmarkItem, output: str) -> dict[str, Any]:
        """Evaluate an explanation result."""
        # Extract concepts mentioned
        concepts_found = self._extract_concepts(output)

        # Check expected concepts
        expected_concepts = item.metadata.get("concepts", [])
        if expected_concepts:
            matched = len({c.lower() for c in expected_concepts} & {c.lower() for c in concepts_found})
            concept_recall = matched / len(expected_concepts)
        else:
            concept_recall = min(1.0, len(concepts_found) / 5)  # Expect at least 5 concepts

        # Check for technical accuracy indicators
        has_register_refs = bool(re.search(r"\b[AXY]\b", output))  # References A, X, Y registers
        has_address_refs = bool(re.search(r"\$[0-9A-Fa-f]{2,6}", output))  # References addresses
        has_opcode_refs = bool(re.search(r"\b(LDA|STA|LDX|STX|JSR|JMP|BEQ|BNE)\b", output))

        # Calculate length quality (not too short, not too long)
        word_count = len(output.split())
        length_score = 1.0
        if word_count < 50:
            length_score = word_count / 50
        elif word_count > 500:
            length_score = max(0.5, 500 / word_count)

        # Calculate overall score
        score = 0.0
        score += concept_recall * 0.4
        score += (0.2 if has_register_refs else 0.0)
        score += (0.2 if has_address_refs else 0.0)
        score += (0.1 if has_opcode_refs else 0.0)
        score *= length_score

        # Passed if reasonable explanation with concepts
        passed = len(concepts_found) >= 2 and word_count >= 30

        return {
            "passed": passed,
            "score": min(1.0, score),
            "metrics": {
                "concepts_found": len(concepts_found),
                "concept_recall": concept_recall,
                "word_count": word_count,
                "has_register_refs": 1.0 if has_register_refs else 0.0,
                "has_address_refs": 1.0 if has_address_refs else 0.0,
            },
            "details": {
                "concepts": concepts_found,
            },
        }
