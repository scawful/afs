"""ASM training data augmentation generator.

Generates variations of 65816 SNES assembly training samples via:
- Instruction paraphrasing (Phase 1)
- Register swaps (Phase 2)
- Address variations (Phase 2)
- Style variations (Phase 2)
"""

from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .base import (
    BaseGenerator,
    GenerationResult,
    TrainingSample,
    clean_instruction,
    is_malformed_output,
    read_jsonl,
)


@dataclass
class AsmAugmentConfig:
    """Configuration for ASM augmentation."""

    # Paraphrase settings
    paraphrase_count: int = 5
    include_original: bool = True
    shuffle_output: bool = True

    # Filtering
    min_instruction_len: int = 10
    input_domains: tuple[str, ...] = ("asm", "asm_optimize", "asm_debug", "asm_hook")

    # Seeding
    random_seed: int | None = None


# =============================================================================
# Instruction Category Detection
# =============================================================================

# Patterns to detect instruction category
CATEGORY_PATTERNS = {
    "write": [
        r"^write\b",
        r"^create\b",
        r"^implement\b",
        r"^build\b",
        r"^make\b",
        r"^generate\b",
        r"^add\b.*routine",
        r"^add\b.*code",
        r"^design\b",
    ],
    "optimize": [
        r"^optimize\b",
        r"^speed\s*up\b",
        r"^make.*faster\b",
        r"^reduce.*cycles?\b",
        r"^improve.*performance\b",
        r"^refactor\b",
        r"\boptimiz",
    ],
    "debug": [
        r"^debug\b",
        r"^fix\b",
        r"^find.*bug\b",
        r"^trace\b",
        r"^why.*crash",
        r"\bcrash",
        r"\bbug\b",
        r"\berror\b",
        r"^investigate\b",
    ],
    "hook": [
        r"^hook\b",
        r"^inject\b",
        r"^patch\b",
        r"^modify\b.*at\b",
        r"^intercept\b",
        r"^hijack\b",
        r"\bhook\b",
        r"\bpatch\b",
    ],
    "explain": [
        r"^explain\b",
        r"^what does\b",
        r"^how does\b",
        r"^describe\b",
        r"^analyze\b",
        r"^walk.*through\b",
    ],
}


def detect_category(instruction: str) -> str:
    """Detect the category of an instruction."""
    instruction_lower = instruction.lower().strip()

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, instruction_lower, re.IGNORECASE):
                return category

    return "write"  # Default to write


def extract_action(instruction: str) -> str:
    """Extract the core action/target from an instruction."""
    instruction = instruction.strip()

    # Remove common prefixes for all categories
    prefixes = [
        # Write category
        r"^write\s+(a\s+)?",
        r"^create\s+(a\s+)?",
        r"^implement\s+(a\s+)?",
        r"^build\s+(a\s+)?",
        r"^make\s+(a\s+)?",
        r"^generate\s+(a\s+)?",
        r"^design\s+(a\s+)?",
        r"^add\s+(a\s+)?",
        r"^code\s+(a\s+)?",
        r"^develop\s+(a\s+)?",
        # Optimize category
        r"^optimize\s+(the\s+)?",
        r"^speed\s*up\s+(the\s+)?",
        r"^refactor\s+(the\s+)?",
        r"^improve\s+(the\s+)?",
        # Debug category
        r"^debug\s+(the\s+)?",
        r"^fix\s+(the\s+)?",
        r"^trace\s+(the\s+)?",
        r"^investigate\s+(the\s+)?",
        # Hook category
        r"^hook\s+(into\s+)?",
        r"^inject\s+(code\s+)?(into\s+)?",
        r"^patch\s+(the\s+)?",
        r"^intercept\s+(the\s+)?",
        # Explain category
        r"^explain\s+(step-by-step\s+)?(how\s+to\s+)?",
        r"^describe\s+(how\s+)?",
        r"^analyze\s+(the\s+)?",
        r"^what\s+does\s+",
        r"^how\s+does\s+",
        r"^walk\s+(me\s+)?through\s+",
        # General
        r"^show\s+(me\s+)?(how\s+to\s+)?",
        r"^provide\s+(a\s+)?",
        r"^i\s+need\s+(a\s+)?",
        r"^how\s+would\s+you\s+",
    ]

    action = instruction

    # Apply prefix removal repeatedly until no more matches
    # This handles cases like "Explain step-by-step how to implement..."
    changed = True
    max_iterations = 5
    iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        for prefix in prefixes:
            new_action = re.sub(prefix, "", action, count=1, flags=re.IGNORECASE)
            if new_action != action:
                action = new_action.strip()
                changed = True
                break

    # Also remove common suffixes
    suffixes = [
        r"\s+routine\.?$",
        r"\s+function\.?$",
        r"\s+code\.?$",
        r"\s+subroutine\.?$",
    ]

    for suffix in suffixes:
        action = re.sub(suffix, "", action, flags=re.IGNORECASE)

    return action.strip()


# =============================================================================
# Paraphrase Templates
# =============================================================================

PARAPHRASE_TEMPLATES = {
    "write": [
        "Write {action}",
        "Implement {action}",
        "Create {action}",
        "I need {action}",
        "How would you write {action}",
        "Build {action}",
        "Generate code for {action}",
        "Code {action}",
        "Develop {action}",
        "Write the assembly for {action}",
    ],
    "optimize": [
        "Optimize {action}",
        "Make {action} faster",
        "Reduce cycles in {action}",
        "This code is slow: {action}. Speed it up",
        "Speed up {action}",
        "Improve the performance of {action}",
        "Refactor {action} for efficiency",
        "How can I make {action} run faster",
        "{action} - optimize this for SNES",
        "Minimize cycle count for {action}",
    ],
    "debug": [
        "Debug {action}",
        "My game crashes when {action}. Find the bug",
        "Fix the bug in {action}",
        "This code causes issues: {action}",
        "Find the error in {action}",
        "Why does {action} crash the game",
        "Trace the problem with {action}",
        "Investigate why {action} fails",
        "{action} is broken - help me fix it",
        "What's wrong with {action}",
    ],
    "hook": [
        "Add a hook for {action}",
        "I want to trigger {action}",
        "Inject code at {action}",
        "Extend functionality to {action}",
        "Hook into {action}",
        "Patch the game to {action}",
        "Intercept {action}",
        "Modify the game to {action}",
        "Create a hook that {action}",
        "How do I inject code for {action}",
    ],
    "explain": [
        "Explain {action}",
        "What does {action} do",
        "How does {action} work",
        "Describe {action}",
        "Walk me through {action}",
        "Analyze {action}",
        "Break down {action}",
        "Help me understand {action}",
        "What's happening in {action}",
        "Can you explain {action}",
    ],
}


def generate_paraphrases(instruction: str, count: int = 5) -> list[str]:
    """Generate paraphrased versions of an instruction."""
    category = detect_category(instruction)
    action = extract_action(instruction)

    if not action:
        action = instruction

    templates = PARAPHRASE_TEMPLATES.get(category, PARAPHRASE_TEMPLATES["write"])

    # Select random templates
    if len(templates) <= count:
        selected = templates[:]
    else:
        selected = random.sample(templates, count)

    paraphrases = []
    for template in selected:
        paraphrased = template.format(action=action)
        # Capitalize first letter
        if paraphrased:
            paraphrased = paraphrased[0].upper() + paraphrased[1:]
        paraphrases.append(paraphrased)

    return paraphrases


# =============================================================================
# ASM Augmentation Generator
# =============================================================================


class AsmAugmentGenerator(BaseGenerator):
    """Generator that augments ASM training samples via paraphrasing."""

    def __init__(
        self,
        input_path: Path | None = None,
        config: AsmAugmentConfig | None = None,
    ):
        super().__init__(name="AsmAugmentGenerator", domain="asm-augmented")
        self.input_path = Path(input_path) if input_path else None
        self.config = config or AsmAugmentConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def generate(self) -> GenerationResult:
        """Generate augmented training samples."""
        result = GenerationResult()

        # Load source samples
        if self.input_path is None:
            raise ValueError("input_path must be set when calling generate()")
        source_samples = read_jsonl(self.input_path)
        result.source_count = len(source_samples)

        for sample in source_samples:
            try:
                augmented = self._augment_sample(sample)
                result.samples.extend(augmented)
            except Exception as e:
                result.errors.append(f"Error augmenting {sample.sample_id}: {e}")
                result.skipped += 1

        # Shuffle if configured
        if self.config.shuffle_output:
            random.shuffle(result.samples)

        return result

    def _augment_sample(self, sample: TrainingSample) -> list[TrainingSample]:
        """Augment a single sample."""
        results = []

        # Skip if domain is not in allowed list
        if sample.domain not in self.config.input_domains:
            return []

        # Skip samples with malformed outputs (e.g., file markers as summaries)
        if is_malformed_output(sample.output, sample.instruction):
            return []

        # Clean the instruction before processing
        cleaned_instruction, was_cleaned = clean_instruction(sample.instruction)

        # Skip if instruction is too short after cleaning
        if len(cleaned_instruction) < self.config.min_instruction_len:
            return []

        # Skip if instruction is empty after cleaning
        if not cleaned_instruction.strip():
            return []

        # Include original if configured (with cleaned instruction)
        if self.config.include_original:
            metadata = {"augmentation": "original", "parent_id": sample.sample_id}
            if was_cleaned:
                metadata["instruction_cleaned"] = True
                metadata["original_instruction"] = sample.instruction

            original = TrainingSample(
                instruction=cleaned_instruction,
                output=sample.output,
                input=sample.input,
                domain=sample.domain,
                source=f"{sample.source}:original",
                sample_id=sample.sample_id,
                quality_score=sample.quality_score,
                embedding=sample.embedding,
                teacher_model=sample.teacher_model,
                teacher_prompt=sample.teacher_prompt,
                timestamp=sample.timestamp,
                kg_entities=sample.kg_entities,
                kg_validated=sample.kg_validated,
                _metadata=metadata,
            )
            results.append(original)

        # Generate paraphrases from the cleaned instruction
        paraphrases = generate_paraphrases(
            cleaned_instruction,
            count=self.config.paraphrase_count,
        )

        for i, paraphrased_instruction in enumerate(paraphrases):
            metadata = {
                "augmentation": "paraphrase",
                "parent_id": sample.sample_id,
                "paraphrase_index": i,
                "source_instruction": cleaned_instruction,
            }
            if was_cleaned:
                metadata["instruction_cleaned"] = True
                metadata["original_raw_instruction"] = sample.instruction

            augmented = TrainingSample(
                instruction=paraphrased_instruction,
                output=sample.output,
                input=sample.input,
                domain=sample.domain,
                source=f"{sample.source}:paraphrased",
                sample_id=str(uuid.uuid4()),
                quality_score=sample.quality_score,
                embedding=None,  # Embeddings need recalculation
                teacher_model=sample.teacher_model,
                teacher_prompt="",  # Clear for augmented samples
                timestamp=datetime.now().isoformat(),
                kg_entities=sample.kg_entities,
                kg_validated=False,  # Needs re-validation
                _metadata=metadata,
            )
            results.append(augmented)

        return results


# =============================================================================
# Phase 2 Augmentation: Register Swaps
# =============================================================================

# Register swap mappings for 65816
# These swaps preserve semantics when the operations are symmetric
REGISTER_SWAP_OPCODES = {
    # A <-> X swaps
    ("LDA", "LDX"): True,
    ("STA", "STX"): True,
    ("INC A", "INX"): True,
    ("DEC A", "DEX"): True,
    ("TAX", "TXA"): True,  # Special: these are inverses
    # A <-> Y swaps
    ("LDA", "LDY"): True,
    ("STA", "STY"): True,
    ("INC A", "INY"): True,
    ("DEC A", "DEY"): True,
    ("TAY", "TYA"): True,
    # X <-> Y swaps
    ("LDX", "LDY"): True,
    ("STX", "STY"): True,
    ("INX", "INY"): True,
    ("DEX", "DEY"): True,
    ("TXY", "TYX"): True,
    ("CPX", "CPY"): True,
}

# Opcodes that use A register implicitly
A_REGISTER_OPCODES = {
    "LDA", "STA", "ADC", "SBC", "AND", "ORA", "EOR", "CMP",
    "ASL", "LSR", "ROL", "ROR", "INC", "DEC",
    "PHA", "PLA", "TAX", "TAY", "TXA", "TYA",
}

# Opcodes that use X register
X_REGISTER_OPCODES = {
    "LDX", "STX", "CPX", "INX", "DEX",
    "TAX", "TXA", "TXY", "TXS", "PHX", "PLX",
}

# Opcodes that use Y register
Y_REGISTER_OPCODES = {
    "LDY", "STY", "CPY", "INY", "DEY",
    "TAY", "TYA", "TYX", "PHY", "PLY",
}


@dataclass
class RegisterSwapConfig:
    """Configuration for register swap augmentation."""

    # Which swaps to allow
    allow_a_x_swap: bool = True
    allow_a_y_swap: bool = True
    allow_x_y_swap: bool = True

    # Validation
    validate_semantics: bool = True  # Check that swap preserves meaning

    # Output
    max_variants: int = 3


class RegisterSwapAugmenter:
    """Swap register usage while preserving semantics.

    For example:
        LDA $00 / TAX / STX $02  ->  LDX $00 / TXA / STA $02

    This creates equivalent code that uses different registers,
    helping the model learn that register choice is often flexible.
    """

    def __init__(self, config: RegisterSwapConfig | None = None):
        self.config = config or RegisterSwapConfig()

    def augment(self, sample: TrainingSample) -> list[TrainingSample]:
        """Generate register-swapped variants of a sample.

        Args:
            sample: Original training sample

        Returns:
            List of augmented samples (may be empty if no valid swaps)
        """
        results = []
        code = sample.output

        # Try each swap type
        swap_types = []
        if self.config.allow_x_y_swap:
            swap_types.append(("X", "Y"))
        if self.config.allow_a_x_swap:
            swap_types.append(("A", "X"))
        if self.config.allow_a_y_swap:
            swap_types.append(("A", "Y"))

        for reg1, reg2 in swap_types:
            swapped = self._try_swap(code, reg1, reg2)
            if swapped and swapped != code:
                augmented = TrainingSample(
                    instruction=sample.instruction,
                    output=swapped,
                    input=sample.input,
                    domain=sample.domain,
                    source=f"{sample.source}:reg_swap_{reg1}_{reg2}",
                    sample_id=str(uuid.uuid4()),
                    quality_score=0.0,  # Needs rescoring
                    embedding=None,
                    teacher_model=sample.teacher_model,
                    teacher_prompt="",
                    timestamp=datetime.now().isoformat(),
                    kg_entities=sample.kg_entities,
                    kg_validated=False,
                    _metadata={
                        "augmentation": "register_swap",
                        "parent_id": sample.sample_id,
                        "swap_type": f"{reg1}<->{reg2}",
                    },
                )
                results.append(augmented)

                if len(results) >= self.config.max_variants:
                    break

        return results

    def _try_swap(self, code: str, reg1: str, reg2: str) -> str | None:
        """Try to swap two registers in code.

        Returns swapped code or None if swap not valid.
        """
        lines = code.split("\n")
        new_lines = []
        made_changes = False

        for line in lines:
            new_line = self._swap_line(line, reg1, reg2)
            if new_line != line:
                made_changes = True
            new_lines.append(new_line)

        if not made_changes:
            return None

        return "\n".join(new_lines)

    def _swap_line(self, line: str, reg1: str, reg2: str) -> str:
        """Swap registers in a single line."""
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            return line

        # Extract opcode
        match = re.match(r"^(\s*)([A-Za-z]{3})\b(.*)$", line)
        if not match:
            return line

        indent, opcode, rest = match.groups()
        opcode_upper = opcode.upper()

        # Determine swap mapping
        if reg1 == "X" and reg2 == "Y":
            swap_map = {
                "LDX": "LDY", "LDY": "LDX",
                "STX": "STY", "STY": "STX",
                "CPX": "CPY", "CPY": "CPX",
                "INX": "INY", "INY": "INX",
                "DEX": "DEY", "DEY": "DEX",
                "PHX": "PHY", "PHY": "PHX",
                "PLX": "PLY", "PLY": "PLX",
                "TXY": "TYX", "TYX": "TXY",
            }
            # Also swap ,X and ,Y addressing
            if opcode_upper in swap_map:
                new_opcode = swap_map[opcode_upper]
                # Preserve case
                if opcode.isupper():
                    return f"{indent}{new_opcode}{rest}"
                return f"{indent}{new_opcode.lower()}{rest}"
            # Swap indexed addressing
            rest_swapped = rest.replace(",X", ",[X]").replace(",Y", ",X").replace("[X]", "Y")
            rest_swapped = rest_swapped.replace(",x", ",[x]").replace(",y", ",x").replace("[x]", "y")
            if rest_swapped != rest:
                return f"{indent}{opcode}{rest_swapped}"

        elif reg1 == "A" and reg2 == "X":
            # Skip if line has any indexing (LDX doesn't support ,X or ,Y in same way as LDA)
            if ",X" in rest.upper() or ",Y" in rest.upper():
                return line
            swap_map = {
                "LDA": "LDX", "LDX": "LDA",
                "STA": "STX", "STX": "STA",
                "PHA": "PHX", "PHX": "PHA",
                "PLA": "PLX", "PLX": "PLA",
                "TAX": "TXA", "TXA": "TAX",
            }
            if opcode_upper in swap_map:
                new_opcode = swap_map[opcode_upper]
                if opcode.isupper():
                    return f"{indent}{new_opcode}{rest}"
                return f"{indent}{new_opcode.lower()}{rest}"

        elif reg1 == "A" and reg2 == "Y":
            # Skip if line has any indexing (LDY doesn't support ,X or ,Y in same way as LDA)
            if ",X" in rest.upper() or ",Y" in rest.upper():
                return line
            swap_map = {
                "LDA": "LDY", "LDY": "LDA",
                "STA": "STY", "STY": "STA",
                "PHA": "PHY", "PHY": "PHA",
                "PLA": "PLY", "PLY": "PLA",
                "TAY": "TYA", "TYA": "TAY",
            }
            if opcode_upper in swap_map:
                new_opcode = swap_map[opcode_upper]
                if opcode.isupper():
                    return f"{indent}{new_opcode}{rest}"
                return f"{indent}{new_opcode.lower()}{rest}"

        return line


# =============================================================================
# Phase 2 Augmentation: Address Variations
# =============================================================================

@dataclass
class AddressVariationConfig:
    """Configuration for address variation augmentation."""

    # Which variations to generate
    direct_to_absolute: bool = True  # $XX -> $00XX
    absolute_to_long: bool = True  # $XXXX -> $7EXXXX (WRAM)
    add_indexing: bool = False  # $XXXX -> $XXXX,X (risky, disabled by default)

    # Validation
    validate_with_asar: bool = False  # Validate output compiles

    # Output
    max_variants: int = 2


class AddressVariationAugmenter:
    """Generate address mode variations.

    Transforms addresses between equivalent forms:
    - Direct page ($XX) <-> Absolute ($00XX)
    - Absolute ($XXXX) <-> Long ($7EXXXX for WRAM)

    This helps the model learn that different addressing modes
    can access the same memory locations.
    """

    def __init__(self, config: AddressVariationConfig | None = None):
        self.config = config or AddressVariationConfig()

    def augment(self, sample: TrainingSample) -> list[TrainingSample]:
        """Generate address variation variants.

        Args:
            sample: Original training sample

        Returns:
            List of augmented samples
        """
        results = []
        code = sample.output

        # Try direct page to absolute
        if self.config.direct_to_absolute:
            expanded = self._expand_direct_page(code)
            if expanded and expanded != code:
                results.append(self._make_sample(
                    sample, expanded, "dp_to_abs", "Direct page expanded to absolute"
                ))

        # Try absolute to long
        if self.config.absolute_to_long:
            expanded = self._expand_to_long(code)
            if expanded and expanded != code:
                results.append(self._make_sample(
                    sample, expanded, "abs_to_long", "Absolute expanded to long"
                ))

        return results[:self.config.max_variants]

    def _make_sample(
        self,
        original: TrainingSample,
        new_output: str,
        variation_type: str,
        description: str,
    ) -> TrainingSample:
        """Create augmented sample."""
        return TrainingSample(
            instruction=original.instruction,
            output=new_output,
            input=original.input,
            domain=original.domain,
            source=f"{original.source}:addr_{variation_type}",
            sample_id=str(uuid.uuid4()),
            quality_score=0.0,
            embedding=None,
            teacher_model=original.teacher_model,
            teacher_prompt="",
            timestamp=datetime.now().isoformat(),
            kg_entities=original.kg_entities,
            kg_validated=False,
            _metadata={
                "augmentation": "address_variation",
                "parent_id": original.sample_id,
                "variation_type": variation_type,
                "description": description,
            },
        )

    def _expand_direct_page(self, code: str) -> str | None:
        """Expand direct page addresses ($XX) to absolute ($00XX).

        Only expands addresses that look like direct page (2 hex digits).
        """
        # Pattern: opcode followed by $XX (but not $XXX or $XXXX)
        # Be careful not to match ,X or ,Y suffixes
        pattern = re.compile(
            r"(\b(?:LDA|STA|LDX|STX|LDY|STY|ADC|SBC|AND|ORA|EOR|CMP|CPX|CPY|"
            r"BIT|ASL|LSR|ROL|ROR|INC|DEC|STZ|TRB|TSB)\b\s+)"
            r"\$([0-9A-Fa-f]{2})(?![0-9A-Fa-f])",
            re.IGNORECASE
        )

        def expand(match):
            prefix = match.group(1)
            addr = match.group(2)
            return f"{prefix}$00{addr}"

        result = pattern.sub(expand, code)
        return result if result != code else None

    def _expand_to_long(self, code: str) -> str | None:
        """Expand absolute addresses ($XXXX) to long ($7EXXXX).

        Assumes WRAM (bank $7E) for most addresses.
        Only expands addresses in WRAM range (< $8000).
        """
        # Pattern: opcode followed by $XXXX (but not $XXXXXX)
        pattern = re.compile(
            r"(\b(?:LDA|STA|LDX|STX|LDY|STY|ADC|SBC|AND|ORA|EOR|CMP|CPX|CPY|"
            r"BIT|ASL|LSR|ROL|ROR|INC|DEC|STZ|TRB|TSB)\b\s+)"
            r"\$([0-9A-Fa-f]{4})(?![0-9A-Fa-f])",
            re.IGNORECASE
        )

        def expand(match):
            prefix = match.group(1)
            addr = match.group(2)
            # Only expand if address is in WRAM range
            addr_int = int(addr, 16)
            if addr_int < 0x8000:
                return f"{prefix}$7E{addr}"
            return match.group(0)  # Don't expand ROM addresses

        result = pattern.sub(expand, code)
        return result if result != code else None


# =============================================================================
# Phase 2 Augmentation: Style Variations
# =============================================================================

@dataclass
class StyleVariationConfig:
    """Configuration for style variation augmentation."""

    # Which variations to generate
    toggle_case: bool = True  # LDA <-> lda
    toggle_comments: bool = True  # Add/remove comment prefixes
    normalize_whitespace: bool = True  # Standardize indentation

    # Comment styles
    comment_prefix: str = ";"

    # Output
    max_variants: int = 2


class StyleVariationAugmenter:
    """Generate style variations of assembly code.

    Transforms code style:
    - Opcode case (LDA vs lda)
    - Comment formatting
    - Whitespace/indentation

    This helps the model be robust to different coding styles.
    """

    def __init__(self, config: StyleVariationConfig | None = None):
        self.config = config or StyleVariationConfig()

    def augment(self, sample: TrainingSample) -> list[TrainingSample]:
        """Generate style variants.

        Args:
            sample: Original training sample

        Returns:
            List of augmented samples
        """
        results = []
        code = sample.output

        # Toggle case
        if self.config.toggle_case:
            toggled = self._toggle_opcode_case(code)
            if toggled and toggled != code:
                results.append(self._make_sample(
                    sample, toggled, "case_toggle", "Opcode case toggled"
                ))

        # Normalize whitespace
        if self.config.normalize_whitespace:
            normalized = self._normalize_whitespace(code)
            if normalized and normalized != code:
                results.append(self._make_sample(
                    sample, normalized, "whitespace", "Whitespace normalized"
                ))

        return results[:self.config.max_variants]

    def _make_sample(
        self,
        original: TrainingSample,
        new_output: str,
        variation_type: str,
        description: str,
    ) -> TrainingSample:
        """Create augmented sample."""
        return TrainingSample(
            instruction=original.instruction,
            output=new_output,
            input=original.input,
            domain=original.domain,
            source=f"{original.source}:style_{variation_type}",
            sample_id=str(uuid.uuid4()),
            quality_score=0.0,
            embedding=None,
            teacher_model=original.teacher_model,
            teacher_prompt="",
            timestamp=datetime.now().isoformat(),
            kg_entities=original.kg_entities,
            kg_validated=False,
            _metadata={
                "augmentation": "style_variation",
                "parent_id": original.sample_id,
                "variation_type": variation_type,
                "description": description,
            },
        )

    def _toggle_opcode_case(self, code: str) -> str | None:
        """Toggle opcode case (upper <-> lower)."""
        lines = code.split("\n")
        new_lines = []

        # Detect current dominant case
        upper_count = 0
        lower_count = 0
        for line in lines:
            match = re.match(r"^\s*([A-Za-z]{3})\b", line)
            if match:
                op = match.group(1)
                if op.isupper():
                    upper_count += 1
                elif op.islower():
                    lower_count += 1

        # Toggle to opposite
        to_lower = upper_count > lower_count

        for line in lines:
            # Match opcode at start of line (allowing for labels/whitespace)
            match = re.match(r"^(\s*)([A-Za-z]{3})(\b.*)$", line)
            if match:
                indent, opcode, rest = match.groups()
                if to_lower:
                    new_lines.append(f"{indent}{opcode.lower()}{rest}")
                else:
                    new_lines.append(f"{indent}{opcode.upper()}{rest}")
            else:
                new_lines.append(line)

        result = "\n".join(new_lines)
        return result if result != code else None

    def _normalize_whitespace(self, code: str) -> str | None:
        """Normalize whitespace and indentation."""
        lines = code.split("\n")
        new_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                new_lines.append("")
                continue

            # Normalize tabs to spaces
            line = line.replace("\t", "    ")

            # Ensure consistent spacing after opcodes
            match = re.match(r"^(\s*)([A-Za-z]{3})(\s+)(.*)$", line)
            if match:
                indent, opcode, spacing, rest = match.groups()
                # Standardize to single space after opcode
                new_lines.append(f"{indent}{opcode} {rest.strip()}")
            else:
                new_lines.append(line)

        result = "\n".join(new_lines)
        return result if result != code else None


# =============================================================================
# Combined Phase 2 Augmenter
# =============================================================================

@dataclass
class Phase2AugmentConfig:
    """Configuration for all Phase 2 augmentation."""

    register_swap: RegisterSwapConfig = field(default_factory=RegisterSwapConfig)
    address_variation: AddressVariationConfig = field(default_factory=AddressVariationConfig)
    style_variation: StyleVariationConfig = field(default_factory=StyleVariationConfig)

    # Which augmenters to enable
    enable_register_swap: bool = True
    enable_address_variation: bool = True
    enable_style_variation: bool = True

    # Overall limits
    max_total_variants: int = 5


class Phase2Augmenter:
    """Combined Phase 2 augmentation (register swap, address, style).

    Use this to apply all Phase 2 augmentations to a sample.
    """

    def __init__(self, config: Phase2AugmentConfig | None = None):
        self.config = config or Phase2AugmentConfig()

        self.register_swap = RegisterSwapAugmenter(self.config.register_swap)
        self.address_variation = AddressVariationAugmenter(self.config.address_variation)
        self.style_variation = StyleVariationAugmenter(self.config.style_variation)

    def augment(self, sample: TrainingSample) -> list[TrainingSample]:
        """Apply all Phase 2 augmentations.

        Args:
            sample: Original training sample

        Returns:
            List of all augmented variants
        """
        results = []

        if self.config.enable_register_swap:
            results.extend(self.register_swap.augment(sample))

        if self.config.enable_address_variation:
            results.extend(self.address_variation.augment(sample))

        if self.config.enable_style_variation:
            results.extend(self.style_variation.augment(sample))

        # Limit total variants
        return results[:self.config.max_total_variants]
