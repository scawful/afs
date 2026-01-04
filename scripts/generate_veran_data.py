#!/usr/bin/env python3
"""
Generate training data for Veran (65816 code explanation expert).

Data sources:
1. Inverted din training examples (code -> explanation)
2. Synthetic code+explanation pairs
3. Pattern library (common idioms)

Usage:
    python scripts/generate_veran_data.py --all
    python scripts/generate_veran_data.py --invert-din
    python scripts/generate_veran_data.py --patterns
    python scripts/generate_veran_data.py --synthetic
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DIN_DATA = MODELS_DIR / "din_combined_training.jsonl"
VERAN_DATA = MODELS_DIR / "veran_explanation_training.jsonl"

# Pattern explanations for common 65816 idioms
PATTERN_LIBRARY = [
    # Memory clearing
    {
        "instruction": "Explain this 65816 code:\nSTZ $10\nSTZ $11\nSTZ $12",
        "output": "Clears three consecutive memory locations to zero.\n\nSTZ (Store Zero) is more efficient than LDA #$00 / STA for clearing memory. Each STZ directly stores zero without needing to load it first.",
        "source": "pattern",
        "category": "memory_clear"
    },
    # REP/SEP mode switching
    {
        "instruction": "Explain this 65816 code:\nREP #$20\nLDA $10\nSTA $20\nSEP #$20",
        "output": "16-bit memory copy operation.\n\n1. REP #$20: Switch accumulator to 16-bit mode\n2. LDA $10: Load 16-bit value from $10-$11\n3. STA $20: Store to $20-$21\n4. SEP #$20: Return to 8-bit mode\n\nThis copies 2 bytes in one operation instead of two separate 8-bit loads/stores.",
        "source": "pattern",
        "category": "mode_switch"
    },
    {
        "instruction": "Explain this 65816 code:\nREP #$30\nLDA #$0000\nTAX\nTAY\nSEP #$30",
        "output": "Clears all three main registers (A, X, Y) to zero.\n\n1. REP #$30: Switch both A and X/Y to 16-bit mode\n2. LDA #$0000: Load zero into A\n3. TAX: Transfer A to X (X = 0)\n4. TAY: Transfer A to Y (Y = 0)\n5. SEP #$30: Return to 8-bit mode\n\nEfficient way to initialize all registers at once.",
        "source": "pattern",
        "category": "initialization"
    },
    # Jump table pattern
    {
        "instruction": "Explain this 65816 code:\nLDA $10\nASL A\nTAX\nLDA.l JumpTable,X\nSTA $00\nLDA.l JumpTable+1,X\nSTA $01\nJMP ($0000)",
        "output": "Indexed indirect jump (jump table pattern).\n\n1. LDA $10: Load index value\n2. ASL A: Double it (table entries are 2-byte addresses)\n3. TAX: Use as X index\n4. Load 16-bit address from table into $00-$01\n5. JMP ($0000): Jump to that address\n\nUsed for state machines and switch-case logic. The index in $10 selects which routine to execute.",
        "source": "pattern",
        "category": "jump_table"
    },
    # DMA setup
    {
        "instruction": "Explain this 65816 code:\nLDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\nLDA #$00\nSTA $4312\nSTA $4313\nLDA #$80\nSTA $4314\nLDA #$01\nSTA $420B",
        "output": "VRAM DMA transfer setup.\n\nRegisters configured:\n- $4300 = $01: DMA mode 1 (write twice to same address)\n- $4301 = $18: Destination is $2118 (VRAM data port)\n- $4312-$4314: Source address in 24-bit format\n- $420B = $01: Trigger DMA on channel 0\n\nThis transfers data from CPU-accessible memory to VRAM. Common for loading graphics during VBlank.",
        "source": "pattern",
        "category": "dma"
    },
    # Increment pattern
    {
        "instruction": "Explain this 65816 code:\nINC $10",
        "output": "Increments the byte at address $10 by one.\n\nThis is the optimized form of:\nLDA $10\nCLC\nADC #$01\nSTA $10\n\nINC directly modifies memory without using the accumulator, saving cycles and code size.",
        "source": "pattern",
        "category": "arithmetic"
    },
    # Decrement pattern
    {
        "instruction": "Explain this 65816 code:\nDEC $10",
        "output": "Decrements the byte at address $10 by one.\n\nThis is the optimized form of:\nLDA $10\nSEC\nSBC #$01\nSTA $10\n\nDEC directly modifies memory, which is faster and smaller than load/subtract/store.",
        "source": "pattern",
        "category": "arithmetic"
    },
    # Loop countdown
    {
        "instruction": "Explain this 65816 code:\nLDX #$0F\n.loop\nLDA $1000,X\nSTA $2000,X\nDEX\nBPL .loop",
        "output": "Memory copy loop (16 bytes, countdown style).\n\n1. LDX #$0F: Start at index 15\n2. Load byte from $1000+X, store to $2000+X\n3. DEX: Decrement index\n4. BPL: Loop while X >= 0 (positive)\n\nCountdown loops are often faster than count-up because BPL checks the sign flag directly after DEX, avoiding a separate CPX comparison.",
        "source": "pattern",
        "category": "loop"
    },
    # TSB/TRB bit manipulation
    {
        "instruction": "Explain this 65816 code:\nLDA #$80\nTSB $10",
        "output": "Sets bit 7 in memory location $10.\n\nTSB (Test and Set Bits) ORs the accumulator with the memory location and stores the result. Equivalent to:\nLDA #$80\nORA $10\nSTA $10\n\nBut TSB does it in one instruction and also sets Z flag based on whether those bits were already set.",
        "source": "pattern",
        "category": "bit_ops"
    },
    {
        "instruction": "Explain this 65816 code:\nLDA #$80\nTRB $10",
        "output": "Clears bit 7 in memory location $10.\n\nTRB (Test and Reset Bits) clears the bits in memory that are set in A. Equivalent to:\nLDA $10\nAND #$7F\nSTA $10\n\nBut TRB is more efficient and also sets Z flag based on whether those bits were set before clearing.",
        "source": "pattern",
        "category": "bit_ops"
    },
    # BIT instruction
    {
        "instruction": "Explain this 65816 code:\nBIT $10\nBPL .positive\nBMI .negative",
        "output": "Tests the sign bit of memory location $10.\n\nBIT copies bit 7 of the operand to the N (negative) flag without modifying A. This allows branching based on the sign:\n- BPL: Branch if positive (bit 7 = 0)\n- BMI: Branch if negative (bit 7 = 1)\n\nUseful for checking flags or sign without loading the value.",
        "source": "pattern",
        "category": "branch"
    },
    # Indirect addressing
    {
        "instruction": "Explain this 65816 code:\nLDA ($10)",
        "output": "Indirect load from pointer at $10.\n\nReads the 16-bit address stored at $10-$11 and loads the byte at that address. This is the 65816's 'pointer dereference' operation.\n\nNote: LDA ($10) is shorter than LDY #$00 / LDA ($10),Y when Y=0 is not needed.",
        "source": "pattern",
        "category": "addressing"
    },
    # XBA swap
    {
        "instruction": "Explain this 65816 code:\nXBA",
        "output": "Exchanges the high and low bytes of the 16-bit accumulator.\n\nIf A = $1234, after XBA: A = $3412\n\nUseful for:\n- Accessing the high byte in 8-bit mode\n- Byte swapping operations\n- Temporary storage without using memory",
        "source": "pattern",
        "category": "register_ops"
    },
    # PHB/PLB bank setup
    {
        "instruction": "Explain this 65816 code:\nPHB\nPHK\nPLB\n; ... code ...\nPLB\nRTL",
        "output": "Data bank setup for subroutine.\n\n1. PHB: Save current data bank\n2. PHK: Push program bank (current code bank)\n3. PLB: Pull into data bank (DB = PB)\n4. ... execute code with DB = code bank ...\n5. PLB: Restore original data bank\n6. RTL: Return long\n\nThis pattern ensures data references use the same bank as the code, common in SNES games with multiple banks.",
        "source": "pattern",
        "category": "bank_ops"
    },
    # Two's complement negation
    {
        "instruction": "Explain this 65816 code:\nLDA $10\nEOR #$FF\nINC A",
        "output": "Two's complement negation (negate value).\n\n1. EOR #$FF: Flip all bits (one's complement)\n2. INC A: Add 1 to get two's complement\n\nResult: A = -($10) or (256 - $10) in unsigned terms.\n\nThis converts a positive value to its negative equivalent, or vice versa.",
        "source": "pattern",
        "category": "arithmetic"
    },
    # Carry chain addition
    {
        "instruction": "Explain this 65816 code:\nLDA $10\nCLC\nADC $11\nADC $12",
        "output": "Multi-byte addition with carry chain.\n\n1. LDA $10: Load first value\n2. CLC: Clear carry for clean start\n3. ADC $11: Add second value\n4. ADC $12: Add third value (including any carry from previous)\n\nThe second ADC doesn't need CLC because we want to include the carry from the first addition. Used for summing multiple values.",
        "source": "pattern",
        "category": "arithmetic"
    },
    # Screen brightness
    {
        "instruction": "Explain this 65816 code:\nLDA #$0F\nSTA $2100",
        "output": "Sets screen brightness to maximum.\n\n$2100 is the INIDISP register (screen display control):\n- Bits 0-3: Brightness (0=dark, F=full)\n- Bit 7: Force blank (0=display on, 1=blanked)\n\nLDA #$0F sets brightness to 15 (max) with display enabled. LDA #$8F would be max brightness but blanked.",
        "source": "pattern",
        "category": "hardware"
    },
    # Joypad reading
    {
        "instruction": "Explain this 65816 code:\nLDA $4218\nSTA $10\nLDA $4219\nSTA $11",
        "output": "Reads joypad 1 button state.\n\n$4218-$4219 contain the auto-read joypad data for controller 1:\n- $4218: B, Y, Select, Start, Up, Down, Left, Right\n- $4219: A, X, L, R (and 4 unused bits)\n\nCopying to $10-$11 allows the game to check button states without re-reading hardware registers.",
        "source": "pattern",
        "category": "hardware"
    },
    # MVN block move
    {
        "instruction": "Explain this 65816 code:\nREP #$30\nLDX #$0000\nLDY #$1000\nLDA #$00FF\nMVN $7E,$7E\nSEP #$30",
        "output": "Block memory move (256 bytes within bank $7E).\n\n1. REP #$30: 16-bit A, X, Y mode (required for MVN)\n2. LDX #$0000: Source address\n3. LDY #$1000: Destination address\n4. LDA #$00FF: Byte count minus 1 (256 bytes)\n5. MVN $7E,$7E: Move Next (source bank, dest bank)\n\nMVN copies A+1 bytes from X to Y, incrementing both. Very fast for large memory copies.",
        "source": "pattern",
        "category": "memory"
    },
]

# Explanations for din examples (inverted)
DIN_EXPLANATIONS = {
    # Key patterns from din training data
    "LDA #$00\nSTA $10\nLDA #$00\nSTA $11\nLDA #$00\nSTA $12":
        "Clears three memory locations by loading zero and storing it to each address. This is an unoptimized pattern - each location requires a separate LDA/STA pair even though the value is always zero.",

    "LDA $4218\nSTA $10\nLDA $4219\nSTA $11":
        "Reads the 16-bit joypad state from hardware registers $4218-$4219 and stores to $10-$11. This is the standard pattern for capturing controller input each frame.",

    "LDA $00\nCLC\nADC #$01\nSTA $00":
        "Increments the value at $00 by one. Loads the current value, adds 1 (with CLC to ensure clean addition), and stores back. This is the unoptimized form of INC $00.",

    "LDA $00\nSEC\nSBC #$01\nSTA $00":
        "Decrements the value at $00 by one. Loads current value, subtracts 1 (SEC sets borrow), stores back. This is the unoptimized form of DEC $00.",

    "LDA $10\nCMP #$00\nBEQ .zero":
        "Checks if $10 is zero and branches if so. The CMP #$00 is redundant because LDA already sets the Z flag based on whether the loaded value is zero.",

    "LDA $10\nAND #$FF":
        "Loads from $10 and ANDs with $FF. In 8-bit mode, AND #$FF has no effect since all bits are already within the 8-bit range. The AND is redundant.",

    "PHA\nPLA":
        "Pushes A to stack then immediately pulls it back. This is a no-op that wastes cycles. Sometimes used intentionally for timing, but usually indicates dead code.",

    "CLC\nCLC":
        "Clears the carry flag twice. The second CLC is redundant since carry is already clear after the first instruction.",

    "SEP #$20\nSEP #$10":
        "Sets 8-bit mode for accumulator, then sets 8-bit mode for index registers. These can be combined into a single SEP #$30 instruction.",

    "REP #$20\nREP #$10":
        "Sets 16-bit mode for accumulator, then for index registers. Can be combined into REP #$30 to set both in one instruction.",

    "LDA #$01\nCMP #$01\nBEQ .equal":
        "Loads 1 into A, compares with 1, branches if equal. Since we just loaded 1, the comparison will always succeed - this should just be BRA .equal.",

    "LDA #$80\nORA $10\nSTA $10":
        "Sets bit 7 of memory location $10. Loads $80 (bit 7 mask), ORs with current value, stores back. This is the unoptimized form of TSB $10.",

    "LDA $10\nAND #$7F\nSTA $10":
        "Clears bit 7 of memory location $10. Loads value, ANDs with $7F (clears bit 7), stores back. This is the unoptimized form of using TRB with mask $80.",

    "LDY #$00\nLDA ($10),Y":
        "Indirect load with Y=0. When the index is always zero, LDA ($10) achieves the same result with shorter code.",
}


def invert_din_examples():
    """Convert din optimization examples into explanation training data."""
    examples = []

    if not DIN_DATA.exists():
        print(f"Warning: {DIN_DATA} not found")
        return examples

    with open(DIN_DATA) as f:
        din_examples = [json.loads(line) for line in f if line.strip()]

    for ex in din_examples:
        # Extract the "before" code from din's instruction
        instruction = ex["instruction"]

        # Parse out the code block after "Optimize this 65816..."
        if ":\n" in instruction:
            code = instruction.split(":\n", 1)[1]
        else:
            continue

        # Look up explanation or generate generic one
        explanation = DIN_EXPLANATIONS.get(code)

        if not explanation:
            # Generate basic explanation for patterns we don't have specific explanations for
            explanation = generate_basic_explanation(code)

        if explanation:
            examples.append({
                "instruction": f"Explain this 65816 code:\n{code}",
                "output": explanation,
                "input": "",
                "domain": "asm",
                "source": "inverted_din",
                "intent": "explanation"
            })

    return examples


def generate_basic_explanation(code: str) -> str:
    """Generate a basic explanation for code we don't have specific explanations for."""
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]

    if not lines:
        return None

    # Count instruction types
    has_lda = any(l.startswith('LDA') for l in lines)
    has_sta = any(l.startswith('STA') for l in lines)
    has_stz = any(l.startswith('STZ') for l in lines)
    has_rep = any(l.startswith('REP') for l in lines)
    has_sep = any(l.startswith('SEP') for l in lines)
    has_branch = any(l.startswith(('BEQ', 'BNE', 'BPL', 'BMI', 'BCC', 'BCS', 'BRA')) for l in lines)
    has_jump = any(l.startswith(('JMP', 'JSR', 'JSL', 'RTL', 'RTS')) for l in lines)

    # Build basic description
    parts = []

    if has_stz:
        parts.append("Clears memory locations using STZ (store zero)")
    elif has_lda and has_sta:
        parts.append("Loads and stores values between memory locations")

    if has_rep and has_sep:
        parts.append("switches between 8-bit and 16-bit modes")
    elif has_rep:
        parts.append("switches to 16-bit mode")
    elif has_sep:
        parts.append("switches to 8-bit mode")

    if has_branch:
        parts.append("with conditional branching")
    if has_jump:
        parts.append("with jump/call instructions")

    if parts:
        return f"This code {', '.join(parts)}.\n\n{len(lines)} instructions total."

    return None


def get_pattern_examples():
    """Return the pattern library examples."""
    examples = []
    for pattern in PATTERN_LIBRARY:
        examples.append({
            "instruction": pattern["instruction"],
            "output": pattern["output"],
            "input": "",
            "domain": "asm",
            "source": pattern.get("source", "pattern"),
            "intent": "explanation"
        })
    return examples


def generate_synthetic_examples():
    """Generate additional synthetic code+explanation pairs."""
    # Template-based generation for common patterns
    templates = [
        # Simple loads
        {
            "instruction": "Explain this 65816 code:\nLDA #$42\nSTA $10",
            "output": "Stores the immediate value $42 into memory location $10.\n\n1. LDA #$42: Load the literal value $42 into accumulator\n2. STA $10: Store A to direct page address $10",
        },
        {
            "instruction": "Explain this 65816 code:\nLDA $20\nSTA $30",
            "output": "Copies a byte from $20 to $30.\n\n1. LDA $20: Load byte from address $20\n2. STA $30: Store to address $30\n\nSimple memory-to-memory copy using accumulator as intermediate.",
        },
        # Comparisons
        {
            "instruction": "Explain this 65816 code:\nLDA $10\nCMP #$05\nBCC .less_than",
            "output": "Compares value at $10 with 5, branches if less.\n\n1. LDA $10: Load value to compare\n2. CMP #$05: Compare with 5 (sets carry if A >= 5)\n3. BCC: Branch if carry clear (A < 5)\n\nUsed for range checking or conditional logic.",
        },
        {
            "instruction": "Explain this 65816 code:\nLDA $10\nCMP #$10\nBCS .greater_equal",
            "output": "Compares value at $10 with 16, branches if greater or equal.\n\n1. LDA $10: Load value\n2. CMP #$10: Compare with $10 (16 decimal)\n3. BCS: Branch if carry set (A >= 16)\n\nCarry is set when A >= operand in CMP.",
        },
        # Shifts
        {
            "instruction": "Explain this 65816 code:\nLDA $10\nASL A\nASL A",
            "output": "Multiplies value at $10 by 4.\n\n1. LDA $10: Load value\n2. ASL A: Shift left (multiply by 2)\n3. ASL A: Shift left again (total: multiply by 4)\n\nASL (Arithmetic Shift Left) is efficient multiplication by powers of 2.",
        },
        {
            "instruction": "Explain this 65816 code:\nLDA $10\nLSR A\nLSR A\nLSR A",
            "output": "Divides value at $10 by 8 (integer division).\n\n1. LDA $10: Load value\n2. Three LSR A: Shift right 3 times = divide by 8\n\nLSR (Logical Shift Right) is efficient division by powers of 2. Remainder is lost.",
        },
        # Index operations
        {
            "instruction": "Explain this 65816 code:\nLDX $10\nLDA $1000,X",
            "output": "Indexed array read using X as offset.\n\n1. LDX $10: Load index value from $10\n2. LDA $1000,X: Load from address ($1000 + X)\n\nThis is array access where $1000 is the base address and X is the index.",
        },
        {
            "instruction": "Explain this 65816 code:\nLDY #$00\n.loop\nLDA ($10),Y\nSTA ($12),Y\nINY\nCPY #$10\nBNE .loop",
            "output": "Copies 16 bytes from pointer at $10 to pointer at $12.\n\n1. Y starts at 0\n2. Indirect load from ($10)+Y\n3. Indirect store to ($12)+Y\n4. Increment Y, loop until Y = 16\n\nUseful when source/destination addresses are stored in zero page pointers.",
        },
        # Stack operations
        {
            "instruction": "Explain this 65816 code:\nPHA\nPHX\nPHY\n; ... code ...\nPLY\nPLX\nPLA",
            "output": "Saves and restores A, X, Y registers around some code.\n\n1. Push all registers to stack (A, X, Y)\n2. Execute intervening code\n3. Pull registers back in reverse order\n\nCommon at start/end of subroutines to preserve caller's registers.",
        },
        # Conditional set
        {
            "instruction": "Explain this 65816 code:\nLDA #$00\nLDX $10\nBEQ .done\nLDA #$01\n.done\nSTA $20",
            "output": "Sets $20 to 1 if $10 is non-zero, else 0.\n\n1. Start with A = 0 (assume false)\n2. Load X from $10 (sets Z flag)\n3. If zero, skip to store\n4. Otherwise set A = 1 (true)\n5. Store result to $20\n\nConverts any non-zero value to 1 (boolean conversion).",
        },
    ]

    examples = []
    for t in templates:
        examples.append({
            "instruction": t["instruction"],
            "output": t["output"],
            "input": "",
            "domain": "asm",
            "source": "synthetic",
            "intent": "explanation"
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate Veran training data")
    parser.add_argument("--invert-din", action="store_true", help="Generate from inverted din examples")
    parser.add_argument("--patterns", action="store_true", help="Include pattern library")
    parser.add_argument("--synthetic", action="store_true", help="Include synthetic examples")
    parser.add_argument("--all", action="store_true", help="Generate all data sources")
    parser.add_argument("--output", type=Path, default=VERAN_DATA, help="Output file")

    args = parser.parse_args()

    if args.all:
        args.invert_din = True
        args.patterns = True
        args.synthetic = True

    if not any([args.invert_din, args.patterns, args.synthetic]):
        parser.print_help()
        return

    all_examples = []

    if args.invert_din:
        print("Generating inverted din examples...")
        examples = invert_din_examples()
        print(f"  Generated {len(examples)} examples")
        all_examples.extend(examples)

    if args.patterns:
        print("Adding pattern library...")
        examples = get_pattern_examples()
        print(f"  Added {len(examples)} pattern examples")
        all_examples.extend(examples)

    if args.synthetic:
        print("Adding synthetic examples...")
        examples = generate_synthetic_examples()
        print(f"  Added {len(examples)} synthetic examples")
        all_examples.extend(examples)

    # Deduplicate by instruction
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = ex["instruction"]
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    print(f"\nTotal unique examples: {len(unique_examples)}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote to {args.output}")


if __name__ == "__main__":
    main()
