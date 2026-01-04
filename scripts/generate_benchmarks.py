#!/usr/bin/env python3
"""Generate benchmark datasets from template libraries.

Creates ~500 benchmark items across all 4 domains:
- Din: Optimization tasks from DIN_PATTERNS
- Farore: Bug fixing tasks from FARORE_BUGS
- Nayru: Code generation tasks from NAYRU_TEMPLATES
- Veran: Code explanation tasks from VERAN_EXAMPLES
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.generators.template_libraries import (
    DIN_PATTERNS,
    FARORE_BUGS,
    NAYRU_TEMPLATES,
    NAYRU_HARDWARE,
    VERAN_EXAMPLES,
    ORACLE_PATTERNS,
    ASAR_SYNTAX,
)


def generate_din_benchmarks() -> list[dict]:
    """Generate Din optimization benchmark items."""
    items = []
    item_id = 0

    difficulty_map = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    # Address variations for generating more test cases
    zp_addrs = ["$10", "$12", "$14", "$20", "$22", "$30", "$40", "$50"]
    abs_addrs = ["$1000", "$1100", "$2000", "$7E0100", "$7E0200"]
    values = ["#$00", "#$01", "#$10", "#$42", "#$FF"]
    regs = [("LDA", "STA"), ("LDX", "STX"), ("LDY", "STY")]

    for difficulty, categories in DIN_PATTERNS.items():
        diff_level = difficulty_map.get(difficulty, 1)

        for category, patterns in categories.items():
            for before, after, description in patterns:
                item_id += 1
                items.append({
                    "id": f"din_{difficulty}_{item_id:03d}",
                    "category": category,
                    "difficulty": diff_level,
                    "code": before,
                    "expected_output": after,
                    "metadata": {
                        "description": description,
                        "task": "optimize",
                    },
                    "expected_metrics": {}
                })

                # Generate variations for basic patterns
                if difficulty == "basic" and "$10" in before:
                    for addr in zp_addrs[1:4]:  # Add 3 variations
                        item_id += 1
                        var_before = before.replace("$10", addr).replace("$11", f"${int(addr[1:], 16)+1:02X}").replace("$12", f"${int(addr[1:], 16)+2:02X}")
                        var_after = after.replace("$10", addr).replace("$11", f"${int(addr[1:], 16)+1:02X}").replace("$12", f"${int(addr[1:], 16)+2:02X}")
                        items.append({
                            "id": f"din_{difficulty}_{item_id:03d}",
                            "category": category,
                            "difficulty": diff_level,
                            "code": var_before,
                            "expected_output": var_after,
                            "metadata": {
                                "description": description + f" (addr variation: {addr})",
                                "task": "optimize",
                            },
                        })

    # Add synthetic redundant load patterns
    for i, addr in enumerate(zp_addrs):
        for val in values[:3]:
            item_id += 1
            next_addr = f"${int(addr[1:], 16)+1:02X}"
            items.append({
                "id": f"din_synth_{item_id:03d}",
                "category": "redundant_loads",
                "difficulty": 1,
                "code": f"LDA {val}\nSTA {addr}\nLDA {val}\nSTA {next_addr}",
                "expected_output": f"LDA {val}\nSTA {addr}\nSTA {next_addr}" if val != "#$00" else f"STZ {addr}\nSTZ {next_addr}",
                "metadata": {"description": "Synthetic redundant load pattern", "task": "optimize"},
            })

    # Add synthetic mode switch patterns
    mode_patterns = [
        ("SEP #$20\nSEP #$10", "SEP #$30", "Combine 8-bit mode switches"),
        ("REP #$20\nREP #$10", "REP #$30", "Combine 16-bit mode switches"),
        ("SEP #$20\nNOP\nSEP #$20", "SEP #$20\nNOP", "Remove redundant SEP"),
        ("REP #$20\nLDA $10\nREP #$20", "REP #$20\nLDA $10", "Remove redundant REP"),
    ]
    for before, after, desc in mode_patterns:
        item_id += 1
        items.append({
            "id": f"din_mode_{item_id:03d}",
            "category": "register_mode",
            "difficulty": 1,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add increment/decrement patterns
    for addr in zp_addrs[:5]:
        item_id += 1
        items.append({
            "id": f"din_inc_{item_id:03d}",
            "category": "increment_decrement",
            "difficulty": 1,
            "code": f"LDA {addr}\nCLC\nADC #$01\nSTA {addr}",
            "expected_output": f"INC {addr}",
            "metadata": {"description": "Use INC instead of LDA/ADC/STA", "task": "optimize"},
        })
        item_id += 1
        items.append({
            "id": f"din_dec_{item_id:03d}",
            "category": "increment_decrement",
            "difficulty": 1,
            "code": f"LDA {addr}\nSEC\nSBC #$01\nSTA {addr}",
            "expected_output": f"DEC {addr}",
            "metadata": {"description": "Use DEC instead of LDA/SBC/STA", "task": "optimize"},
        })

    # Add loop optimization patterns
    loop_sizes = [8, 16, 32, 64]
    for size in loop_sizes:
        item_id += 1
        items.append({
            "id": f"din_loop_{item_id:03d}",
            "category": "loop_optimization",
            "difficulty": 2,
            "code": f"LDX #$00\nloop:\nLDA $1000,X\nSTA $2000,X\nINX\nCPX #${size:02X}\nBNE loop",
            "expected_output": f"LDX #${size-1:02X}\nloop:\nLDA $1000,X\nSTA $2000,X\nDEX\nBPL loop",
            "metadata": {"description": f"Count down to avoid CPX (size={size})", "task": "optimize"},
        })

    # Add shift/multiply optimizations
    multiply_patterns = [
        ("ASL A\nASL A\nASL A", "ASL A\nASL A\nASL A", "Multiply by 8 via shifts"),
        ("LDA $10\nASL A\nCLC\nADC $10", "LDA $10\nSTA $00\nASL A\nADC $00", "Multiply by 3"),
        ("LDA $10\nASL A\nASL A\nCLC\nADC $10", "LDA $10\nSTA $00\nASL A\nASL A\nADC $00", "Multiply by 5"),
        ("LDA $10\nASL A\nASL A\nASL A\nSEC\nSBC $10", "LDA $10\nSTA $00\nASL A\nASL A\nASL A\nSBC $00", "Multiply by 7"),
    ]
    for before, after, desc in multiply_patterns:
        item_id += 1
        items.append({
            "id": f"din_mult_{item_id:03d}",
            "category": "multiplication",
            "difficulty": 2,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add branch simplification patterns
    branch_patterns = [
        ("CMP #$00\nBEQ label", "BEQ label", "CMP #$00 redundant before BEQ"),
        ("CMP #$00\nBNE label", "BNE label", "CMP #$00 redundant before BNE"),
        ("LDA $10\nCMP #$00\nBEQ label", "LDA $10\nBEQ label", "LDA sets Z flag"),
        ("LDA $10\nCMP #$00\nBNE label", "LDA $10\nBNE label", "LDA sets Z flag"),
        ("AND #$FF\nBNE label", "BNE label", "AND #$FF is identity"),
        ("ORA #$00\nBNE label", "BNE label", "ORA #$00 is identity"),
        ("EOR #$00\nBNE label", "BNE label", "EOR #$00 is identity"),
        ("ASL A\nLSR A\nBNE label", "AND #$FE\nBNE label", "Shift pair clears bit 0"),
    ]
    for before, after, desc in branch_patterns:
        item_id += 1
        items.append({
            "id": f"din_branch_{item_id:03d}",
            "category": "branch_optimization",
            "difficulty": 1,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add addressing mode optimizations
    addressing_patterns = [
        ("LDA $7E0000", "LDA $0000", "Use absolute instead of long for bank $7E"),
        ("STA $7E0010", "STA $10", "Use zero page for low addresses"),
        ("LDA #$00\nLDA $10,X", "LDA $10,X", "Redundant LDA before indexed"),
        ("TXA\nTAY\nLDA table,Y", "LDA table,X", "Use X directly for index"),
        ("PHA\nTXA\nTAY\nPLA\nLDA table,Y", "LDA table,X", "Complex index transfer"),
    ]
    for before, after, desc in addressing_patterns:
        item_id += 1
        items.append({
            "id": f"din_addr_{item_id:03d}",
            "category": "addressing_mode",
            "difficulty": 2,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add 16-bit operation optimizations
    word_patterns = [
        ("LDA $10\nSTA $20\nLDA $11\nSTA $21", "REP #$20\nLDA $10\nSTA $20\nSEP #$20", "Use 16-bit copy"),
        ("STZ $10\nSTZ $11", "REP #$20\nSTZ $10\nSEP #$20", "Use 16-bit STZ"),
        ("LDA $10\nCLC\nADC $12\nSTA $14\nLDA $11\nADC $13\nSTA $15", "REP #$20\nCLC\nLDA $10\nADC $12\nSTA $14\nSEP #$20", "Use 16-bit add"),
        ("INC $10\nBNE +\nINC $11\n+", "REP #$20\nINC $10\nSEP #$20", "Use 16-bit increment"),
        ("LDA $10\nORA $11\nBNE label", "REP #$20\nLDA $10\nSEP #$20\nBNE label", "16-bit zero check"),
    ]
    for before, after, desc in word_patterns:
        item_id += 1
        items.append({
            "id": f"din_word_{item_id:03d}",
            "category": "16bit_optimization",
            "difficulty": 2,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add stack optimizations
    stack_patterns = [
        ("PHA\nPLA", "", "Push/pull with no use"),
        ("PHA\nTAX\nPLA", "TAX", "Save A around transfer"),
        ("PHA\nPHX\nPLX\nPLA", "PHA\nPLA", "Unnecessary X push"),
        ("PHP\nCLC\nPLP", "CLC", "Unnecessary processor save"),
        ("PHA\nLDA $10\nSTA $20\nPLA", "LDA $10\nSTA $20", "A not needed after"),
    ]
    for before, after, desc in stack_patterns:
        item_id += 1
        items.append({
            "id": f"din_stack_{item_id:03d}",
            "category": "stack_optimization",
            "difficulty": 1,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add dead code removal patterns
    dead_code_patterns = [
        ("LDA $10\nLDA $11", "LDA $11", "First LDA overwritten"),
        ("STA $10\nSTA $10", "STA $10", "Duplicate store"),
        ("STZ $10\nLDA #$00\nSTA $10", "STZ $10", "Store zero twice"),
        ("INC $10\nDEC $10", "", "Increment then decrement"),
        ("SEC\nCLC\nADC $10", "CLC\nADC $10", "SEC overwritten by CLC"),
        ("REP #$20\nSEP #$20\nLDA $10", "LDA $10", "Mode switch cancelled"),
        ("NOP\nNOP\nNOP", "", "Remove NOPs"),
    ]
    for before, after, desc in dead_code_patterns:
        item_id += 1
        items.append({
            "id": f"din_dead_{item_id:03d}",
            "category": "dead_code",
            "difficulty": 1,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add strength reduction patterns
    strength_patterns = [
        ("LDA $10\nCLC\nADC #$01\nSTA $10", "INC $10", "ADC #$01 to INC"),
        ("LDA $10\nSEC\nSBC #$01\nSTA $10", "DEC $10", "SBC #$01 to DEC"),
        ("LDA $10\nASL A\nSTA $10", "ASL $10", "In-memory shift"),
        ("LDA $10\nLSR A\nSTA $10", "LSR $10", "In-memory shift right"),
        ("LDA $10\nROL A\nSTA $10", "ROL $10", "In-memory rotate"),
        ("LDX $10\nINX\nSTX $10", "INC $10", "Via X to INC"),
        ("LDY $10\nDEY\nSTY $10", "DEC $10", "Via Y to DEC"),
    ]
    for before, after, desc in strength_patterns:
        item_id += 1
        items.append({
            "id": f"din_strength_{item_id:03d}",
            "category": "strength_reduction",
            "difficulty": 1,
            "code": before,
            "expected_output": after,
            "metadata": {"description": desc, "task": "optimize"},
        })

    # Add Oracle patterns for Din
    if "din" in ORACLE_PATTERNS:
        for name, pattern in ORACLE_PATTERNS["din"].items():
            if isinstance(pattern, tuple) and len(pattern) >= 3:
                before, after, desc = pattern[:3]
                item_id += 1
                items.append({
                    "id": f"din_oracle_{item_id:03d}",
                    "category": "oracle_" + name,
                    "difficulty": 3,  # Advanced
                    "code": before,
                    "expected_output": after,
                    "metadata": {
                        "description": desc,
                        "task": "optimize",
                        "source": "oracle-of-secrets",
                    },
                })

    return items


def generate_farore_benchmarks() -> list[dict]:
    """Generate Farore debugging benchmark items."""
    items = []
    item_id = 0

    difficulty_map = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    for difficulty, categories in FARORE_BUGS.items():
        diff_level = difficulty_map.get(difficulty, 1)

        for category, bugs in categories.items():
            for bug in bugs:
                item_id += 1
                items.append({
                    "id": f"farore_{difficulty}_{item_id:03d}",
                    "category": category,
                    "difficulty": diff_level,
                    "code": bug.get("buggy", ""),
                    "expected_output": bug.get("fix", ""),
                    "metadata": {
                        "issue": bug.get("issue", ""),
                        "explanation": bug.get("explanation", ""),
                        "symptom": bug.get("issue", "unexpected behavior"),
                    },
                })

    # Add synthetic mode mismatch bugs
    mode_bugs = [
        ("LDA #$1234\nSTA $10", "REP #$20\nLDA #$1234\nSTA $10\nSEP #$20", "16-bit value in 8-bit mode"),
        ("LDA #$ABCD\nSTA $20", "REP #$20\nLDA #$ABCD\nSTA $20\nSEP #$20", "16-bit value in 8-bit mode"),
        ("REP #$20\nLDA $10\nSEP #$20\nSTA $20", "REP #$20\nLDA $10\nSTA $20\nSEP #$20", "Store before mode switch"),
        ("LDX #$1000\nSTX $10", "REP #$10\nLDX #$1000\nSTX $10\nSEP #$10", "16-bit X in 8-bit mode"),
    ]
    for buggy, fix, issue in mode_bugs:
        item_id += 1
        items.append({
            "id": f"farore_mode_{item_id:03d}",
            "category": "mode_mismatch",
            "difficulty": 1,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Only low byte stored"},
        })

    # Add stack imbalance bugs
    stack_bugs = [
        ("PHA\nPHX\nJSR sub\nPLA\nRTS", "PHA\nPHX\nJSR sub\nPLX\nPLA\nRTS", "Missing PLX"),
        ("PHP\nPHA\nJSR sub\nPLA\nRTS", "PHP\nPHA\nJSR sub\nPLA\nPLP\nRTS", "Missing PLP"),
        ("PHY\nPHX\nPHA\nJSR sub\nPLA\nPLX\nRTS", "PHY\nPHX\nPHA\nJSR sub\nPLA\nPLX\nPLY\nRTS", "Missing PLY"),
    ]
    for buggy, fix, issue in stack_bugs:
        item_id += 1
        items.append({
            "id": f"farore_stack_{item_id:03d}",
            "category": "stack_imbalance",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Crash on RTS - wrong return address"},
        })

    # Add branch range bugs
    for distance in [150, 200, 256]:
        item_id += 1
        items.append({
            "id": f"farore_branch_{item_id:03d}",
            "category": "branch_range",
            "difficulty": 1,
            "code": f"BRA far_label  ; {distance} bytes away",
            "expected_output": "BRL far_label  ; Use long branch",
            "metadata": {"issue": f"Branch target {distance} bytes away exceeds BRA range", "symptom": "Assembler error"},
        })

    # Add DMA bugs (missing bank)
    dma_bugs = [
        ("LDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\nLDA #<src\nSTA $4302\nLDA #>src\nSTA $4303\nLDA #$01\nSTA $420B",
         "LDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\nLDA #<src\nSTA $4302\nLDA #>src\nSTA $4303\nLDA #^src\nSTA $4304\nLDA #$01\nSTA $420B",
         "Missing DMA source bank register $4304"),
    ]
    for buggy, fix, issue in dma_bugs:
        item_id += 1
        items.append({
            "id": f"farore_dma_{item_id:03d}",
            "category": "dma_issues",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong data transferred"},
        })

    # Add register corruption bugs
    reg_bugs = [
        ("LDA $10\nJSR calc\nSTA $20", "PHA\nJSR calc\nPLA\nSTA $20", "A corrupted by subroutine"),
        ("LDX $10\nJSR calc\nSTX $20", "PHX\nJSR calc\nPLX\nSTX $20", "X corrupted by subroutine"),
        ("TXA\nJSR calc\nTAX\nSTX $20", "PHX\nJSR calc\nPLX\nSTX $20", "Register transfer doesn't preserve"),
        ("LDY $10\nJSR calc\nSTY $20", "PHY\nJSR calc\nPLY\nSTY $20", "Y corrupted by subroutine"),
        ("LDA $10\nLDX $11\nJSR calc\nSTA $20\nSTX $21", "PHA\nPHX\nJSR calc\nPLX\nPLA\nSTA $20\nSTX $21", "A and X corrupted"),
    ]
    for buggy, fix, issue in reg_bugs:
        item_id += 1
        items.append({
            "id": f"farore_reg_{item_id:03d}",
            "category": "register_corruption",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong value stored"},
        })

    # Add carry flag bugs
    carry_bugs = [
        ("LDA $10\nADC $12\nSTA $14", "CLC\nLDA $10\nADC $12\nSTA $14", "Missing CLC before ADC"),
        ("LDA $10\nSBC $12\nSTA $14", "SEC\nLDA $10\nSBC $12\nSTA $14", "Missing SEC before SBC"),
        ("LDA $10\nADC #$10\nADC #$20\nSTA $14", "CLC\nLDA $10\nADC #$10\nCLC\nADC #$20\nSTA $14", "Carry propagation between ADCs"),
        ("CLC\nLDA $10\nADC $11\nLDA $12\nADC $13\nSTA $14", "CLC\nLDA $10\nADC $11\nCLC\nLDA $12\nADC $13\nSTA $14", "Carry not cleared between operations"),
    ]
    for buggy, fix, issue in carry_bugs:
        item_id += 1
        items.append({
            "id": f"farore_carry_{item_id:03d}",
            "category": "carry_flag",
            "difficulty": 1,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Incorrect arithmetic result"},
        })

    # Add VBLANK timing bugs
    vblank_bugs = [
        ("STA $2118", "LDA $4212\nAND #$80\nBEQ -\nSTA $2118", "VRAM write outside VBLANK"),
        ("STA $2122", "LDA $4212\nAND #$80\nBEQ -\nSTA $2122", "CGRAM write outside VBLANK"),
        ("STZ $2104", "LDA $4212\nAND #$80\nBEQ -\nSTZ $2104", "OAM write outside VBLANK"),
        ("LDA #$80\nSTA $2100\nSTA $2118", "LDA #$80\nSTA $2100\nWAI\nSTA $2118", "No wait after force blank"),
    ]
    for buggy, fix, issue in vblank_bugs:
        item_id += 1
        items.append({
            "id": f"farore_vblank_{item_id:03d}",
            "category": "vblank_timing",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Graphical corruption"},
        })

    # Add interrupt handling bugs
    irq_bugs = [
        ("IRQ:\nLDA $10\nSTA $20\nRTI", "IRQ:\nPHA\nLDA $10\nSTA $20\nPLA\nRTI", "IRQ doesn't preserve A"),
        ("IRQ:\nPHA\nLDA $10\nSTA $20\nPLA\nRTI", "IRQ:\nPHA\nPHX\nLDA $10\nSTA $20\nPLX\nPLA\nRTI", "IRQ doesn't preserve X (if used)"),
        ("NMI:\nINC $10\nRTS", "NMI:\nINC $10\nRTI", "Using RTS instead of RTI"),
        ("NMI:\nPHA\nPHX\nJSR handler\nPLA\nPLX\nRTI", "NMI:\nPHA\nPHX\nJSR handler\nPLX\nPLA\nRTI", "Stack pull order reversed"),
    ]
    for buggy, fix, issue in irq_bugs:
        item_id += 1
        items.append({
            "id": f"farore_irq_{item_id:03d}",
            "category": "interrupt_handling",
            "difficulty": 3,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Register corruption or crash"},
        })

    # Add addressing mode bugs
    addr_bugs = [
        ("LDA $00,X\n; X > $FF", "LDA $0000,X", "Zero page wrap-around with X > 255"),
        ("STA ($10)\n; DP not 0", "STA [$10]", "Direct page indirect vs long indirect"),
        ("LDA $10,X\n; accessing $7E00xx", "LDA $7E0010,X", "Assuming bank 0 for WRAM access"),
        ("JMP ($1000)", "JML [$1000]", "JMP indirect doesn't load bank"),
        ("JSR $018000", "JSL $018000", "Cross-bank call needs JSL"),
    ]
    for buggy, fix, issue in addr_bugs:
        item_id += 1
        items.append({
            "id": f"farore_addr_{item_id:03d}",
            "category": "addressing_mode",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Accessing wrong memory location"},
        })

    # Add comparison logic bugs
    cmp_bugs = [
        ("CMP #$10\nBCS greater", "CMP #$10\nBCS greater_or_equal", "BCS includes equal case"),
        ("CMP #$80\nBPL positive", "CMP #$80\nBCC less_than_128", "Sign flag vs unsigned compare"),
        ("LDA $10\nCMP $11\nBEQ equal\nBCS greater\nBCC less", "LDA $10\nCMP $11\nBEQ equal\nBCC less\nBCS greater", "BCS/BCC after equal check"),
        ("CPX #$00\nBEQ done", "DEX\nBMI done", "Simpler zero check"),
    ]
    for buggy, fix, issue in cmp_bugs:
        item_id += 1
        items.append({
            "id": f"farore_cmp_{item_id:03d}",
            "category": "comparison_logic",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong branch taken"},
        })

    # Add loop termination bugs
    loop_bugs = [
        ("LDX #$10\n.lp:\nDEX\nBNE .lp", "LDX #$10\n.lp:\nDEX\nBPL .lp", "BNE misses X=0 iteration"),
        ("LDY #$FF\n.lp:\nINY\nCPY #$10\nBNE .lp", "LDY #$00\n.lp:\nINY\nCPY #$10\nBNE .lp", "Starting Y at wrong value"),
        ("LDX #$00\n.lp:\nINX\nBNE .lp", "LDX #$00\n.lp:\nINX\nCPX #$10\nBNE .lp", "Infinite loop - no termination"),
        (".lp:\nDEC $10\nBNE .lp", "LDA $10\n.lp:\nDEC A\nBNE .lp\nSTA $10", "Modifying memory in tight loop"),
    ]
    for buggy, fix, issue in loop_bugs:
        item_id += 1
        items.append({
            "id": f"farore_loop_{item_id:03d}",
            "category": "loop_termination",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Loop runs wrong number of times"},
        })

    # Add 16-bit operation bugs
    word_bugs = [
        ("INC $10\nBNE +\nINC $11\n+", "REP #$20\nINC $10\nSEP #$20", "Manual 16-bit increment"),
        ("LDA $10\nCLC\nADC #$01\nSTA $10\nBCC +\nINC $11\n+", "REP #$20\nINC $10\nSEP #$20", "Manual carry propagation"),
        ("LDA $10\nSTA $20\nLDA $11\nSTA $21", "REP #$20\nLDA $10\nSTA $20\nSEP #$20", "Two 8-bit copies instead of one 16-bit"),
        ("STZ $10\nSTZ $11\nSTZ $12\nSTZ $13", "REP #$20\nSTZ $10\nSTZ $12\nSEP #$20", "Four STZ instead of two 16-bit"),
    ]
    for buggy, fix, issue in word_bugs:
        item_id += 1
        items.append({
            "id": f"farore_word_{item_id:03d}",
            "category": "16bit_operations",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Inefficient or incorrect word operation"},
        })

    # Add off-by-one bugs
    offbyone_bugs = [
        ("LDX #$10\n.lp:\nLDA $1000,X\nSTA $2000,X\nDEX\nBNE .lp", "LDX #$0F\n.lp:\nLDA $1000,X\nSTA $2000,X\nDEX\nBPL .lp", "BNE misses index 0"),
        ("LDY #$00\n.lp:\nSTA $1000,Y\nINY\nCPY #$10\nBCC .lp", "LDY #$00\n.lp:\nSTA $1000,Y\nINY\nCPY #$10\nBNE .lp", "BCC vs BNE for count"),
        ("LDA #$00\n.lp:\nINC A\nCMP #$10\nBCC .lp", "LDA #$01\n.lp:\nINC A\nCMP #$10\nBCC .lp", "Starting at wrong value"),
        ("LDX count\n.lp:\nDEX\nBMI .done\nJSR process\nBRA .lp\n.done:", "LDX count\nBEQ .done\n.lp:\nDEX\nJSR process\nBNE .lp\n.done:", "Process called extra time"),
    ]
    for buggy, fix, issue in offbyone_bugs:
        item_id += 1
        items.append({
            "id": f"farore_obo_{item_id:03d}",
            "category": "off_by_one",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong iteration count"},
        })

    # Add pointer bugs
    pointer_bugs = [
        ("LDA #<ptr\nSTA $10", "LDA #<ptr\nSTA $10\nLDA #>ptr\nSTA $11", "Missing high byte of pointer"),
        ("LDA ($10)\nSTA $20", "LDA ($10),Y\nSTA $20", "Missing Y index for indirect"),
        ("LDA [$10]\nSTA $20", "LDA [$10]\nSTA $20\nLDA #^bank\nSTA $12", "Long indirect missing bank"),
        ("LDA table,X\nSTA ($20)", "LDA table,X\nLDY #$00\nSTA ($20),Y", "Indirect store needs Y"),
    ]
    for buggy, fix, issue in pointer_bugs:
        item_id += 1
        items.append({
            "id": f"farore_ptr_{item_id:03d}",
            "category": "pointer_bugs",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Accessing wrong memory"},
        })

    # Add timing bugs
    timing_bugs = [
        ("LDA $4218\nAND #$80\nBNE pressed", "LDA $4212\nAND #$01\nBNE -\nLDA $4218\nAND #$80\nBNE pressed", "Reading joypad during auto-read"),
        ("STA $2118\nSTA $2118", "STA $2118\nLDA $2139\nSTA $2118", "Back-to-back VRAM writes"),
        ("LDA $2134\nSTA $10", "LDA #$00\nSTA $211B\nLDA #$00\nSTA $211C\nLDA $2134\nSTA $10", "Reading multiplier without delay"),
        ("STA $4202\nSTA $4203\nLDA $4216", "STA $4202\nSTA $4203\nNOP\nNOP\nLDA $4216", "Reading multiply result too fast"),
    ]
    for buggy, fix, issue in timing_bugs:
        item_id += 1
        items.append({
            "id": f"farore_timing_{item_id:03d}",
            "category": "timing_issues",
            "difficulty": 3,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Incorrect or corrupted data"},
        })

    # Add bank boundary bugs
    bank_bugs = [
        ("JSR $FF00\n; crosses bank", "JSL $01FF00", "JSR can't cross bank boundary"),
        ("JMP $FFFF\n; next instruction at $10000", "JML $010000", "JMP wraps within bank"),
        ("BRA +127\n; target is 200 bytes away", "BRL target", "BRA range exceeded"),
        ("LDA $FFFF,X\n; X=$10", "LDA.l $00FFFF,X", "Indexed access crosses bank"),
    ]
    for buggy, fix, issue in bank_bugs:
        item_id += 1
        items.append({
            "id": f"farore_bank_{item_id:03d}",
            "category": "bank_boundary",
            "difficulty": 3,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Jump to wrong location"},
        })

    # Add flag state bugs
    flag_bugs = [
        ("PHP\nREP #$20\nPLP\nLDA $10", "PHP\nREP #$20\nLDA $10\nPLP", "PLP restores wrong mode"),
        ("SEI\nJSR handler\nCLI", "PHP\nSEI\nJSR handler\nPLP", "CLI unconditionally enables IRQ"),
        ("CLV\nADC $10\nBVC nooverflow", "ADC $10\nBVC nooverflow", "CLV before ADC hides overflow"),
        ("SEC\nROR A\nCLC\nROR A", "ROR A\nROR A", "Unnecessary flag manipulation"),
    ]
    for buggy, fix, issue in flag_bugs:
        item_id += 1
        items.append({
            "id": f"farore_flag_{item_id:03d}",
            "category": "flag_state",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong flag state"},
        })

    # Add initialization bugs
    init_bugs = [
        ("LDA $10\nBEQ done", "STZ $10\nLDA $10\nBEQ done", "Uninitialized memory read"),
        ("LDX #$00\nSTX $10\n.lp:\nINC $10\nBNE .lp", "STZ $10\n.lp:\nINC $10\nBNE .lp", "Redundant LDX for STZ"),
        ("LDA $10\nSTA $20", "STZ $10\nLDA $10\nSTA $20", "Missing init before use"),
        ("TXA\nSTA $10", "PHX\nTXA\nSTA $10\nPLX", "TXA clobbers A without save"),
    ]
    for buggy, fix, issue in init_bugs:
        item_id += 1
        items.append({
            "id": f"farore_init_{item_id:03d}",
            "category": "initialization",
            "difficulty": 1,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Undefined behavior"},
        })

    # Add signed arithmetic bugs
    signed_bugs = [
        ("CMP #$80\nBCS negative", "CMP #$80\nBMI negative", "BCS doesn't work for signed"),
        ("LDA $10\nCMP #$00\nBCS positive", "LDA $10\nBPL positive", "Signed positive check"),
        ("LDA $10\nSEC\nSBC $11\nBMI less", "LDA $10\nCMP $11\nBMI less", "Subtraction vs compare"),
        ("LDA $10\nCLC\nADC #$80\nBCS overflow", "LDA $10\nCLC\nADC #$80\nBVS overflow", "Signed overflow check"),
    ]
    for buggy, fix, issue in signed_bugs:
        item_id += 1
        items.append({
            "id": f"farore_signed_{item_id:03d}",
            "category": "signed_arithmetic",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong comparison result"},
        })

    # Add memory access bugs
    mem_bugs = [
        ("LDA $1FFF,X\n; X = $10", "LDA $1FFF,X", "Correct but document page crossing"),
        ("LDA ($FE),Y\n; Y > $01", "LDA [$FE],Y", "Indirect wrap at page boundary"),
        ("STA $FFFF", "STA.l $00FFFF", "Ambiguous bank access"),
        ("LDA $2100", "LDA.l $002100", "Hardware register needs long"),
    ]
    for buggy, fix, issue in mem_bugs:
        item_id += 1
        items.append({
            "id": f"farore_mem_{item_id:03d}",
            "category": "memory_access",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Reading wrong address"},
        })

    # Add subroutine bugs
    sub_bugs = [
        ("JSR far_routine\n; in different bank", "JSL far_routine", "Cross-bank call needs JSL"),
        ("JSR routine\nRTL", "JSL routine\nRTL", "Mismatched JSR/RTL"),
        ("JSL routine\nRTS", "JSL routine\nRTL", "Mismatched JSL/RTS"),
        ("JSR ($1000)\n; indirect call", "JSR ($1000,X)", "Wrong indirect call mode"),
    ]
    for buggy, fix, issue in sub_bugs:
        item_id += 1
        items.append({
            "id": f"farore_sub_{item_id:03d}",
            "category": "subroutine_call",
            "difficulty": 2,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Wrong return or crash"},
        })

    # Add bit manipulation bugs
    bit_bugs = [
        ("LDA $10\nAND #$80\nBNE set", "LDA $10\nBMI set", "Use BMI instead of AND"),
        ("LDA $10\nAND #$01\nBNE odd", "LDA $10\nLSR A\nBCS odd", "Use carry for bit 0"),
        ("LDA $10\nORA #$00\nSTA $10", "LDA $10\nSTA $10", "ORA #$00 does nothing"),
        ("LDA $10\nEOR #$00\nSTA $10", "LDA $10\nSTA $10", "EOR #$00 does nothing"),
        ("LDA $10\nAND #$FF\nSTA $10", "LDA $10\nSTA $10", "AND #$FF does nothing"),
    ]
    for buggy, fix, issue in bit_bugs:
        item_id += 1
        items.append({
            "id": f"farore_bit_{item_id:03d}",
            "category": "bit_manipulation",
            "difficulty": 1,
            "code": buggy,
            "expected_output": fix,
            "metadata": {"issue": issue, "symptom": "Inefficient or incorrect"},
        })

    # Add Oracle Farore patterns
    if "farore" in ORACLE_PATTERNS:
        for name, bug_data in ORACLE_PATTERNS["farore"].items():
            if isinstance(bug_data, dict):
                item_id += 1
                items.append({
                    "id": f"farore_oracle_{item_id:03d}",
                    "category": "oracle_" + name,
                    "difficulty": 3,
                    "code": bug_data.get("buggy", ""),
                    "expected_output": bug_data.get("fix", ""),
                    "metadata": {
                        "issue": bug_data.get("issue", ""),
                        "explanation": bug_data.get("explanation", ""),
                        "source": "oracle-of-secrets",
                    },
                })

    return items


def generate_nayru_benchmarks() -> list[dict]:
    """Generate Nayru code generation benchmark items."""
    items = []
    item_id = 0

    difficulty_map = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    for difficulty, templates in NAYRU_TEMPLATES.items():
        diff_level = difficulty_map.get(difficulty, 1)

        for template in templates:
            item_id += 1
            task = template.get("task", "")
            code = template.get("code", "")

            items.append({
                "id": f"nayru_{difficulty}_{item_id:03d}",
                "category": "generation",
                "difficulty": diff_level,
                "code": task,  # Task description as "code" field
                "expected_output": code.strip(),
                "metadata": {
                    "task": task,
                    "expected_entities": [],
                },
            })

    # Expanded hardware-based tasks
    hw_tasks = [
        # Link state
        ("link_state", "Read Link's X and Y coordinates", ["$7E0022", "$7E0020"]),
        ("link_state", "Check Link's facing direction", ["$7E002F"]),
        ("link_state", "Modify Link's health", ["$7E036C"]),
        ("link_state", "Get Link's current action state", ["$7E005D"]),
        ("link_state", "Set Link's movement speed", ["$7E0031"]),
        ("link_state", "Check if Link is at full health", ["$7E036C", "$7E036D"]),
        # Sprite/OAM
        ("sprite_oam", "Write a sprite to OAM buffer", ["$0800"]),
        ("sprite_oam", "Clear all sprites from OAM", ["$0800"]),
        ("sprite_oam", "Set sprite X position", ["$0800"]),
        ("sprite_oam", "Set sprite palette and priority", ["$0803"]),
        # PPU
        ("ppu_registers", "Set screen brightness to half", ["$2100"]),
        ("ppu_registers", "Configure VRAM auto-increment", ["$2115"]),
        ("ppu_registers", "Set BG mode to Mode 1", ["$2105"]),
        ("ppu_registers", "Enable force blank", ["$2100"]),
        ("ppu_registers", "Set VRAM address for write", ["$2116"]),
        # DMA
        ("dma_channels", "Set up DMA channel 0 for VRAM transfer", ["$4300", "$420B"]),
        ("dma_channels", "Configure DMA for OAM update", ["$4300"]),
        ("dma_channels", "Set DMA source address and bank", ["$4302", "$4304"]),
        # Joypad
        ("joypad", "Check if A button is pressed", ["$4219"]),
        ("joypad", "Wait for joypad auto-read to complete", ["$4212"]),
        ("joypad", "Check D-pad direction", ["$4218"]),
        ("joypad", "Detect newly pressed buttons", ["$4218"]),
        ("joypad", "Check if Start is pressed", ["$4218"]),
    ]

    # Add basic generation tasks
    basic_tasks = [
        ("Clear a range of memory to zero", ["STZ"]),
        ("Copy 16 bytes from source to destination", ["LDA", "STA"]),
        ("Implement a simple countdown timer", ["DEC", "BNE"]),
        ("Calculate A times 2", ["ASL"]),
        ("Calculate A times 4", ["ASL"]),
        ("Check if a value is zero", ["BEQ"]),
        ("Check if value is greater than threshold", ["CMP", "BCS"]),
        ("Swap high and low bytes of accumulator", ["XBA"]),
        ("Preserve registers before subroutine call", ["PHA", "PHX"]),
        ("Implement infinite loop with NMI wait", ["WAI"]),
        # More arithmetic
        ("Calculate A times 8", ["ASL"]),
        ("Calculate A divided by 2", ["LSR"]),
        ("Calculate A divided by 4", ["LSR"]),
        ("Add two 8-bit values and store result", ["CLC", "ADC"]),
        ("Subtract two 8-bit values", ["SEC", "SBC"]),
        ("Add two 16-bit values", ["REP", "ADC"]),
        ("Negate a signed value", ["EOR", "INC"]),
        ("Calculate absolute value", ["BPL", "EOR"]),
        ("Clamp value to maximum", ["CMP", "BCC"]),
        ("Calculate modulo by power of 2", ["AND"]),
        # Bit manipulation
        ("Set bit 7 of a memory location", ["ORA"]),
        ("Clear bit 7 of a memory location", ["AND"]),
        ("Toggle bit 0", ["EOR"]),
        ("Test if bit 3 is set", ["AND", "BNE"]),
        ("Rotate bits left through carry", ["ROL"]),
        ("Extract low nibble from byte", ["AND"]),
        ("Extract high nibble from byte", ["LSR"]),
        # Control flow
        ("Jump to address in table based on index", ["ASL", "TAX", "JMP"]),
        ("Call subroutine and return value in A", ["JSR"]),
        ("Branch if A equals specific value", ["CMP", "BEQ"]),
        ("Branch if A is less than value", ["CMP", "BCC"]),
        ("Implement switch/case with 4 options", ["CMP", "BEQ"]),
        # Memory operations
        ("Fill 256 bytes with a value", ["LDX", "STA"]),
        ("Copy memory block with overlap handling", ["LDX", "LDA", "STA"]),
        ("Swap two memory locations", ["LDA", "LDX", "STA"]),
        ("Compare two memory blocks", ["CMP", "BNE"]),
        ("Search for byte in array", ["CMP", "BEQ"]),
        # Stack operations
        ("Push 16-bit value to stack", ["REP", "PHA"]),
        ("Pop 16-bit value from stack", ["REP", "PLA"]),
        ("Save and restore all registers", ["PHP", "PHA", "PHX", "PHY"]),
        ("Create local variable space on stack", ["TSC", "SEC", "SBC"]),
    ]
    for task, entities in basic_tasks:
        item_id += 1
        items.append({
            "id": f"nayru_basic_{item_id:03d}",
            "category": "basic_ops",
            "difficulty": 1,
            "code": task,
            "metadata": {"task": task, "expected_entities": entities},
        })

    for hw_type, task, entities in hw_tasks:
        item_id += 1
        hw_info = NAYRU_HARDWARE.get(hw_type, {})
        context = hw_info.get("description", "")

        items.append({
            "id": f"nayru_hw_{item_id:03d}",
            "category": hw_type,
            "difficulty": 2,
            "code": task,
            "metadata": {
                "task": task,
                "context": context,
                "expected_entities": entities,
            },
        })

    # Add intermediate generation tasks
    intermediate_tasks = [
        ("Update sprite position based on velocity", ["sprite", "velocity"]),
        ("Implement horizontal collision detection", ["collision", "sprite"]),
        ("Implement vertical collision detection", ["collision", "sprite"]),
        ("Animate sprite by cycling through frames", ["animation", "timer"]),
        ("Implement entity state machine", ["state", "switch"]),
        ("Calculate distance between two points", ["subtraction", "absolute"]),
        ("Implement random number generator", ["seed", "multiply"]),
        ("Decompress simple RLE data", ["loop", "copy"]),
        ("Sort array of 8 values", ["comparison", "swap"]),
        ("Find minimum value in array", ["compare", "loop"]),
        ("Calculate CRC checksum", ["XOR", "rotate"]),
        ("Parse tile map data", ["pointer", "loop"]),
        ("Implement double buffering swap", ["pointer", "swap"]),
        ("Calculate signed multiplication", ["multiply", "sign"]),
        ("Implement smooth scrolling", ["scroll", "offset"]),
        ("Handle 16-bit signed comparison", ["sign", "compare"]),
        ("Implement circular buffer", ["wrap", "index"]),
        ("Calculate trajectory angle", ["atan", "lookup"]),
        ("Fade screen brightness smoothly", ["timer", "increment"]),
        ("Implement text drawing routine", ["font", "tile"]),
    ]
    for task, entities in intermediate_tasks:
        item_id += 1
        items.append({
            "id": f"nayru_inter_{item_id:03d}",
            "category": "intermediate_ops",
            "difficulty": 2,
            "code": task,
            "metadata": {"task": task, "expected_entities": entities},
        })

    # Add advanced generation tasks
    advanced_tasks = [
        ("Implement sprite DMA routine with double buffer", ["DMA", "$420B", "$4300"]),
        ("Set up Mode 7 rotation matrix", ["Mode 7", "$211B", "$211E"]),
        ("Configure HDMA for gradient effect", ["HDMA", "$4340"]),
        ("Implement APU communication protocol", ["SPC700", "$2140"]),
        ("Set up interrupt vectors", ["NMI", "IRQ", "vector"]),
        ("Implement fast memory copy via MVN", ["MVN", "block"]),
        ("Configure screen mode and layers", ["$2105", "$212C"]),
        ("Set up color math blending", ["$2130", "$2131"]),
        ("Implement window masking effect", ["$2123", "$2126"]),
        ("Configure mosaic effect per layer", ["$2106", "mosaic"]),
        ("Set up VRAM address increment mode", ["$2115", "VRAM"]),
        ("Implement palette rotation", ["CGRAM", "$2122"]),
        ("Configure Super FX chip registers", ["$3030", "SFX"]),
        ("Set up DSP-1 for mode 7 calculation", ["DSP", "$2180"]),
        ("Implement SA-1 memory mapping", ["SA-1", "BW-RAM"]),
    ]
    for task, entities in advanced_tasks:
        item_id += 1
        items.append({
            "id": f"nayru_adv_{item_id:03d}",
            "category": "advanced_ops",
            "difficulty": 3,
            "code": task,
            "metadata": {"task": task, "expected_entities": entities},
        })

    # Add expert ALTTP-specific tasks
    alttp_tasks = [
        ("Initialize overworld map state", ["$7E0010", "map"]),
        ("Load dungeon room data", ["$7E0480", "room"]),
        ("Spawn enemy sprite", ["$0D00", "sprite"]),
        ("Process ancilla projectile", ["$0C00", "ancilla"]),
        ("Update pushable block state", ["$7E0500", "block"]),
        ("Handle door transition", ["$7E0698", "door"]),
        ("Update boss AI state", ["boss", "state"]),
        ("Process menu cursor movement", ["$7E0C00", "menu"]),
        ("Handle item pickup and inventory", ["$7E0340", "item"]),
        ("Implement chest opening sequence", ["chest", "animation"]),
        ("Update Link's animation frame", ["$7E001C", "animation"]),
        ("Check for hookshot collision", ["ancilla", "collision"]),
        ("Process boomerang return", ["$0C00", "boomerang"]),
        ("Handle bomb timer and explosion", ["$0C00", "bomb"]),
        ("Update arrow trajectory", ["ancilla", "velocity"]),
        ("Process fairy bottle effect", ["$7E036C", "health"]),
        ("Handle magic meter update", ["$7E036E", "magic"]),
        ("Check for water tile collision", ["$7E0403", "collision"]),
        ("Handle pit fall damage", ["$7E005D", "action"]),
        ("Update sword charge timer", ["$7E003D", "sword"]),
        ("Process shield deflection", ["$7E003E", "shield"]),
        ("Handle pushing animation", ["$7E0028", "push"]),
    ]
    for task, entities in alttp_tasks:
        item_id += 1
        items.append({
            "id": f"nayru_alttp_{item_id:03d}",
            "category": "alttp_specific",
            "difficulty": 4,
            "code": task,
            "metadata": {"task": task, "expected_entities": entities, "game": "alttp"},
        })

    # Add Oracle Nayru patterns
    if "nayru" in ORACLE_PATTERNS:
        for name, code in ORACLE_PATTERNS["nayru"].items():
            if isinstance(code, str):
                item_id += 1
                items.append({
                    "id": f"nayru_oracle_{item_id:03d}",
                    "category": "oracle_" + name,
                    "difficulty": 3,
                    "code": f"Implement {name.replace('_', ' ')}",
                    "expected_output": code.strip(),
                    "metadata": {
                        "task": name.replace("_", " "),
                        "source": "oracle-of-secrets",
                    },
                })

    return items


def generate_veran_benchmarks() -> list[dict]:
    """Generate Veran explanation benchmark items."""
    items = []
    item_id = 0

    difficulty_map = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    for difficulty, examples in VERAN_EXAMPLES.items():
        diff_level = difficulty_map.get(difficulty, 1)

        for example in examples:
            item_id += 1
            code = example.get("code", "")
            concepts = example.get("concepts", [])

            items.append({
                "id": f"veran_{difficulty}_{item_id:03d}",
                "category": "explanation",
                "difficulty": diff_level,
                "code": code.strip(),
                "metadata": {
                    "concepts": concepts,
                },
            })

    # Add instruction explanation items
    instructions = [
        ("LDA #$42", ["immediate addressing", "accumulator", "load"]),
        ("LDA $10", ["zero page addressing", "memory read"]),
        ("LDA $1000", ["absolute addressing", "memory read"]),
        ("LDA $7E0100", ["long addressing", "WRAM"]),
        ("LDA ($10),Y", ["indirect indexed addressing", "pointer"]),
        ("LDA $10,X", ["indexed addressing", "array access"]),
        ("STA $2100", ["store operation", "hardware register"]),
        ("STZ $10", ["store zero", "memory clear"]),
        ("REP #$20", ["mode switch", "16-bit accumulator"]),
        ("SEP #$30", ["mode switch", "8-bit registers"]),
        ("JSR routine", ["subroutine call", "stack", "return address"]),
        ("JSL $7E8000", ["long subroutine call", "bank crossing"]),
        ("RTS", ["return from subroutine", "stack pop"]),
        ("RTL", ["return long", "24-bit return"]),
        ("BEQ label", ["branch on equal", "zero flag"]),
        ("BNE label", ["branch not equal", "conditional"]),
        ("BPL label", ["branch positive", "sign flag"]),
        ("BMI label", ["branch minus", "negative flag"]),
        ("BRA label", ["unconditional branch", "relative jump"]),
        ("BRL label", ["branch long", "16-bit offset"]),
        ("PHA", ["push accumulator", "stack"]),
        ("PLA", ["pull accumulator", "stack"]),
        ("PHX", ["push X register", "stack"]),
        ("PHP", ["push processor flags", "state preservation"]),
        ("CLC", ["clear carry", "flag manipulation"]),
        ("SEC", ["set carry", "flag manipulation"]),
        ("ASL A", ["arithmetic shift left", "multiply by 2"]),
        ("LSR A", ["logical shift right", "divide by 2"]),
        ("ROL A", ["rotate left", "carry flag"]),
        ("ADC #$10", ["add with carry", "arithmetic"]),
        ("SBC #$10", ["subtract with borrow", "arithmetic"]),
        ("CMP #$42", ["compare", "flag setting"]),
        ("AND #$0F", ["bitwise AND", "masking"]),
        ("ORA #$80", ["bitwise OR", "flag setting"]),
        ("EOR #$FF", ["exclusive OR", "bit toggle"]),
        ("INC $10", ["increment memory", "counter"]),
        ("DEC $10", ["decrement memory", "countdown"]),
        ("INX", ["increment X", "loop counter"]),
        ("DEY", ["decrement Y", "loop counter"]),
        ("TAX", ["transfer A to X", "register copy"]),
        ("TXA", ["transfer X to A", "register copy"]),
        ("XBA", ["exchange B and A", "byte swap"]),
        ("WAI", ["wait for interrupt", "power saving"]),
        ("NOP", ["no operation", "timing/padding"]),
    ]

    for code, concepts in instructions:
        item_id += 1
        items.append({
            "id": f"veran_instr_{item_id:03d}",
            "category": "instruction",
            "difficulty": 1,
            "code": code,
            "metadata": {"concepts": concepts},
        })

    # Add code pattern explanations
    patterns = [
        # Loops
        ("LDX #$10\nloop:\nDEX\nBNE loop", ["countdown loop", "index register", "branch"]),
        ("LDY #$00\n.lp:\nLDA (src),Y\nSTA (dst),Y\nINY\nCPY #$20\nBNE .lp", ["memory copy loop", "indirect addressing"]),
        # Mode switching
        ("REP #$20\nLDA $10\nSTA $20\nSEP #$20", ["16-bit transfer", "word copy"]),
        ("SEP #$30\nLDA #$42\nREP #$20", ["mode switching", "register size"]),
        # Stack operations
        ("PHA\nPHX\nJSR sub\nPLX\nPLA", ["register preservation", "subroutine call"]),
        ("PHB\nPHK\nPLB\nJSR sub\nPLB", ["bank preservation", "data bank"]),
        # Arithmetic
        ("CLC\nLDA $10\nADC $12\nSTA $14", ["16-bit addition", "carry propagation"]),
        ("SEC\nLDA $10\nSBC $12\nSTA $14", ["subtraction", "borrow"]),
        # Bit manipulation
        ("LDA $10\nAND #$0F\nORA #$80\nSTA $10", ["masking", "flag setting"]),
        ("LDA $10\nEOR #$FF\nINC A\nSTA $10", ["two's complement", "negation"]),
        # Hardware interaction
        ("LDA $4212\nAND #$80\nBEQ -", ["VBLANK wait", "hardware polling"]),
        ("LDA #$80\nSTA $2115", ["VRAM increment mode", "PPU setup"]),
    ]

    for code, concepts in patterns:
        item_id += 1
        items.append({
            "id": f"veran_pattern_{item_id:03d}",
            "category": "pattern",
            "difficulty": 2,
            "code": code,
            "metadata": {"concepts": concepts},
        })

    # Add ASAR syntax examples for explanation
    asar_examples = [
        ("address_operators", ASAR_SYNTAX.get("address_operators", {})),
        ("labels", ASAR_SYNTAX.get("labels", {})),
        ("data_directives", ASAR_SYNTAX.get("data_directives", {})),
    ]

    for category, examples_dict in asar_examples:
        if isinstance(examples_dict, dict):
            for name, code in examples_dict.items():
                item_id += 1
                items.append({
                    "id": f"veran_asar_{item_id:03d}",
                    "category": f"asar_{category}",
                    "difficulty": 2,
                    "code": code,
                    "metadata": {
                        "concepts": ["ASAR syntax", category, name],
                    },
                })

    # Add SNES hardware register explanations
    register_explanations = [
        ("$2100", ["INIDISP", "screen brightness", "force blank"]),
        ("$2105", ["BGMODE", "screen mode", "tile size"]),
        ("$2107", ["BG1SC", "tilemap address", "size"]),
        ("$210B", ["BG12NBA", "character address", "tiles"]),
        ("$210D", ["BG1HOFS", "scroll register", "horizontal"]),
        ("$2115", ["VMAIN", "VRAM increment", "address mode"]),
        ("$2116", ["VMADDL", "VRAM address low"]),
        ("$2118", ["VMDATAL", "VRAM data low", "write"]),
        ("$2121", ["CGADD", "CGRAM address", "palette"]),
        ("$2122", ["CGDATA", "CGRAM data", "color"]),
        ("$212C", ["TM", "main screen", "layer enable"]),
        ("$212D", ["TS", "subscreen", "layer enable"]),
        ("$2130", ["CGWSEL", "color math", "window"]),
        ("$2131", ["CGADSUB", "color math", "operation"]),
        ("$4200", ["NMITIMEN", "interrupt enable", "NMI"]),
        ("$4210", ["RDNMI", "NMI flag", "read"]),
        ("$4212", ["HVBJOY", "status", "VBLANK"]),
        ("$4218", ["JOY1L", "joypad 1 low", "input"]),
        ("$4300", ["DMAP0", "DMA parameters", "channel 0"]),
        ("$420B", ["MDMAEN", "DMA enable", "trigger"]),
    ]
    for addr, concepts in register_explanations:
        item_id += 1
        items.append({
            "id": f"veran_reg_{item_id:03d}",
            "category": "hardware_register",
            "difficulty": 2,
            "code": addr,
            "metadata": {"concepts": concepts, "type": "register"},
        })

    # Add advanced code pattern explanations
    advanced_patterns = [
        # DMA patterns
        ("LDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\nLDA #$00\nSTA $4302\nLDA #$10\nSTA $4303\nLDA #$7E\nSTA $4304\nLDA #$00\nSTA $4305\nLDA #$08\nSTA $4306\nLDA #$01\nSTA $420B",
         ["DMA transfer", "VRAM write", "channel 0"]),
        # HDMA patterns
        ("LDA #$00\nSTA $4340\nLDA #$21\nSTA $4341\nLDA #<table\nSTA $4342\nLDA #>table\nSTA $4343\nLDA #^table\nSTA $4344\nLDA #$10\nSTA $420C",
         ["HDMA", "gradient", "color register"]),
        # Multiplication
        ("REP #$20\nLDA $10\nSTA $4202\nSTA $4203\nNOP\nNOP\nNOP\nLDA $4216\nSEP #$20",
         ["hardware multiply", "8x8", "result"]),
        # Division
        ("REP #$20\nLDA dividend\nSTA $4204\nSEP #$20\nLDA divisor\nSTA $4206\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nLDA $4214",
         ["hardware divide", "quotient", "remainder"]),
        # Sprite draw
        ("LDA $00\nSTA $0800,Y\nLDA $01\nSTA $0801,Y\nLDA $02\nSTA $0802,Y\nLDA $03\nSTA $0803,Y",
         ["OAM write", "sprite position", "tile/attr"]),
        # Bank switch
        ("PHB\nLDA #$7E\nPHA\nPLB\nLDA $0000\nPLB",
         ["bank switch", "data bank", "WRAM access"]),
        # Direct page
        ("PHD\nREP #$20\nLDA #$0000\nTCD\nSEP #$20\nLDA $10\nPLD",
         ["direct page", "zero page", "fast access"]),
    ]
    for code, concepts in advanced_patterns:
        item_id += 1
        items.append({
            "id": f"veran_advpat_{item_id:03d}",
            "category": "advanced_pattern",
            "difficulty": 3,
            "code": code,
            "metadata": {"concepts": concepts},
        })

    # Add ALTTP-specific code explanations
    alttp_patterns = [
        ("LDA $7E0010\nAND #$40", ["game module", "submodule check"]),
        ("LDA $7E0022\nSTA $7E0360", ["Link X position", "sprite coordinate"]),
        ("LDA $0D00,X\nBEQ .skip", ["sprite status", "active check"]),
        ("LDA $0E20,X\nASL #4\nTAY", ["sprite type", "lookup index"]),
        ("JSL $01B2D4", ["Sprite_SetupBasicHitbox", "ALTTP routine"]),
        ("JSL $01C39E", ["Sprite_CheckDamageFromPlayer", "collision"]),
        ("LDA $0D90,X\nBEQ .no_timer", ["sprite timer", "countdown"]),
        ("LDA $0F10,X\nSTA $0D80,X", ["sprite subtype", "initialization"]),
        ("LDA $0E70,X\nAND #$04", ["sprite bump damage", "knockback"]),
        ("LDA $0F40,X\nSTA $0D60,X", ["sprite AI phase", "state machine"]),
        ("LDA $0FB0,X\nCMP #$10", ["sprite speed", "velocity check"]),
        ("LDA $0E00,X\nORA #$40", ["sprite flags", "invincibility"]),
        ("LDA $0DA0,X\nBEQ .ground", ["sprite Z height", "airborne check"]),
        ("JSL $06FA9D", ["Sprite_SpawnDynamically", "spawn routine"]),
        ("JSL $01B7B3", ["Sprite_DrawShadow", "shadow rendering"]),
        ("LDA $0DC0,X\nASL A\nTAY", ["sprite graphics", "tile offset"]),
        ("JSL $00E0A3", ["Sprite_DamageFlash", "hit effect"]),
        ("LDA $0F20,X\nSTA $0D10,X", ["sprite direction", "facing copy"]),
    ]
    for code, concepts in alttp_patterns:
        item_id += 1
        items.append({
            "id": f"veran_alttp_{item_id:03d}",
            "category": "alttp_pattern",
            "difficulty": 3,
            "code": code,
            "metadata": {"concepts": concepts, "game": "alttp"},
        })

    # Add complete code examples
    complete_examples = [
        ("wait_loops", ASAR_SYNTAX.get("wait_loops", "")),
        ("joypad_read", ASAR_SYNTAX.get("joypad_read", "")),
        ("dma_setup_correct", ASAR_SYNTAX.get("dma_setup_correct", "")),
    ]

    for name, code in complete_examples:
        if code:
            item_id += 1
            items.append({
                "id": f"veran_complete_{item_id:03d}",
                "category": "complete_routine",
                "difficulty": 3,
                "code": code.strip(),
                "metadata": {
                    "concepts": [name.replace("_", " "), "complete routine", "SNES hardware"],
                },
            })

    # Add Oracle Veran patterns (documentation)
    if "veran" in ORACLE_PATTERNS:
        for name, doc in ORACLE_PATTERNS["veran"].items():
            if isinstance(doc, str):
                item_id += 1
                items.append({
                    "id": f"veran_oracle_{item_id:03d}",
                    "category": "oracle_docs",
                    "difficulty": 4,
                    "code": doc.strip()[:500],  # Truncate long docs
                    "metadata": {
                        "concepts": ["sprite system", "memory map", "game mechanics"],
                        "source": "oracle-of-secrets",
                    },
                })

    return items


def save_benchmarks(items: list[dict], output_path: Path) -> int:
    """Save benchmark items to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    return len(items)


def main():
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks"

    print("Generating benchmark datasets from template_libraries...")
    print()

    # Generate each domain
    domains = [
        ("din", generate_din_benchmarks),
        ("farore", generate_farore_benchmarks),
        ("nayru", generate_nayru_benchmarks),
        ("veran", generate_veran_benchmarks),
    ]

    total = 0
    for domain, generator in domains:
        items = generator()
        output_path = benchmarks_dir / domain / "benchmark.jsonl"
        count = save_benchmarks(items, output_path)
        total += count
        print(f"  {domain}: {count} items -> {output_path}")

    print()
    print(f"Total: {total} benchmark items generated")

    # Update metadata
    metadata = {
        "version": "2.0",
        "generated": "2026-01-04",
        "domains": {
            domain: {
                "file": f"{domain}/benchmark.jsonl",
                "count": len(generator())
            }
            for domain, generator in domains
        },
        "total_items": total,
    }

    metadata_path = benchmarks_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
