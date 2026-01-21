#!/usr/bin/env python3
"""
Generate 250 debugging-focused training samples using farore-7b-v5 model.
Focus: 65816 assembly debugging, error diagnosis, and register/memory analysis.
"""

import json
import requests
import time
from datetime import datetime
from typing import Generator
from pathlib import Path

# LMStudio API configuration
API_BASE = "http://localhost:1234/v1"
MODEL_ID = "gguf/ollama/farore-7b-v5-q8.gguf"
OUTPUT_PATH = Path.home() / ".context/training/lmstudio_synthetic/farore_debugging_samples.jsonl"

# Debugging focus prompts - organized by category
DEBUGGING_PROMPTS = {
    "mode_mismatches": [
        "A 65816 subroutine is called in native mode but uses 16-bit addressing assuming emulation mode. The JSL instruction jumps to $030000 but accesses registers as if in 8-bit mode. What's wrong? Diagnose the bug.",
        "Code switches to emulation mode with SED but later assumes 16-bit A register for a calculation. RTI restores native mode. How does this cause corruption?",
        "A function uses SEP #$20 (set 8-bit accumulator) but then tries to do 16-bit math with LDA #$1234. What registers get clobbered?",
        "Mode byte $4D contains $C0 (native mode, 16-bit A/X/Y) but the code treats it as emulation mode. What memory accesses fail?",
        "After CLI to enable interrupts, an interrupt fires in the middle of a 16-bit operation. The ISR doesn't preserve bit 5 of P register. Why is this dangerous?",
    ],

    "stack_corruption": [
        "A JSR pushes return address to stack, but the stack pointer X register is loaded with $01FF without initializing. Subsequent PHA overwrites memory. Where?",
        "Nested subroutines PHA/PLA values but one level forgets to PLA. The RTL pops wrong value. How do you trace this?",
        "Stack grows from $01FF downward. Code uses TSC to read stack pointer but gets $01FD. Next three pushes hit $01FC, $01FB, $01FA. Is this safe?",
        "A routine does 3x PHA then 4x PLA. Stack underflow reads from $0200+. What's in memory there? How to detect?",
        "Code does PEI ($10) to push from direct page but $10 is the loop counter being decremented. Stack data gets corrupted each loop. Debug approach?",
    ],

    "jsr_rtl_bugs": [
        "JSL $030000 jumps into bank $03, but the code has no RTL - only RTS. The return address gets corrupted. Why does the CPU crash?",
        "A function JSR's to $8000 in the ROM, modifies A, but a nested JSR forgot to RTL. The original function returns to wrong address. Stack diagram?",
        "JML (long jump) to $028000 works, but conditional branch skips the stack setup. Later JSR assumes valid return address. What breakpoint catches this?",
        "RTL from a long subroutine pops 3 bytes of address but the return address was only 2 bytes (RTS). Why the crash at $xxxxFF?",
        "A JSL chain: JSL A -> JSL B -> RTS (missing RTL in B). The second RTL reads wrong return bank. How to trace with CPU trace log?",
    ],

    "register_analysis": [
        "After CLC, A=$1234, X=$5678, Y=$9ABC. An ADC $0000 adds without overflow. What are A/P/N/V flags after? Show register state.",
        "REP #$30 sets 16-bit A and X/Y. Next, LDA #$80, followed by ASL A (shift left). What's the carry flag? N flag?",
        "X=$FF00, Y=$0001. SBC #$0001 subtracts 1 from A (assume A=$0000). Carry flag? Zero flag? Why is the result important?",
        "STX $0000, STY $0002 stores two 16-bit values. Then LDA $0001. Which byte of X is loaded? Endianness issue?",
        "After SEP #$30 (8-bit A/X/Y), LDA #$80, ASL A (8-bit shift). Carry flag set? How does this differ from 16-bit ASL?",
    ],

    "memory_corruption": [
        "Code writes to $7EFC without bounds checking. Nearby is the stack at $7F00+. Later, stack corruption detected. How to pinpoint the write?",
        "DMA controller at $4300-$43FF is used without page boundary checks. Writing via $4304/$4305 jumps pages. What addresses are clobbered?",
        "LDA ($10) indirect indexed, X=$FF. Address $10/$11 = $7F00. Load becomes $7F00 + $FF = $7FFF. Next indirect read wraps? Wrap behavior?",
        "Code uses BIT $2000 to check VRAM port, but $2000 is actually program ROM. Reading it corrupts timing. How does this manifest?",
        "A routine increments $0000 as a counter, but $0000 is the native flag in the CPU state. Incrementing it changes mode. Debug method?",
    ],

    "timing_issues": [
        "Code reads PPU $2002 (status) to sync with vertical blank, but doesn't wait for the right scanline. Frame rate drops. Why?",
        "HDMA channel runs during active display, transferring from ROM to VRAM. Timing misalignment causes graphical glitches. Diagnosis approach?",
        "A busy-wait loop polls PPU status, but the loop is too fast for the actual PPU cycle timing. What's the consequence? Measurement?",
        "NMI (non-maskable interrupt, V-blank) fires after rendering. Code assumes it can modify sprites, but the read happened mid-transfer. Race condition?",
        "Code waits for V-blank using STA $4200 (enable NMI), but the interrupt handler wasn't set up first. What happens? Trace logic?",
    ],

    "dma_problems": [
        "HDMA mode 0, direct transfer from $10xxxx to $2118 (VRAM port), count=$0100. Mid-transfer, VRAM wraps around. Consequences?",
        "Code sets DMA source $xx4302/$4303 to $7E0000 (WRAM), destination $2119 (VRAM data). DMA controller reads WRAM correctly, but output address wraps. Why?",
        "Chained DMA: Channel 0 transfers 256 bytes, then Channel 1 should auto-start. But channel 1's source wasn't initialized. Garbage transferred?",
        "DMA to PPU port $2122 (CGRAM address/data) without bank switching. VRAM data ends up in wrong CGRAM address. Debugging approach?",
        "HDMA timing: Transfer happens during rendering. Palette changes mid-scanline cause color glitches. How to detect HDMA timing conflicts?",
    ],

    "error_diagnosis": [
        "Game crashes after a few minutes of play. Stack dump shows PC=$8F0010, return address=$030100. Is this a stack corruption or bad branch?",
        "Emulator breakpoint never hits expected subroutine. CPU trace shows JSL to $028000, but PC never reaches it. Branch condition issue?",
        "Register dump at crash: A=$FFFF, X=$0000, Y=$7EFF, SP=$01FA, P=$73. Is the carry flag set correctly for the next operation?",
        "Sprite graphics are corrupted. OAM (object attribute memory) was read via DMA as $2104, but should be written to $2104. Direction reversed?",
        "Sound playback stutters. Sample counter incremented too often in ISR. CPU trace shows CPU time = 99% usage. Where's the bottleneck?",
    ],

    "instruction_misuse": [
        "Code uses SBC (subtract with carry) without understanding the carry flag. Carry=0 before subtraction means subtract 1 extra. Why the off-by-one?",
        "BIT instruction used to test flags but also affects A register on some CPUs. Developer assumes no A modification. Real consequence?",
        "ROR (rotate right) vs LSR (logical shift right) confusion. ROR preserves carry into bit 7, LSR loads 0. How does this break rotation logic?",
        "BRL (branch long) uses 16-bit signed offset but limited to ±32KB. Branch to $050000 from $008000 fails silently. Why and how to detect?",
        "COP (coprocessor) instruction triggers interrupt, but ISR resets D register. Later indirect addressing in D uses old value. State preservation?",
    ],
}

class DebuggingTrainingSampleGenerator:
    """Generate debugging-focused training samples from farore model."""

    def __init__(self, api_base: str, model_id: str):
        self.api_base = api_base
        self.model_id = model_id
        self.session = requests.Session()
        self.sample_count = 0

    def generate_completion(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LMStudio completions API."""
        url = f"{self.api_base}/completions"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False,
        }

        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"].strip()
        except requests.RequestException as e:
            print(f"API Error: {e}")
            return ""

    def create_training_sample(
        self,
        category: str,
        problem_prompt: str,
        problem_text: str
    ) -> dict:
        """Create a training sample with diagnosis-first format."""

        # Generate diagnosis response
        diagnosis_prompt = f"""You are a 65816 assembly debugging expert. Analyze this debugging problem and provide:
1. Root cause diagnosis
2. Evidence/symptoms
3. Debug steps
4. Solution

Problem: {problem_text}

Provide a structured diagnosis:"""

        diagnosis = self.generate_completion(diagnosis_prompt, max_tokens=400)

        if not diagnosis:
            return None

        self.sample_count += 1

        return {
            "id": f"farore-debug-{self.sample_count:03d}",
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "problem": problem_text,
            "diagnosis": diagnosis,
            "model": "farore-7b-v5-q8",
            "focus_areas": ["65816", "debugging", "error_diagnosis", "register_analysis"],
            "metadata": {
                "sample_number": self.sample_count,
                "prompt_type": "diagnosis_first",
                "total_planned": 250,
            }
        }

    def generate_all_samples(self) -> Generator[dict, None, None]:
        """Generate all 250 training samples."""

        total_prompts = sum(len(v) for v in DEBUGGING_PROMPTS.values())
        samples_per_prompt = max(1, 250 // total_prompts)

        print(f"Total debugging prompts: {total_prompts}")
        print(f"Samples per prompt: {samples_per_prompt}")
        print(f"Expected total: ~{total_prompts * samples_per_prompt}")
        print()

        for category, prompts in DEBUGGING_PROMPTS.items():
            print(f"Generating samples for category: {category}")
            print(f"  Prompts in category: {len(prompts)}")

            for prompt_idx, prompt in enumerate(prompts):
                for variation in range(samples_per_prompt):
                    # Add variation to prompt for diversity
                    if variation == 0:
                        prompt_text = prompt
                    else:
                        prompt_text = f"{prompt} (Alternative: Consider the edge case scenario.)"

                    sample = self.create_training_sample(
                        category=category,
                        problem_prompt=prompt,
                        problem_text=prompt_text
                    )

                    if sample:
                        yield sample
                        print(f"  [{self.sample_count}/250] {category} - Generated")

                        # Rate limiting to avoid overwhelming the API
                        if self.sample_count % 10 == 0:
                            time.sleep(1)
                    else:
                        print(f"  [{self.sample_count}/250] {category} - Failed")


def save_samples_to_jsonl(samples: Generator[dict, None, None], output_path: Path):
    """Save samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\nSaved to: {output_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("65816 Debugging Training Sample Generator")
    print("=" * 70)
    print(f"API: {API_BASE}")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    generator = DebuggingTrainingSampleGenerator(API_BASE, MODEL_ID)

    # Generate all samples
    samples = generator.generate_all_samples()

    # Save to JSONL
    save_samples_to_jsonl(samples, OUTPUT_PATH)

    # Print summary statistics
    print("\n" + "=" * 70)
    print(f"Generation Complete!")
    print(f"Total samples generated: {generator.sample_count}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
