#!/usr/bin/env python3
"""
Generate SNES hardware-focused training samples using veran-7b-v4 via LMStudio API.

Focuses on:
- PPU registers and operations ($2100-$213F)
- DMA channels and configuration ($4300-$430A)
- HDMA for effects
- Mode 7 graphics
- OAM sprite system
- VRAM timing and access
- Palette operations

Output: ~/.context/training/lmstudio_synthetic/veran_hardware_samples.jsonl
"""

import json
import requests
import time
from typing import Optional
from pathlib import Path
import sys

# Configuration
LMSTUDIO_API = "http://localhost:1234/v1"
MODEL_ID = "gguf/ollama/veran-7b-v4-q8.gguf"
OUTPUT_DIR = Path.home() / ".context" / "training" / "lmstudio_synthetic"
OUTPUT_FILE = OUTPUT_DIR / "veran_hardware_samples.jsonl"
NUM_SAMPLES = 250

# SNES hardware focus areas with specific register ranges and topics
HARDWARE_FOCUS_AREAS = [
    {
        "name": "PPU Registers",
        "ranges": ["$2100-$213F"],
        "topics": [
            "screen brightness and effects ($2100)",
            "object size and base address ($2101)",
            "character location ($2102-$2103)",
            "VRAM address increment ($2115)",
            "VRAM read/write ($2118-$2119)",
            "background layer configuration ($211A-$211D)",
            "background scroll position ($210D-$2110)",
            "window mask settings ($2123-$212E)",
            "color math operations ($2130-$2132)",
            "mode selection and screen height ($2133)",
        ]
    },
    {
        "name": "DMA/HDMA",
        "ranges": ["$4300-$430A"],
        "topics": [
            "DMA control ($4300) - transfer mode and direction",
            "DMA destination register ($4301)",
            "DMA source address ($4302-$4304)",
            "DMA transfer size ($4305-$4306)",
            "HDMA table indirect addressing",
            "HDMA line counter and repeat patterns",
            "HDMA gradient effects",
            "dual-channel DMA transfers",
            "DMA timing and cycle-stealing",
            "DMA interrupt handling",
        ]
    },
    {
        "name": "Mode 7 Graphics",
        "ranges": ["$211B-$2134"],
        "topics": [
            "mode 7 matrix values ($211B-$2134)",
            "center point calculation",
            "rotation and scaling mathematics",
            "perspective effects in Mode 7",
            "Mode 7 tilemap addressing",
            "Mode 7 character storage",
            "rotation angle calculations",
            "Mode 7 on top of other layers",
            "Mode 7 timing and rendering",
            "Mode 7 effects and transformations",
        ]
    },
    {
        "name": "OAM Sprite System",
        "ranges": ["$2101-$2104"],
        "topics": [
            "object size selection ($2101)",
            "OAM address setup ($2101-$2104)",
            "sprite data format in OAM",
            "high table for priority bits",
            "sprite coordinate updates",
            "sprite palette selection",
            "sprite priority ordering",
            "OAM DMA transfers",
            "OAM refresh timing",
            "rotating sprite patterns",
        ]
    },
    {
        "name": "VRAM Operations",
        "ranges": ["$2115-$2119"],
        "topics": [
            "VRAM address increment mode ($2115)",
            "VRAM address setup ($2116-$2117)",
            "VRAM data write ($2119)",
            "VRAM data read ($2139-$213A)",
            "VRAM refresh timing",
            "character data storage",
            "tilemap data organization",
            "VRAM access patterns",
            "VRAM DMA transfers",
            "double-buffering in VRAM",
        ]
    },
    {
        "name": "Palette Operations",
        "ranges": ["$2121-$2128"],
        "topics": [
            "CGRAM address setup ($2121)",
            "color data write ($2122)",
            "color data read ($2139)",
            "palette entry format (BGR555)",
            "layer palette bank selection",
            "sprite palette organization",
            "color math operations",
            "fade effects via CGRAM",
            "CGRAM DMA transfers",
            "dynamic palette updates",
        ]
    },
    {
        "name": "Interrupt & Timing",
        "ranges": ["$4200-$4212"],
        "topics": [
            "interrupt enable register ($4200)",
            "HV counter reads ($213C-$213D)",
            "scanline timing ($4207-$4208)",
            "H-counter timing ($4209-$420A)",
            "NMI and IRQ handling",
            "VBLANK and HBLANK periods",
            "CPU/PPU synchronization",
            "cycle-exact timing requirements",
            "interrupt latency",
            "timing-sensitive register access",
        ]
    },
    {
        "name": "Color Math & Effects",
        "ranges": ["$2130-$2132"],
        "topics": [
            "color math enable ($2130)",
            "color math operation ($2130)",
            "arithmetic select ($2131)",
            "subtract mask settings",
            "half-intensity math",
            "screen addition/subtraction",
            "back color math",
            "window effects with color math",
            "fade in/out implementations",
            "special effects programming",
        ]
    },
    {
        "name": "Window & Layer Control",
        "ranges": ["$2123-$212E"],
        "topics": [
            "BG1 window mask settings ($2123)",
            "BG2 window mask settings ($2124)",
            "BG3 window mask settings ($2125)",
            "BG4 and OBJ window mask ($2126)",
            "window position registers ($2126-$2129)",
            "window logic operations ($212E)",
            "window clipping effects",
            "layer masking",
            "window animation",
            "complex window patterns",
        ]
    },
    {
        "name": "APU Communication",
        "ranges": ["$2140-$2143"],
        "topics": [
            "APU port 0-3 reads/writes",
            "sound effect triggers",
            "music queue management",
            "echo and reverb parameters",
            "sample rate control",
            "APU synchronization",
            "SPC700 communication protocol",
            "APU reset procedures",
            "APU upload sequences",
            "audio timing and buffering",
        ]
    },
]

# Sample templates for different hardware concepts
SAMPLE_TEMPLATES = [
    {
        "type": "register_explanation",
        "template": "Explain what register ${register} does and provide 65816 assembly code to {action}"
    },
    {
        "type": "dma_scenario",
        "template": "Write 65816 assembly to {action} using DMA channel {channel}. Include register setup at {range}"
    },
    {
        "type": "timing_question",
        "template": "When {timing_condition}, what is the timing behavior of {register}? Include cycle counts."
    },
    {
        "type": "effect_implementation",
        "template": "Implement a {effect} effect using {hardware_feature}. Show how to update {register} to achieve this."
    },
    {
        "type": "bug_analysis",
        "template": "What bug can occur if you read from {register} {timing_issue}? How do you work around it?"
    },
    {
        "type": "optimization",
        "template": "Optimize this VRAM/OAM/CGRAM access pattern: {pattern}. Show assembly and explain the performance gain."
    },
    {
        "type": "protocol",
        "template": "Describe the correct protocol for {operation}. Include all necessary register writes and timing constraints."
    },
]

# Action templates for varied prompts
ACTIONS = [
    "set screen brightness",
    "configure DMA transfer",
    "update sprite position",
    "apply a color fade",
    "setup Mode 7 rotation",
    "configure HDMA line effect",
    "read PPU status safely",
    "queue audio samples",
    "apply window masking",
    "create palette animation",
    "implement screen split effect",
    "setup scanline interrupt",
    "transfer tilemap data",
    "configure sprite priorities",
    "setup gradient effect",
]

TIMING_CONDITIONS = [
    "during VBLANK",
    "during HBLANK",
    "during active display",
    "at the scanline boundary",
    "during DMA operation",
    "immediately after RESET",
    "in rapid succession",
    "at $2119 with auto-increment",
    "between frame boundaries",
    "during HDMA transfer",
]

EFFECTS = [
    "screen fade in/out",
    "palette rotation",
    "Mode 7 rotation",
    "scanline displacement",
    "color addition/subtraction",
    "sprite flicker",
    "parallax scrolling",
    "mosaic",
    "interlacing",
    "screen distortion",
]

HARDWARE_FEATURES = [
    "PPU color math",
    "HDMA",
    "DMA",
    "sprite doubling",
    "Mode 7",
    "window masking",
    "layer priority",
    "VRAM paging",
]

DMA_CHANNELS = ["0", "1", "2", "3", "4", "5", "6", "7"]
REGISTERS = [
    "$2100", "$2101", "$2102", "$2103", "$2104", "$2115", "$2116", "$2117",
    "$2118", "$2119", "$2121", "$2122", "$2123", "$2124", "$2125", "$2126",
    "$2130", "$2131", "$2132", "$2133", "$4300", "$4301", "$4302", "$4305",
    "$4307", "$213C", "$213D", "$4200", "$4207", "$2140",
]

PATTERNS = [
    "sequential writes to VRAM",
    "burst DMA to CGRAM",
    "OAM transfer with address wrapping",
    "interleaved reads and writes",
    "conditional HDMA setup",
]

OPERATIONS = [
    "clear VRAM safely",
    "transfer tilemap data",
    "update palette during VBLANK",
    "read OAM status",
    "configure Mode 7 matrix",
    "queue audio command",
]

TIMING_ISSUES = [
    "during or immediately after VBLANK",
    "in tight loops",
    "with auto-increment enabled",
    "at scanline boundaries",
    "during active display",
]


class SNESHardwarePromptGenerator:
    """Generate varied SNES hardware-focused prompts."""

    def __init__(self):
        self.prompt_count = 0

    def generate_prompt(self, focus_area: dict, template: dict) -> str:
        """Generate a single prompt based on template and focus area."""
        template_text = template["template"]

        # Build substitution dictionary
        substitutions = {
            "action": self._pick_random(ACTIONS),
            "register": self._pick_random(REGISTERS),
            "channel": self._pick_random(DMA_CHANNELS),
            "range": self._pick_random(focus_area["ranges"]),
            "timing_condition": self._pick_random(TIMING_CONDITIONS),
            "effect": self._pick_random(EFFECTS),
            "hardware_feature": self._pick_random(HARDWARE_FEATURES),
            "pattern": self._pick_random(PATTERNS),
            "timing_issue": self._pick_random(TIMING_ISSUES),
            "operation": self._pick_random(OPERATIONS),
        }

        prompt = template_text
        for key, value in substitutions.items():
            prompt = prompt.replace(f"{{{key}}}", value)

        self.prompt_count += 1
        return prompt

    @staticmethod
    def _pick_random(items):
        """Pick a random item from list."""
        import random
        return random.choice(items)


def create_training_sample(
    prompt: str,
    completion: str,
    focus_area: str,
    template_type: str,
    sample_id: int,
) -> dict:
    """Create a TrainingSample JSON object."""
    return {
        "id": f"veran_hardware_{sample_id:04d}",
        "type": "snes_hardware_training",
        "focus_area": focus_area,
        "template_type": template_type,
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "model": "veran-7b-v4-q8",
            "api": "lmstudio",
            "hardware_focused": True,
            "timestamp": time.time(),
        }
    }


def call_lmstudio_completion(prompt: str, max_tokens: int = 400) -> Optional[str]:
    """Call LMStudio completions API."""
    try:
        response = requests.post(
            f"{LMSTUDIO_API}/completions",
            json={
                "model": MODEL_ID,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("choices") and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
    except requests.RequestException as e:
        print(f"API Error: {e}", file=sys.stderr)
    return None


def generate_samples(num_samples: int = NUM_SAMPLES):
    """Generate SNES hardware training samples."""
    print(f"Generating {num_samples} SNES hardware training samples...")
    print(f"Model: {MODEL_ID}")
    print(f"API: {LMSTUDIO_API}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    generator = SNESHardwarePromptGenerator()
    samples_written = 0
    samples_failed = 0
    start_time = time.time()

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for sample_num in range(1, num_samples + 1):
            try:
                # Select random focus area and template
                import random
                focus_area = random.choice(HARDWARE_FOCUS_AREAS)
                template = random.choice(SAMPLE_TEMPLATES)

                # Generate prompt
                prompt = generator.generate_prompt(focus_area, template)

                # Call API
                print(f"[{sample_num}/{num_samples}] Generating: {template['type']}...", end=" ")
                completion = call_lmstudio_completion(prompt)

                if completion:
                    sample = create_training_sample(
                        prompt=prompt,
                        completion=completion,
                        focus_area=focus_area["name"],
                        template_type=template["type"],
                        sample_id=sample_num,
                    )
                    f.write(json.dumps(sample) + "\n")
                    samples_written += 1
                    print("✓")
                else:
                    samples_failed += 1
                    print("✗ (no response)")

                # Rate limiting
                time.sleep(0.5)

                # Progress report every 25 samples
                if sample_num % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = sample_num / elapsed
                    remaining = (num_samples - sample_num) / rate if rate > 0 else 0
                    print(
                        f"\n  Progress: {samples_written}/{sample_num} successful "
                        f"({100*samples_written//sample_num}%) | "
                        f"ETA: {remaining:.0f}s\n"
                    )

            except Exception as e:
                samples_failed += 1
                print(f"✗ (error: {e})")

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Samples written: {samples_written}")
    print(f"Samples failed:  {samples_failed}")
    print(f"Total time:      {elapsed:.1f}s")
    print(f"Output file:     {OUTPUT_FILE}")
    print(f"File size:       {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print("=" * 70)

    return samples_written, samples_failed


if __name__ == "__main__":
    try:
        written, failed = generate_samples()
        sys.exit(0 if written > 0 else 1)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
