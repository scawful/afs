"""Mesen2 save state parser for Oracle of Secrets training data.

Parses Mesen2 save states (.mss files) to extract game variables and generate
training samples based on actual game states. This allows Majora to understand
real gameplay scenarios and variable states.

Mesen2 save state format:
- Compressed ZIP file containing:
  - SaveState.mss (main state)
  - Video RAM
  - Work RAM (WRAM) - Contains game variables
  - Save RAM (SRAM) - Contains persistent save data
  - State metadata

Oracle of Secrets Memory Layout:
- WRAM $7E0730+ : Custom game variables
- SRAM $7EF38A+ : Persistent save data (masks, items, etc.)

Usage:
    python3 -m afs.oracle.savestate_parser \
        --save-state ~/path/to/oracle.mss \
        --output ~/.context/training/oracle/savestate_samples.jsonl
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..generators.base import TrainingSample


@dataclass
class MemoryRegion:
    """Parsed memory region from save state."""
    name: str  # e.g., "WRAM", "SRAM"
    base_address: int  # Base address in SNES memory map
    data: bytes  # Raw memory data


@dataclass
class GameVariable:
    """Parsed game variable from memory."""
    name: str  # Variable name from memory map
    address: int  # SNES address (e.g., $7E0730)
    value: int  # Current value
    size: int  # Size in bytes (1, 2, 4)
    description: str  # What this variable represents


class Mesen2SaveState:
    """Parser for Mesen2 save state files.

    Mesen2 save states are ZIP archives containing:
    - SaveState.mss: Main state file with registers and memory
    - Additional binary data for VRAM, etc.

    The .mss file structure (approximate):
    - Header (version, emulator info)
    - CPU registers (A, X, Y, SP, PC, etc.)
    - Memory dumps (WRAM, SRAM, etc.)
    """

    def __init__(self, save_state_path: Path):
        """Initialize parser with save state file path."""
        self.save_state_path = Path(save_state_path).expanduser().resolve()

        if not self.save_state_path.exists():
            raise FileNotFoundError(f"Save state not found: {self.save_state_path}")

    def parse(self) -> dict[str, MemoryRegion]:
        """Parse save state and extract memory regions.

        Returns:
            Dictionary of memory regions by name (WRAM, SRAM, etc.)
        """
        regions = {}

        # Mesen2 save states are ZIP files
        with zipfile.ZipFile(self.save_state_path, 'r') as zf:
            # List files in archive
            files = zf.namelist()

            # Extract main save state
            if 'SaveState.mss' in files:
                mss_data = zf.read('SaveState.mss')
                regions.update(self._parse_mss(mss_data))

            # Extract additional memory dumps if present
            for filename in files:
                if 'WRAM' in filename or 'WorkRam' in filename:
                    wram_data = zf.read(filename)
                    regions['WRAM'] = MemoryRegion(
                        name='WRAM',
                        base_address=0x7E0000,
                        data=wram_data
                    )

                if 'SRAM' in filename or 'SaveRam' in filename:
                    sram_data = zf.read(filename)
                    regions['SRAM'] = MemoryRegion(
                        name='SRAM',
                        base_address=0x700000,
                        data=sram_data
                    )

        return regions

    def _parse_mss(self, mss_data: bytes) -> dict[str, MemoryRegion]:
        """Parse main SaveState.mss file.

        The MSS format varies by Mesen2 version. This is a best-effort parser
        that looks for known memory patterns.
        """
        regions = {}

        # TODO: Implement proper MSS parsing based on Mesen2 format
        # For now, return empty dict and rely on separate memory dumps
        # The actual parsing would require reverse engineering the binary format

        return regions

    def extract_variables(
        self,
        memory_map: dict[str, tuple[int, int, str]]
    ) -> list[GameVariable]:
        """Extract game variables from memory using memory map.

        Args:
            memory_map: Dictionary of {name: (address, size, description)}

        Returns:
            List of parsed game variables with current values
        """
        regions = self.parse()
        variables = []

        for var_name, (address, size, description) in memory_map.items():
            # Determine which region this address is in
            if 0x7E0000 <= address <= 0x7FFFFF:
                # WRAM
                if 'WRAM' not in regions:
                    continue
                offset = address - 0x7E0000
                region_data = regions['WRAM'].data
            elif 0x700000 <= address <= 0x7FFFFF:
                # SRAM
                if 'SRAM' not in regions:
                    continue
                offset = address - 0x700000
                region_data = regions['SRAM'].data
            else:
                continue

            # Extract value based on size
            if offset + size > len(region_data):
                continue

            if size == 1:
                value = region_data[offset]
            elif size == 2:
                value = struct.unpack('<H', region_data[offset:offset+2])[0]
            elif size == 4:
                value = struct.unpack('<I', region_data[offset:offset+4])[0]
            else:
                continue

            variables.append(GameVariable(
                name=var_name,
                address=address,
                value=value,
                size=size,
                description=description
            ))

        return variables


class SaveStateTrainingGenerator:
    """Generate training data from save state analysis.

    Patterns:
    - Variable state queries: "What is the current value of [var]?" → value from save
    - Game state analysis: "What state is the game in?" → analysis of multiple variables
    - Progression checks: "Has the player completed [quest]?" → check quest flags
    """

    def __init__(
        self,
        save_state_path: Path,
        memory_map: dict[str, tuple[int, int, str]]
    ):
        """Initialize generator with save state and memory map."""
        self.save_state_path = save_state_path
        self.memory_map = memory_map
        self.parser = Mesen2SaveState(save_state_path)

    def generate(self) -> Iterator[TrainingSample]:
        """Generate training samples from save state."""
        # Extract all variables from save state
        variables = self.parser.extract_variables(self.memory_map)

        if not variables:
            print(f"Warning: No variables extracted from {self.save_state_path}")
            return

        # Generate samples for individual variables
        for var in variables:
            # Variable value query
            instruction = f"In Oracle of Secrets, what is the current value of {var.name}?"
            output = f"The current value of {var.name} (at address ${var.address:06X}) is {var.value:#x} ({var.value}). "
            output += var.description

            thinking = f"Reading from save state at address ${var.address:06X}. "
            thinking += f"This is a {var.size}-byte variable in {'WRAM' if var.address >= 0x7E0000 else 'SRAM'}."

            yield TrainingSample(
                instruction=instruction,
                output=output,
                thinking=thinking,
                domain="oracle_savestate",
                source=str(self.save_state_path.name),
                _metadata={
                    "address": f"${var.address:06X}",
                    "value": var.value,
                    "variable_name": var.name
                }
            )

        # Generate game state analysis sample
        state_summary = self._analyze_game_state(variables)
        instruction = "Analyze the current game state in Oracle of Secrets."
        output = state_summary
        thinking = f"Analyzing {len(variables)} game variables from save state."

        yield TrainingSample(
            instruction=instruction,
            output=output,
            thinking=thinking,
            domain="oracle_savestate",
            source=str(self.save_state_path.name),
            _metadata={
                "num_variables": len(variables),
                "analysis_type": "full_state"
            }
        )

    def _analyze_game_state(self, variables: list[GameVariable]) -> str:
        """Analyze game state from variables and generate summary."""
        analysis = "Current Oracle of Secrets Game State:\n\n"

        # Group variables by category
        categories = {
            "Player": [],
            "Quest": [],
            "Items": [],
            "Masks": [],
            "System": []
        }

        for var in variables:
            # Categorize based on variable name
            name_lower = var.name.lower()
            if any(kw in name_lower for kw in ['link', 'player', 'health', 'magic']):
                categories["Player"].append(var)
            elif any(kw in name_lower for kw in ['quest', 'flag', 'event']):
                categories["Quest"].append(var)
            elif any(kw in name_lower for kw in ['item', 'inventory']):
                categories["Items"].append(var)
            elif 'mask' in name_lower:
                categories["Masks"].append(var)
            else:
                categories["System"].append(var)

        # Build analysis by category
        for category, vars_in_cat in categories.items():
            if not vars_in_cat:
                continue

            analysis += f"{category} State:\n"
            for var in vars_in_cat[:5]:  # Limit to 5 per category
                analysis += f"  - {var.name}: {var.value} ({var.description})\n"

            if len(vars_in_cat) > 5:
                analysis += f"  ... and {len(vars_in_cat) - 5} more {category.lower()} variables\n"

            analysis += "\n"

        return analysis


# Oracle of Secrets Memory Map (from MemoryMap.md)
# This is a subset - full map should be loaded from documentation
ORACLE_MEMORY_MAP = {
    # WRAM variables (from Docs/Core/MemoryMap.md)
    "LinkFaceDir": (0x7E002F, 1, "Link's facing direction (0-3)"),
    "LinkX": (0x7E0022, 2, "Link's X position"),
    "LinkY": (0x7E0020, 2, "Link's Y position"),
    "LinkState": (0x7E005D, 1, "Link's current state"),
    "RoomIndex": (0x7E00A0, 2, "Current room/dungeon index"),

    # SRAM variables (from Docs/Core/MemoryMap.md)
    "SaveSlot": (0x7EF38A, 1, "Current save slot"),
    "PlayerName": (0x7EF38B, 12, "Player name (12 characters)"),
    "Hearts": (0x7EF36C, 1, "Current hearts"),
    "MaxHearts": (0x7EF36B, 1, "Maximum hearts"),

    # Add more from memory map as needed
}


def load_memory_map_from_docs(oracle_path: Path) -> dict[str, tuple[int, int, str]]:
    """Load memory map from Oracle documentation.

    Parses Docs/Core/MemoryMap.md to extract all variables.
    Returns dictionary of {name: (address, size, description)}
    """
    memory_map_path = oracle_path / "Docs" / "Core" / "MemoryMap.md"

    if not memory_map_path.exists():
        print(f"Warning: Memory map not found at {memory_map_path}, using hardcoded map")
        return ORACLE_MEMORY_MAP

    # TODO: Parse MemoryMap.md to extract all variables
    # For now, return hardcoded map
    return ORACLE_MEMORY_MAP


def main():
    """CLI for generating training data from save states."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate training data from Mesen2 save states"
    )
    parser.add_argument(
        "--save-state",
        type=Path,
        required=True,
        help="Path to Mesen2 save state file (.mss)"
    )
    parser.add_argument(
        "--oracle-path",
        type=Path,
        default=Path("~/src/hobby/oracle-of-secrets"),
        help="Path to Oracle repository (for memory map)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )

    args = parser.parse_args()

    # Expand paths
    save_state_path = args.save_state.expanduser().resolve()
    oracle_path = args.oracle_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Save State Training Data Generator")
    print("=" * 60)
    print(f"\nSave state: {save_state_path}")
    print(f"Oracle path: {oracle_path}")
    print(f"Output: {output_path}")

    # Load memory map
    memory_map = load_memory_map_from_docs(oracle_path)
    print(f"\nLoaded {len(memory_map)} variables from memory map")

    # Generate samples
    generator = SaveStateTrainingGenerator(save_state_path, memory_map)

    count = 0
    with open(output_path, 'w') as f:
        for sample in generator.generate():
            f.write(json.dumps(sample.to_dict()) + '\n')
            count += 1

    print(f"\n✓ Generated {count} training samples from save state")
    print(f"  Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
