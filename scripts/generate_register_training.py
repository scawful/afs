#!/usr/bin/env python3
"""Generate training data with explicit register name emphasis.

The key insight: Put the register name FIRST and REPEAT it.
This helps the model strongly associate addresses with names.
"""

import json
from pathlib import Path

# Register definitions with multiple training variations
REGISTERS = {
    # PPU Control Registers
    "$2100": {
        "name": "INIDISP",
        "full_name": "Initial Display Settings",
        "category": "ppu_control",
        "examples": [
            {
                "code": "LDA #$80\nSTA $2100",
                "output": "INIDISP ($2100) - Force blank ON.\n\nThis writes to the INIDISP register at $2100:\n- Bit 7 = 1: Force blank enabled (screen off)\n- Bits 0-3 = 0: Brightness minimum\n\nINIDISP controls display blanking and brightness. Force blank is required for VRAM access outside V-blank."
            },
            {
                "code": "LDA #$0F\nSTA $2100",
                "output": "INIDISP ($2100) - Display ON, max brightness.\n\nThis writes $0F to INIDISP:\n- Bit 7 = 0: Force blank OFF (display active)\n- Bits 0-3 = $F: Maximum brightness (15)\n\nINIDISP is written after initialization to enable the screen."
            },
            {
                "code": "LDA #$8F\nSTA $2100",
                "output": "INIDISP ($2100) - Force blank with brightness preserved.\n\nWrites $8F to INIDISP register:\n- Bit 7 = 1: Force blank ON\n- Bits 0-3 = $F: Brightness = 15 (preserved for later)\n\nCommon during VRAM updates or scene transitions."
            },
            {
                "code": "LDA #$00\nSTA $2100",
                "output": "INIDISP ($2100) - Force blank OFF, zero brightness.\n\nWrites $00 to INIDISP:\n- Bit 7 = 0: Display enabled\n- Bits 0-3 = 0: Brightness at minimum (black)\n\nUsed for fade-to-black effects by gradually decreasing brightness."
            },
            {
                "code": "STZ $2100",
                "output": "INIDISP ($2100) - Clears display settings.\n\nSTZ stores zero to INIDISP at $2100:\n- Display ON (no force blank)\n- Brightness = 0 (black screen)\n\nINIDISP register controls PPU display state."
            },
        ]
    },
    "$2105": {
        "name": "BGMODE",
        "full_name": "BG Mode and Character Size",
        "category": "ppu_control",
        "examples": [
            {
                "code": "LDA #$01\nSTA $2105",
                "output": "BGMODE ($2105) - Sets Mode 1.\n\nWrites to BGMODE register at $2105:\n- Bits 0-2 = 1: BG Mode 1\n- Bits 4-7 = 0: All BGs use 8x8 tiles\n\nBGMODE Mode 1: BG1/BG2 = 16 colors (4bpp), BG3 = 4 colors (2bpp)."
            },
            {
                "code": "LDA #$07\nSTA $2105",
                "output": "BGMODE ($2105) - Sets Mode 7.\n\nWrites $07 to BGMODE:\n- Bits 0-2 = 7: BG Mode 7 (rotation/scaling)\n\nBGMODE Mode 7 provides matrix transformation for pseudo-3D effects."
            },
            {
                "code": "LDA #$09\nSTA $2105",
                "output": "BGMODE ($2105) - Mode 1 with 16x16 tiles.\n\nWrites to BGMODE:\n- Bits 0-2 = 1: Mode 1\n- Bit 4 = 1: BG1 uses 16x16 tiles\n\nBGMODE controls background graphics mode and tile sizes."
            },
        ]
    },
    "$2115": {
        "name": "VMAIN",
        "full_name": "VRAM Address Increment Mode",
        "category": "vram",
        "examples": [
            {
                "code": "LDA #$80\nSTA $2115",
                "output": "VMAIN ($2115) - VRAM increment on high byte.\n\nWrites to VMAIN register at $2115:\n- Bit 7 = 1: Increment address after writing to VMDATAH ($2119)\n- Bits 0-1 = 0: Increment by 1 word\n\nVMAIN controls how VRAM address increments during writes."
            },
            {
                "code": "LDA #$00\nSTA $2115",
                "output": "VMAIN ($2115) - VRAM increment on low byte.\n\nWrites $00 to VMAIN:\n- Bit 7 = 0: Increment after writing to VMDATAL ($2118)\n- Bits 0-1 = 0: Increment by 1 word\n\nVMAIN is set before VRAM operations."
            },
            {
                "code": "LDA #$81\nSTA $2115",
                "output": "VMAIN ($2115) - Increment by 32 words.\n\nWrites $81 to VMAIN register:\n- Bit 7 = 1: Increment on high byte write\n- Bits 0-1 = 1: Increment by 32 words\n\nVMAIN 32-word increment is used for column-based tilemap updates."
            },
        ]
    },
    "$2116": {
        "name": "VMADDL",
        "full_name": "VRAM Address Low",
        "category": "vram",
        "examples": [
            {
                "code": "STZ $2116\nSTZ $2117",
                "output": "VMADDL/VMADDH ($2116-$2117) - Sets VRAM address to $0000.\n\nVMADDL ($2116) and VMADDH ($2117) form the 16-bit VRAM word address.\nWriting zeros sets address to start of VRAM.\n\nVRAM is 64KB addressed as 32K words."
            },
            {
                "code": "LDX #$1000\nSTX $2116",
                "output": "VMADDL/VMADDH ($2116-$2117) - Sets VRAM address to $1000.\n\n16-bit write to VMADDL sets both low and high bytes.\nVRAM address $1000 is commonly used for tilemap data.\n\nVMADDL/H must be set before VRAM read/write operations."
            },
        ]
    },
    "$2121": {
        "name": "CGADD",
        "full_name": "Palette Address",
        "category": "palette",
        "examples": [
            {
                "code": "STZ $2121",
                "output": "CGADD ($2121) - Sets palette address to 0.\n\nWrites to CGADD register at $2121:\n- Sets palette index for CGDATA writes\n- $00 = first color (typically transparent)\n\nCGADD points to one of 256 palette entries."
            },
            {
                "code": "LDA #$80\nSTA $2121",
                "output": "CGADD ($2121) - Sets palette address to 128.\n\nWrites $80 to CGADD:\n- Palette index 128 (second half of palette)\n\nCGADD is used before palette uploads to set starting color."
            },
        ]
    },
    "$2122": {
        "name": "CGDATA",
        "full_name": "Palette Data",
        "category": "palette",
        "examples": [
            {
                "code": "LDA #$1F\nSTA $2122\nLDA #$00\nSTA $2122",
                "output": "CGDATA ($2122) - Writes one color (red).\n\nCGDATA at $2122 is written twice per color:\n- First write: Low byte ($1F = red component)\n- Second write: High byte ($00)\n\nSNES uses 15-bit BGR: %0BBBBBGGGGGRRRRR\n$001F = pure red (max red, no green/blue)."
            },
            {
                "code": "LDA #$00\nSTA $2122\nLDA #$7C\nSTA $2122",
                "output": "CGDATA ($2122) - Writes one color (blue).\n\nTwo writes to CGDATA for 15-bit color:\n- Low byte: $00\n- High byte: $7C\n\n$7C00 = %01111100 00000000 = pure blue.\nCGDATA auto-increments CGADD after each color."
            },
        ]
    },
    "$212C": {
        "name": "TM",
        "full_name": "Main Screen Designation",
        "category": "layers",
        "examples": [
            {
                "code": "LDA #$01\nSTA $212C",
                "output": "TM ($212C) - Enables BG1 on main screen.\n\nWrites to TM register at $212C:\n- Bit 0 = 1: BG1 enabled\n\nTM controls which layers appear on the main screen.\nBits: 0=BG1, 1=BG2, 2=BG3, 3=BG4, 4=Sprites."
            },
            {
                "code": "LDA #$11\nSTA $212C",
                "output": "TM ($212C) - Enables BG1 and sprites.\n\nWrites $11 to TM:\n- Bit 0 = 1: BG1 enabled\n- Bit 4 = 1: Sprites enabled\n\nTM is the main screen layer enable register."
            },
            {
                "code": "LDA #$1F\nSTA $212C",
                "output": "TM ($212C) - Enables all layers.\n\nWrites $1F to TM:\n- Bits 0-3 = 1: All four BG layers\n- Bit 4 = 1: Sprites\n\nTM controls main screen visibility for all 5 layers."
            },
        ]
    },
    # CPU Registers
    "$4200": {
        "name": "NMITIMEN",
        "full_name": "NMI/Timer/Joypad Enable",
        "category": "cpu_control",
        "examples": [
            {
                "code": "LDA #$81\nSTA $4200",
                "output": "NMITIMEN ($4200) - Enables NMI and joypad auto-read.\n\nWrites to NMITIMEN register at $4200:\n- Bit 7 = 1: NMI enabled (V-blank interrupt)\n- Bit 0 = 1: Joypad auto-read enabled\n\nNMITIMEN is the main interrupt control register."
            },
            {
                "code": "LDA #$80\nSTA $4200",
                "output": "NMITIMEN ($4200) - Enables NMI only.\n\nWrites $80 to NMITIMEN:\n- Bit 7 = 1: NMI enabled\n- Bit 0 = 0: Manual joypad read\n\nNMITIMEN enables V-blank interrupt for game loop timing."
            },
            {
                "code": "LDA #$01\nSTA $4200",
                "output": "NMITIMEN ($4200) - Enables joypad auto-read only.\n\nWrites $01 to NMITIMEN:\n- Bit 7 = 0: NMI disabled\n- Bit 0 = 1: Joypad auto-read enabled\n\nNMITIMEN without NMI is used during initialization."
            },
            {
                "code": "STZ $4200",
                "output": "NMITIMEN ($4200) - Disables all interrupts.\n\nClears NMITIMEN register:\n- NMI disabled\n- H/V IRQ disabled\n- Joypad auto-read disabled\n\nNMITIMEN is often cleared during initialization."
            },
        ]
    },
    "$420B": {
        "name": "MDMAEN",
        "full_name": "DMA Enable",
        "category": "dma",
        "examples": [
            {
                "code": "LDA #$01\nSTA $420B",
                "output": "MDMAEN ($420B) - Triggers DMA on channel 0.\n\nWrites to MDMAEN register at $420B:\n- Bit 0 = 1: Enable DMA channel 0\n\nMDMAEN starts DMA transfer. CPU halts until complete.\nChannel bits: 0=$01, 1=$02, 2=$04, 3=$08, etc."
            },
            {
                "code": "LDA #$03\nSTA $420B",
                "output": "MDMAEN ($420B) - Triggers DMA on channels 0 and 1.\n\nWrites $03 to MDMAEN:\n- Bit 0 = 1: Channel 0\n- Bit 1 = 1: Channel 1\n\nMDMAEN can trigger multiple DMA channels simultaneously."
            },
            {
                "code": "LDA #$80\nSTA $420B",
                "output": "MDMAEN ($420B) - Triggers DMA on channel 7.\n\nWrites $80 to MDMAEN:\n- Bit 7 = 1: Channel 7 enabled\n\nMDMAEN register enables any of 8 DMA channels (0-7)."
            },
        ]
    },
    "$420C": {
        "name": "HDMAEN",
        "full_name": "HDMA Enable",
        "category": "dma",
        "examples": [
            {
                "code": "LDA #$01\nSTA $420C",
                "output": "HDMAEN ($420C) - Enables HDMA on channel 0.\n\nWrites to HDMAEN register at $420C:\n- Bit 0 = 1: HDMA channel 0 active\n\nHDMAEN enables per-scanline DMA for effects like gradients and wavy backgrounds."
            },
            {
                "code": "STZ $420C",
                "output": "HDMAEN ($420C) - Disables all HDMA.\n\nClears HDMAEN register to stop all HDMA channels.\n\nHDMAEN should be disabled during VRAM updates."
            },
        ]
    },
    "$4210": {
        "name": "RDNMI",
        "full_name": "NMI Flag and CPU Version",
        "category": "cpu_status",
        "examples": [
            {
                "code": "LDA $4210",
                "output": "RDNMI ($4210) - Reads NMI status and clears flag.\n\nReads from RDNMI register at $4210:\n- Bit 7: NMI occurred (reading clears this)\n- Bits 0-3: CPU version\n\nRDNMI must be read to acknowledge NMI."
            },
            {
                "code": "BIT $4210",
                "output": "RDNMI ($4210) - Tests NMI flag.\n\nBIT instruction reads RDNMI and sets N flag from bit 7.\n\nRDNMI reading clears the NMI flag. Used in NMI handler."
            },
        ]
    },
    "$4212": {
        "name": "HVBJOY",
        "full_name": "H/V Blank and Joypad Status",
        "category": "cpu_status",
        "examples": [
            {
                "code": "LDA $4212\nAND #$01\nBNE WaitJoy",
                "output": "HVBJOY ($4212) - Waits for joypad read to complete.\n\nReads HVBJOY register at $4212:\n- Bit 0: Joypad auto-read in progress\n\nHVBJOY must be polled before reading joypad data."
            },
            {
                "code": "-\n  BIT $4212\n  BPL -",
                "output": "HVBJOY ($4212) - Waits for V-blank.\n\nBIT reads HVBJOY, N flag = bit 7 (V-blank status).\nLoop until bit 7 = 1 (V-blank active).\n\nHVBJOY is used for frame synchronization."
            },
        ]
    },
    # DMA Channel Registers
    "$4300": {
        "name": "DMAP0",
        "full_name": "DMA Control Channel 0",
        "category": "dma",
        "examples": [
            {
                "code": "LDA #$01\nSTA $4300",
                "output": "DMAP0 ($4300) - DMA mode 1 (two-register write).\n\nWrites to DMAP0 register at $4300:\n- Bits 0-2 = 1: Mode 1 (write to two consecutive registers)\n- Bit 7 = 0: A-bus to B-bus (CPU to PPU)\n\nDMAP0 mode 1 is used for VRAM transfers via $2118/$2119."
            },
            {
                "code": "LDA #$00\nSTA $4300",
                "output": "DMAP0 ($4300) - DMA mode 0 (single register).\n\nWrites $00 to DMAP0:\n- Bits 0-2 = 0: Mode 0 (one register)\n- Bit 7 = 0: CPU to PPU direction\n\nDMAP0 mode 0 for OAM or single-byte transfers."
            },
            {
                "code": "LDA #$80\nSTA $4300",
                "output": "DMAP0 ($4300) - DMA reverse direction.\n\nWrites $80 to DMAP0:\n- Bit 7 = 1: B-bus to A-bus (PPU to CPU)\n\nDMAP0 reverse DMA reads from PPU registers."
            },
        ]
    },
    "$4301": {
        "name": "BBAD0",
        "full_name": "DMA B-Bus Address Channel 0",
        "category": "dma",
        "examples": [
            {
                "code": "LDA #$18\nSTA $4301",
                "output": "BBAD0 ($4301) - DMA target is VRAM data low.\n\nWrites to BBAD0 register at $4301:\n- $18 = low byte of $2118 (VMDATAL)\n\nBBAD0 sets the B-bus destination. Combined with mode 1, writes to $2118/$2119."
            },
            {
                "code": "LDA #$22\nSTA $4301",
                "output": "BBAD0 ($4301) - DMA target is palette data.\n\nWrites $22 to BBAD0:\n- $22 = low byte of $2122 (CGDATA)\n\nBBAD0 for palette DMA transfers."
            },
            {
                "code": "LDA #$04\nSTA $4301",
                "output": "BBAD0 ($4301) - DMA target is OAM data.\n\nWrites $04 to BBAD0:\n- $04 = low byte of $2104 (OAMDATA)\n\nBBAD0 for sprite table DMA uploads."
            },
        ]
    },
    "$4302": {
        "name": "A1T0L",
        "full_name": "DMA Source Address Low Channel 0",
        "category": "dma",
        "examples": [
            {
                "code": "LDA #<SourceData\nSTA $4302\nLDA #>SourceData\nSTA $4303",
                "output": "A1T0L/A1T0H ($4302-$4303) - Sets DMA source address.\n\nWrites 16-bit address within bank:\n- A1T0L ($4302): Low byte of source\n- A1T0H ($4303): High byte of source\n\nA1T0 is the source address for DMA channel 0."
            },
        ]
    },
    "$4304": {
        "name": "A1B0",
        "full_name": "DMA Source Bank Channel 0",
        "category": "dma",
        "examples": [
            {
                "code": "LDA #^SourceData\nSTA $4304",
                "output": "A1B0 ($4304) - Sets DMA source bank.\n\nWrites bank byte to A1B0 register:\n- ^Label syntax gets the bank byte\n\nA1B0 completes the 24-bit DMA source address."
            },
        ]
    },
    "$4305": {
        "name": "DAS0L",
        "full_name": "DMA Size Low Channel 0",
        "category": "dma",
        "examples": [
            {
                "code": "REP #$20\nLDA #$1000\nSTA $4305\nSEP #$20",
                "output": "DAS0L/DAS0H ($4305-$4306) - Sets DMA size to 4096 bytes.\n\n16-bit write to DAS0:\n- $1000 = 4096 bytes to transfer\n\nDAS0 is the byte count. $0000 means 65536 bytes."
            },
            {
                "code": "LDA #$00\nSTA $4305\nLDA #$02\nSTA $4306",
                "output": "DAS0L/DAS0H ($4305-$4306) - Sets DMA size to 512 bytes.\n\nWrites $0200 to DAS0:\n- 512 bytes for tilemap transfer\n\nDAS0 register pair holds transfer byte count."
            },
        ]
    },
    # Joypad registers
    "$4218": {
        "name": "JOY1L",
        "full_name": "Joypad 1 Data Low",
        "category": "input",
        "examples": [
            {
                "code": "LDA $4218",
                "output": "JOY1L ($4218) - Reads joypad 1 buttons.\n\nReads from JOY1L register at $4218 after auto-read:\n- Bit 7: A button\n- Bit 6: X button\n- Bit 5: L button\n- Bit 4: R button\n\nJOY1L contains A, X, L, R buttons for player 1."
            },
            {
                "code": "REP #$20\nLDA $4218\nSEP #$20",
                "output": "JOY1L/JOY1H ($4218-$4219) - Reads full joypad state.\n\n16-bit read of JOY1 register pair:\n- Low byte: A, X, L, R (and more)\n- High byte: B, Y, Select, Start, Up, Down, Left, Right\n\nJOY1L/H contain all button states after auto-read."
            },
        ]
    },
}


def generate_training_data():
    """Generate training examples from register definitions."""
    examples = []

    for address, reg_info in REGISTERS.items():
        for ex in reg_info["examples"]:
            examples.append({
                "instruction": f"Explain this 65816 code:\n{ex['code']}",
                "output": ex["output"],
                "input": "",
                "domain": "snes_hardware",
                "source": "register_emphasis",
                "intent": "explanation",
                "category": reg_info["category"],
                "register_name": reg_info["name"],
                "register_address": address,
            })

    return examples


def main():
    output_dir = Path(__file__).parent.parent / "models"
    output_file = output_dir / "veran_register_emphasis.jsonl"

    examples = generate_training_data()

    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} training examples")
    print(f"Saved to: {output_file}")

    # Show register coverage
    print("\nRegister coverage:")
    for addr, info in REGISTERS.items():
        print(f"  {addr} {info['name']}: {len(info['examples'])} examples")

    print(f"\nTotal registers: {len(REGISTERS)}")


if __name__ == "__main__":
    main()
