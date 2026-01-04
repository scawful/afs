#!/usr/bin/env python3
"""
Generate SNES hardware training data for Veran.

Creates training examples for:
- PPU registers ($2100-$213F)
- CPU registers ($4200-$421F)
- DMA/HDMA patterns ($4300-$437F)
- Common SNES initialization sequences
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "models" / "veran_snes_hardware.jsonl"

# SNES Register Reference
SNES_REGISTERS = {
    # PPU Registers
    "$2100": {
        "name": "INIDISP",
        "desc": "Screen Display Register",
        "bits": "Bit 7: Force blank (1=screen off), Bits 0-3: Brightness (0=dark, F=max)"
    },
    "$2101": {
        "name": "OBSEL",
        "desc": "Object Size and Character Address",
        "bits": "Bits 5-7: Sprite size, Bits 3-4: Name select, Bits 0-2: Base address"
    },
    "$2105": {
        "name": "BGMODE",
        "desc": "Background Mode and Tile Size",
        "bits": "Bits 0-2: BG mode (0-7), Bit 3: BG3 priority, Bits 4-7: BG tile sizes"
    },
    "$2106": {
        "name": "MOSAIC",
        "desc": "Screen Pixelation",
        "bits": "Bits 4-7: Mosaic size, Bits 0-3: BG enable flags"
    },
    "$2107": {
        "name": "BG1SC",
        "desc": "BG1 Tilemap Address and Size",
        "bits": "Bits 2-7: Base address, Bits 0-1: H/V mirror (0=32x32, 1=64x32, 2=32x64, 3=64x64)"
    },
    "$2115": {
        "name": "VMAIN",
        "desc": "VRAM Address Increment Mode",
        "bits": "Bit 7: Increment on high/low, Bits 0-1: Increment amount (0=1, 1=32, 2=128)"
    },
    "$2116": {
        "name": "VMADDL",
        "desc": "VRAM Address Low Byte",
        "bits": "Low 8 bits of VRAM word address"
    },
    "$2117": {
        "name": "VMADDH",
        "desc": "VRAM Address High Byte",
        "bits": "High 8 bits of VRAM word address"
    },
    "$2118": {
        "name": "VMDATAL",
        "desc": "VRAM Data Write Low",
        "bits": "Write low byte of VRAM data"
    },
    "$2119": {
        "name": "VMDATAH",
        "desc": "VRAM Data Write High",
        "bits": "Write high byte of VRAM data"
    },
    "$2121": {
        "name": "CGADD",
        "desc": "CGRAM (Palette) Address",
        "bits": "8-bit palette index (0-255)"
    },
    "$2122": {
        "name": "CGDATA",
        "desc": "CGRAM Data Write",
        "bits": "15-bit BGR color (write twice: low then high)"
    },
    "$212C": {
        "name": "TM",
        "desc": "Main Screen Designation",
        "bits": "Bit 4: OBJ, Bits 0-3: BG1-BG4 enable"
    },
    "$212D": {
        "name": "TS",
        "desc": "Sub Screen Designation",
        "bits": "Bit 4: OBJ, Bits 0-3: BG1-BG4 enable"
    },
    "$2130": {
        "name": "CGWSEL",
        "desc": "Color Math Control A",
        "bits": "Bits 6-7: Clip colors, Bits 4-5: Prevent math, Bit 1: Subscreen, Bit 0: Direct color"
    },
    "$2131": {
        "name": "CGADSUB",
        "desc": "Color Math Control B",
        "bits": "Bit 7: Add/subtract, Bit 6: Half, Bits 0-5: Layer enable"
    },
    "$2132": {
        "name": "COLDATA",
        "desc": "Fixed Color Data",
        "bits": "Bit 7: Blue, Bit 6: Green, Bit 5: Red, Bits 0-4: Intensity"
    },
    "$2133": {
        "name": "SETINI",
        "desc": "Screen Mode/Video Select",
        "bits": "Bit 7: External sync, Bit 6: Mode 7 EXTBG, Bit 3: Pseudo-hires, Bit 2: Overscan, Bit 0: Interlace"
    },
    # APU Communication
    "$2140": {
        "name": "APUIO0",
        "desc": "APU Communication Port 0",
        "bits": "Read/write 8-bit data to SPC700"
    },
    "$2141": {
        "name": "APUIO1",
        "desc": "APU Communication Port 1",
        "bits": "Read/write 8-bit data to SPC700"
    },
    "$2142": {
        "name": "APUIO2",
        "desc": "APU Communication Port 2",
        "bits": "Read/write 8-bit data to SPC700"
    },
    "$2143": {
        "name": "APUIO3",
        "desc": "APU Communication Port 3",
        "bits": "Read/write 8-bit data to SPC700"
    },
    # WRAM Access
    "$2180": {
        "name": "WMDATA",
        "desc": "WRAM Data Read/Write",
        "bits": "Sequential WRAM access (auto-increments address)"
    },
    "$2181": {
        "name": "WMADDL",
        "desc": "WRAM Address Low",
        "bits": "Bits 0-7 of 17-bit WRAM address"
    },
    "$2182": {
        "name": "WMADDM",
        "desc": "WRAM Address Middle",
        "bits": "Bits 8-15 of 17-bit WRAM address"
    },
    "$2183": {
        "name": "WMADDH",
        "desc": "WRAM Address High",
        "bits": "Bit 0: Bank select (0=$7E, 1=$7F)"
    },
    # CPU Registers
    "$4200": {
        "name": "NMITIMEN",
        "desc": "Interrupt Enable Register",
        "bits": "Bit 7: NMI enable, Bits 4-5: IRQ mode, Bit 0: Auto-joypad read"
    },
    "$4201": {
        "name": "WRIO",
        "desc": "Programmable I/O Port",
        "bits": "I/O port output; Bit 7 triggers PPU counter latch"
    },
    "$4202": {
        "name": "WRMPYA",
        "desc": "Multiplicand A",
        "bits": "8-bit unsigned multiplicand"
    },
    "$4203": {
        "name": "WRMPYB",
        "desc": "Multiplicand B",
        "bits": "8-bit unsigned multiplier (triggers multiply on write)"
    },
    "$4204": {
        "name": "WRDIVL",
        "desc": "Dividend Low",
        "bits": "Low 8 bits of 16-bit dividend"
    },
    "$4205": {
        "name": "WRDIVH",
        "desc": "Dividend High",
        "bits": "High 8 bits of 16-bit dividend"
    },
    "$4206": {
        "name": "WRDIVB",
        "desc": "Divisor",
        "bits": "8-bit divisor (triggers divide on write)"
    },
    "$4207": {
        "name": "HTIMEL",
        "desc": "H-Count Timer Low",
        "bits": "Low 8 bits of H-counter compare value"
    },
    "$4209": {
        "name": "VTIMEL",
        "desc": "V-Count Timer Low",
        "bits": "Low 8 bits of V-counter compare value"
    },
    "$420B": {
        "name": "MDMAEN",
        "desc": "DMA Enable",
        "bits": "Bits 0-7: Enable DMA channels 0-7 (triggers transfer)"
    },
    "$420C": {
        "name": "HDMAEN",
        "desc": "HDMA Enable",
        "bits": "Bits 0-7: Enable HDMA channels 0-7"
    },
    "$420D": {
        "name": "MEMSEL",
        "desc": "ROM Access Speed",
        "bits": "Bit 0: 1=FastROM (6 cycles), 0=SlowROM (8 cycles)"
    },
    "$4210": {
        "name": "RDNMI",
        "desc": "NMI Flag and Version",
        "bits": "Bit 7: NMI flag (cleared on read), Bits 0-3: CPU version"
    },
    "$4211": {
        "name": "TIMEUP",
        "desc": "IRQ Flag",
        "bits": "Bit 7: IRQ flag (cleared on read)"
    },
    "$4212": {
        "name": "HVBJOY",
        "desc": "PPU Status",
        "bits": "Bit 7: V-blank, Bit 6: H-blank, Bit 0: Auto-joypad in progress"
    },
    "$4214": {
        "name": "RDDIVL",
        "desc": "Division Quotient Low",
        "bits": "Low 8 bits of division result"
    },
    "$4215": {
        "name": "RDDIVH",
        "desc": "Division Quotient High",
        "bits": "High 8 bits of division result"
    },
    "$4216": {
        "name": "RDMPYL",
        "desc": "Multiply/Remainder Low",
        "bits": "Low byte of multiply product or division remainder"
    },
    "$4217": {
        "name": "RDMPYH",
        "desc": "Multiply/Remainder High",
        "bits": "High byte of multiply product or division remainder"
    },
    "$4218": {
        "name": "JOY1L",
        "desc": "Joypad 1 Data Low",
        "bits": "A, X, L, R buttons"
    },
    "$4219": {
        "name": "JOY1H",
        "desc": "Joypad 1 Data High",
        "bits": "B, Y, Select, Start, Up, Down, Left, Right"
    },
    # DMA Registers (Channel 0)
    "$4300": {
        "name": "DMAP0",
        "desc": "DMA Channel 0 Control",
        "bits": "Bit 7: Direction (0=A→B, 1=B→A), Bits 0-2: Transfer mode"
    },
    "$4301": {
        "name": "BBAD0",
        "desc": "DMA Channel 0 B-Bus Address",
        "bits": "PPU register address ($21xx) - just the low byte"
    },
    "$4302": {
        "name": "A1T0L",
        "desc": "DMA Channel 0 Source Address Low",
        "bits": "Low byte of source address"
    },
    "$4303": {
        "name": "A1T0H",
        "desc": "DMA Channel 0 Source Address High",
        "bits": "High byte of source address"
    },
    "$4304": {
        "name": "A1B0",
        "desc": "DMA Channel 0 Source Bank",
        "bits": "Bank byte of source address"
    },
    "$4305": {
        "name": "DAS0L",
        "desc": "DMA Channel 0 Size Low",
        "bits": "Low byte of transfer size (0=65536 bytes)"
    },
    "$4306": {
        "name": "DAS0H",
        "desc": "DMA Channel 0 Size High",
        "bits": "High byte of transfer size"
    },
}

# Training examples for SNES hardware patterns
SNES_EXAMPLES = [
    # Screen Display Control
    {
        "code": "LDA #$80\nSTA $2100",
        "explanation": """Enables force blank (screen off).

$2100 is INIDISP (Initial Display Settings):
- Bit 7 = 1: Force blank enabled (screen blanked, VRAM accessible)
- Bits 0-3 = 0: Brightness at minimum

Force blank is required when writing to VRAM outside V-blank. The PPU ignores rendering and allows unrestricted VRAM access.""",
        "category": "ppu_control"
    },
    {
        "code": "LDA #$0F\nSTA $2100",
        "explanation": """Sets screen to maximum brightness with display enabled.

$2100 is INIDISP:
- Bit 7 = 0: Force blank disabled (screen active)
- Bits 0-3 = $F: Maximum brightness (15)

Used after initialization to turn on the display.""",
        "category": "ppu_control"
    },
    {
        "code": "LDA #$8F\nSTA $2100",
        "explanation": """Force blank ON with max brightness setting preserved.

$2100 is INIDISP:
- Bit 7 = 1: Force blank (screen off)
- Bits 0-3 = $F: Brightness = 15 (stored for when display enabled)

Common during level transitions or when doing heavy VRAM updates.""",
        "category": "ppu_control"
    },
    # DMA to VRAM
    {
        "code": "LDA #$80\nSTA $2115\nLDX #$0000\nSTX $2116",
        "explanation": """Sets up VRAM address for writing.

$2115 is VMAIN (VRAM Address Increment Mode):
- Bit 7 = 1: Increment on writes to $2119 (VMDATAH)
- Bits 0-1 = 0: Increment by 1 word

$2116-$2117 is VMADDL/H (VRAM word address):
- Set to $0000 (start of VRAM)

This prepares for sequential VRAM writes using $2118/$2119.""",
        "category": "vram_access"
    },
    {
        "code": "LDA #$01\nSTA $4300\nLDA #$18\nSTA $4301",
        "explanation": """Configures DMA channel 0 for VRAM transfer.

$4300 is DMAP0 (DMA Control):
- Bit 7 = 0: Transfer A-bus → B-bus (CPU → PPU)
- Bits 0-2 = 1: Mode 1 (write to two registers: $2118, $2119)

$4301 is BBAD0 (B-bus destination):
- $18 = Low byte of $2118 (VMDATAL)

Mode 1 writes alternating bytes to $2118 and $2119 for 16-bit VRAM transfers.""",
        "category": "dma"
    },
    {
        "code": "REP #$20\nLDA #SourceData\nSTA $4302\nSEP #$20\nLDA #^SourceData\nSTA $4304",
        "explanation": """Sets DMA source address (24-bit).

$4302-$4303 (A1T0L/H): 16-bit address within bank
$4304 (A1B0): Bank byte

Uses REP/SEP to efficiently write 16-bit address, then 8-bit bank.
The ^SourceData syntax gets the bank byte of the label.""",
        "category": "dma"
    },
    {
        "code": "REP #$20\nLDA #$1000\nSTA $4305\nSEP #$20",
        "explanation": """Sets DMA transfer size to 4096 bytes.

$4305-$4306 (DAS0L/H): Transfer byte count
- $1000 = 4096 bytes

Note: If size is $0000, it transfers 65536 bytes (64KB).""",
        "category": "dma"
    },
    {
        "code": "LDA #$01\nSTA $420B",
        "explanation": """Triggers DMA transfer on channel 0.

$420B is MDMAEN (DMA Enable):
- Bit 0 = 1: Enable channel 0

Writing any non-zero value starts the DMA. CPU halts until transfer completes.
Channel bits: 0=$01, 1=$02, 2=$04, 3=$08, 4=$10, 5=$20, 6=$40, 7=$80.""",
        "category": "dma"
    },
    # Full DMA Pattern
    {
        "code": """LDA #$80
STA $2100
LDA #$80
STA $2115
STZ $2116
STZ $2117
LDA #$01
STA $4300
LDA #$18
STA $4301
LDA #<TileData
STA $4302
LDA #>TileData
STA $4303
LDA #^TileData
STA $4304
LDA #<TileDataEnd-TileData
STA $4305
LDA #>TileDataEnd-TileData
STA $4306
LDA #$01
STA $420B""",
        "explanation": """Complete DMA transfer of tile data to VRAM.

1. Force blank on ($2100) - required for VRAM access
2. Set VRAM increment mode ($2115 = $80, increment on high byte write)
3. Set VRAM destination address to $0000 ($2116-$2117)
4. Configure DMA channel 0:
   - Mode 1: Two-register write ($4300 = $01)
   - Target: VMDATAL at $2118 ($4301 = $18)
   - Source: TileData address ($4302-$4304)
   - Size: Length of tile data ($4305-$4306)
5. Trigger DMA ($420B = $01)

This is the standard pattern for uploading graphics to VRAM.""",
        "category": "dma"
    },
    # CGRAM (Palette)
    {
        "code": "STZ $2121\nLDA #$00\nSTA $2122\nLDA #$7C\nSTA $2122",
        "explanation": """Writes a single color to palette index 0.

$2121 is CGADD (palette address):
- $00 = First color slot

$2122 is CGDATA (color data, write twice):
- First write: Low byte ($00)
- Second write: High byte ($7C)

SNES uses 15-bit BGR: %0BBBBBGG GGGRRRRR
$7C00 = %01111100 00000000 = Blue (max blue, no green/red).""",
        "category": "palette"
    },
    {
        "code": "LDA #$00\nSTA $2121\nLDA #$01\nSTA $4300\nLDA #$22\nSTA $4301",
        "explanation": """Sets up DMA for palette transfer.

$2121 = $00: Start at palette index 0
$4300 = $01: Mode 1 (two-register write)
$4301 = $22: Target CGDATA ($2122)

Palette DMA writes color bytes in pairs (low, high) automatically.
Used for bulk palette uploads (faster than manual writes).""",
        "category": "palette"
    },
    # OAM (Sprites)
    {
        "code": "STZ $2102\nSTZ $2103",
        "explanation": """Resets OAM address to 0.

$2102-$2103 is OAMADDL/H:
- Sets the sprite table write address
- OAM holds 128 sprites × 4 bytes + 32 bytes high table

After this, writes to $2104 (OAMDATA) go to the start of sprite table.""",
        "category": "oam"
    },
    {
        "code": "LDA #$00\nSTA $4300\nLDA #$04\nSTA $4301",
        "explanation": """Configures DMA for OAM transfer.

$4300 = $00: Mode 0 (single register write)
$4301 = $04: Target OAMDATA ($2104)

Mode 0 writes sequential bytes to one register.
Used for uploading 544-byte sprite table to OAM.""",
        "category": "oam"
    },
    # Background Mode
    {
        "code": "LDA #$01\nSTA $2105",
        "explanation": """Sets BG mode 1 (4-color BG3, 16-color BG1/BG2).

$2105 is BGMODE:
- Bits 0-2 = 1: Mode 1
- Bits 4-7 = 0: All BGs use 8×8 tiles

Mode 1: BG1/BG2 = 4bpp (16 colors), BG3 = 2bpp (4 colors).
Most common mode for SNES games.""",
        "category": "bgmode"
    },
    {
        "code": "LDA #$09\nSTA $2105",
        "explanation": """Mode 1 with 16×16 tiles on BG1.

$2105 is BGMODE:
- Bits 0-2 = 1: Mode 1
- Bit 4 = 1: BG1 uses 16×16 tiles

16×16 tiles reduce tilemap size but use 4 tile slots per visual tile.""",
        "category": "bgmode"
    },
    {
        "code": "LDA #$07\nSTA $2105",
        "explanation": """Sets Mode 7 (rotation/scaling).

$2105 is BGMODE:
- Bits 0-2 = 7: Mode 7

Mode 7 provides a single 256-color layer with matrix transformation.
Used for pseudo-3D effects (F-Zero, Mario Kart, Pilotwings).""",
        "category": "bgmode"
    },
    # Tilemap Setup
    {
        "code": "LDA #$00\nSTA $2107",
        "explanation": """Sets BG1 tilemap to VRAM $0000, 32×32 tiles.

$2107 is BG1SC:
- Bits 2-7 = 0: Base address $0000
- Bits 0-1 = 0: 32×32 tile map (1 screen)

Address formula: Base = (value & $FC) << 8.""",
        "category": "tilemap"
    },
    {
        "code": "LDA #$04\nSTA $2107",
        "explanation": """Sets BG1 tilemap to VRAM $0400, 32×32 tiles.

$2107 is BG1SC:
- Bits 2-7 = $01: Base address $0400
- Bits 0-1 = 0: 32×32 tiles

Common setup: Tilemap at $0400, tile graphics at $0000.""",
        "category": "tilemap"
    },
    # Character Base
    {
        "code": "LDA #$01\nSTA $210B",
        "explanation": """Sets BG1 character base to VRAM $1000.

$210B is BG12NBA:
- Bits 0-3 = 1: BG1 tiles at $1000
- Bits 4-7 = 0: BG2 tiles at $0000

Address formula: Base = (nibble) << 12.""",
        "category": "tilemap"
    },
    # Layer Enable
    {
        "code": "LDA #$11\nSTA $212C",
        "explanation": """Enables BG1 and sprites on main screen.

$212C is TM (Main Screen Designation):
- Bit 0 = 1: BG1 enabled
- Bit 4 = 1: Sprites enabled

Main screen is the primary display layer.""",
        "category": "layers"
    },
    {
        "code": "LDA #$1F\nSTA $212C",
        "explanation": """Enables all layers on main screen.

$212C is TM:
- Bits 0-3 = $F: BG1-BG4 all enabled
- Bit 4 = 1: Sprites enabled

$1F = %00011111 enables everything.""",
        "category": "layers"
    },
    # NMI/IRQ
    {
        "code": "LDA #$81\nSTA $4200",
        "explanation": """Enables NMI and auto-joypad read.

$4200 is NMITIMEN:
- Bit 7 = 1: V-blank NMI enabled
- Bit 0 = 1: Auto-joypad read enabled

NMI fires at start of V-blank. Auto-joypad reads controller data to $4218-$421F.""",
        "category": "interrupts"
    },
    {
        "code": "LDA #$A1\nSTA $4200",
        "explanation": """Enables NMI, V-count IRQ, and auto-joypad.

$4200 is NMITIMEN:
- Bit 7 = 1: NMI enabled
- Bit 5 = 1: V-count IRQ enabled
- Bit 0 = 1: Auto-joypad enabled

V-count IRQ fires at scanline specified in $4209-$420A.""",
        "category": "interrupts"
    },
    {
        "code": "LDA $4210",
        "explanation": """Reads and acknowledges NMI flag.

$4210 is RDNMI:
- Bit 7: NMI occurred (cleared on read)
- Bits 0-3: CPU version

Must read this register in NMI handler to acknowledge the interrupt.""",
        "category": "interrupts"
    },
    # Hardware Multiply
    {
        "code": "LDA $10\nSTA $4202\nLDA $11\nSTA $4203\nNOP\nNOP\nNOP\nNOP\nLDA $4216\nSTA $12\nLDA $4217\nSTA $13",
        "explanation": """Hardware multiplication: $10 × $11 → $12-$13.

$4202 (WRMPYA): First multiplicand
$4203 (WRMPYB): Second multiplicand (triggers multiply)
$4216-$4217 (RDMPYL/H): 16-bit result

4 NOPs (or 8 cycles) wait for multiply to complete.
Result is unsigned 8-bit × 8-bit = 16-bit product.""",
        "category": "math"
    },
    # Hardware Divide
    {
        "code": "REP #$20\nLDA $10\nSTA $4204\nSEP #$20\nLDA $12\nSTA $4206\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nREP #$20\nLDA $4214\nSTA $14\nSEP #$20",
        "explanation": """Hardware division: $10-$11 ÷ $12 → quotient $14-$15.

$4204-$4205 (WRDIVL/H): 16-bit dividend
$4206 (WRDIVB): 8-bit divisor (triggers divide)
$4214-$4215 (RDDIVL/H): 16-bit quotient
$4216-$4217 (RDMPYL/H): 16-bit remainder

8 NOPs (16 cycles) wait for division to complete.
Result: 16-bit ÷ 8-bit = 16-bit quotient + 16-bit remainder.""",
        "category": "math"
    },
    # Joypad Reading
    {
        "code": "-\nLDA $4212\nAND #$01\nBNE -\nLDA $4218\nSTA $10\nLDA $4219\nSTA $11",
        "explanation": """Waits for auto-joypad, then reads controller 1.

$4212 is HVBJOY:
- Bit 0: Auto-joypad in progress

Wait loop ensures joypad read is complete before accessing data.

$4218-$4219 (JOY1L/H) contains:
- $4218: A, X, L, R buttons
- $4219: B, Y, Select, Start, D-pad""",
        "category": "input"
    },
    # HDMA Setup
    {
        "code": "LDA #$00\nSTA $4300\nLDA #$21\nSTA $4301\nLDA #<HDMATable\nSTA $4302\nLDA #>HDMATable\nSTA $4303\nLDA #^HDMATable\nSTA $4304\nLDA #$01\nSTA $420C",
        "explanation": """Sets up HDMA channel 0 for palette effects.

$4300 = $00: Mode 0, A→B (indirect table to PPU)
$4301 = $21: Target CGADD ($2121)
$4302-$4304: HDMA table address

$420C = $01: Enable HDMA channel 0

HDMA transfers data each scanline based on a table.
Used for gradient effects, water reflections, etc.""",
        "category": "hdma"
    },
    {
        "code": "LDA #$02\nSTA $4300\nLDA #$0D\nSTA $4301",
        "explanation": """HDMA for BG1 horizontal scroll (parallax).

$4300 = $02: Mode 2 (write same byte to register twice)
$4301 = $0D: Target BG1HOFS ($210D)

Mode 2 writes each table byte twice, good for 16-bit scroll registers.
Creates parallax scrolling effects per scanline.""",
        "category": "hdma"
    },
    # Mode 7
    {
        "code": "LDA #$80\nSTA $211A",
        "explanation": """Mode 7 settings: Screen-over, repeat mode.

$211A is M7SEL:
- Bit 7 = 1: Fill with tile 0 outside map
- Bit 6 = 0: Normal (vs. fill with transparent)
- Bits 0-1 = 0: No flipping

Used for F-Zero style tracks where outside area shows sky/tile 0.""",
        "category": "mode7"
    },
    {
        "code": "LDA #$01\nSTA $211B\nSTZ $211B\nSTZ $211C\nSTZ $211C\nSTZ $211D\nSTZ $211D\nLDA #$01\nSTA $211E\nSTZ $211E",
        "explanation": """Sets Mode 7 identity matrix (no transformation).

Matrix registers (write twice each: low, high):
$211B (M7A) = $0100: cos(0) = 1.0
$211C (M7B) = $0000: sin(0) = 0
$211D (M7C) = $0000: -sin(0) = 0
$211E (M7D) = $0100: cos(0) = 1.0

Fixed-point format: 1 sign bit, 7 integer, 8 fraction.
$0100 = 1.0 in this format.""",
        "category": "mode7"
    },
    # V-blank Wait
    {
        "code": "-\nLDA $4212\nBPL -",
        "explanation": """Waits for V-blank to start.

$4212 is HVBJOY:
- Bit 7: V-blank flag (1 during V-blank)

BPL branches while bit 7 is 0 (not in V-blank).
Loop exits when V-blank begins.""",
        "category": "timing"
    },
    {
        "code": "-\nLDA $4212\nBMI -",
        "explanation": """Waits for V-blank to end.

$4212 is HVBJOY:
- Bit 7: V-blank flag

BMI branches while bit 7 is 1 (in V-blank).
Loop exits when active display begins.""",
        "category": "timing"
    },
    # FastROM
    {
        "code": "LDA #$01\nSTA $420D",
        "explanation": """Enables FastROM mode.

$420D is MEMSEL:
- Bit 0 = 1: FastROM (6 master cycles per ROM access)
- Bit 0 = 0: SlowROM (8 master cycles)

Only works if ROM supports fast timing. Speeds up execution ~25%.""",
        "category": "system"
    },
    # WRAM DMA
    {
        "code": "STZ $2181\nSTZ $2182\nLDA #$7E\nSTA $2183\nLDA #$08\nSTA $4300\nLDA #$80\nSTA $4301",
        "explanation": """Sets up DMA to clear WRAM.

$2181-$2183: WRAM address = $7E:0000 (start of WRAM)

$4300 = $08: Mode 0, fixed source (don't increment A-bus)
$4301 = $80: Target WMDATA ($2180)

With fixed source, same byte is written repeatedly.
Used to zero-fill WRAM at startup.""",
        "category": "wram"
    },
    # Common Init Pattern
    {
        "code": """SEI
CLC
XCE
REP #$30
LDA #$0000
TCD
LDX #$01FF
TXS
SEP #$30""",
        "explanation": """Standard SNES initialization sequence.

1. SEI: Disable IRQ interrupts
2. CLC + XCE: Switch to native 65816 mode (clear emulation)
3. REP #$30: 16-bit A, X, Y
4. LDA #$0000, TCD: Set direct page to $0000
5. LDX #$01FF, TXS: Set stack pointer to $01FF
6. SEP #$30: Back to 8-bit mode

This is the minimum CPU setup at game startup.""",
        "category": "init"
    },
]

def generate_training_data():
    """Generate SNES hardware training examples."""
    examples = []

    for ex in SNES_EXAMPLES:
        examples.append({
            "instruction": f"Explain this 65816 code:\n{ex['code']}",
            "output": ex["explanation"].strip(),
            "input": "",
            "domain": "snes_hardware",
            "source": "curated",
            "intent": "explanation",
            "category": ex["category"]
        })

    # Also generate individual register explanations
    for addr, info in SNES_REGISTERS.items():
        code = f"STA {addr}"
        explanation = f"""Writes to {info['name']} ({addr}).

{info['desc']}.

Bit fields: {info['bits']}."""

        examples.append({
            "instruction": f"Explain this 65816 code:\n{code}",
            "output": explanation,
            "input": "",
            "domain": "snes_hardware",
            "source": "register_reference",
            "intent": "explanation",
            "category": "register"
        })

    return examples

def main():
    print("Generating SNES hardware training data...")

    examples = generate_training_data()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} examples")
    print(f"  - Pattern examples: {len(SNES_EXAMPLES)}")
    print(f"  - Register references: {len(SNES_REGISTERS)}")
    print(f"Saved to {OUTPUT_FILE}")

    # Show category breakdown
    categories = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
