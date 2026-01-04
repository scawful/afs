"""Code pattern libraries for curriculum-based generation.

Provides concrete code examples for each expert domain and difficulty level,
enabling the LLM to generate actual assembly code rather than explanations.

IMPORTANT: All code uses ASAR assembler syntax, NOT ca65/cc65 syntax.
Key ASAR syntax rules:
- Address operators: label&$FFFF (low word), label>>8 (high byte), ^label (bank)
- Mode: lorom/hirom at file start, org $address for placement
- Local labels: .label (dot prefix) or -/+ for anonymous
- Data: db/dw/dl (not .byte/.word)
- No .SEGMENT directives
"""

from __future__ import annotations


# =============================================================================
# ASAR SYNTAX REFERENCE - Critical for correct code generation
# =============================================================================

ASAR_SYNTAX = {
    "header": """lorom  ; or hirom, sa1rom, etc.

org $008000  ; Place code at SNES address""",

    "address_operators": {
        "low_word": "REP #$20 : LDA #SpriteData&$FFFF  ; Low 16 bits (16-bit A)",
        "high_byte": "SEP #$20 : LDA #SpriteData>>8  ; High byte of low word",
        "bank_byte": "SEP #$20 : LDA #SpriteData>>16  ; Bank byte (8-bit A, use >>16 NOT ^)",
        "full_24bit": "LDA.l SpriteData  ; Long addressing (24-bit)",
    },

    "dma_setup_correct": """
; CORRECT ASAR DMA setup (NOT ca65 .LOWORD/.BANKBYTE)
LoadSpritesToVRAM:
    PHP
    REP #$20                ; 16-bit A for word writes
    LDA #$1801              ; Mode 1, write to $2118
    STA $4300               ; Writes to $4300 AND $4301
    LDA #SpriteData&$FFFF   ; Source address low word
    STA $4302               ; Writes to $4302 AND $4303
    SEP #$20                ; 8-bit A for bank byte!
    LDA #SpriteData>>16     ; Bank byte (use >>16, NOT ^)
    STA $4304
    REP #$20                ; Back to 16-bit
    LDA #$0800              ; Transfer size
    STA $4305
    LDA #$4000              ; VRAM destination
    STA $2116
    SEP #$20                ; 8-bit for control registers
    LDA #$80
    STA $2115               ; VRAM increment mode
    LDA #$01
    STA $420B               ; Start DMA
    PLP
    RTS

SpriteData:
    db $00,$00,$00,$00      ; Tile data here
""",

    "labels": {
        "global": "MyRoutine:",
        "local": ".local_label  ; Dot prefix for local scope",
        "anonymous_back": "-  ; Anonymous label (branch back with BRA -)",
        "anonymous_fwd": "+  ; Anonymous label (branch forward with BRA +)",
        "sublabel": "Routine_SubLabel:  ; Underscore convention",
    },

    "data_directives": {
        "byte": "db $42, $43, $44  ; Define bytes",
        "word": "dw $1234, $5678   ; Define 16-bit words",
        "long": "dl $123456        ; Define 24-bit long",
        "fill": "padbyte $00 : pad $8100  ; Fill to address",
        "table": "table_data:\n    dw label1, label2, label3",
    },

    "common_mistakes": {
        "wrong_loword": ".LOWORD(label)  ; WRONG - ca65 syntax",
        "right_loword": "label&$FFFF     ; CORRECT - ASAR syntax",
        "wrong_bankbyte": ".BANKBYTE(label)  ; WRONG - ca65 syntax",
        "also_wrong_bank": "^label            ; WRONG - not valid in ASAR",
        "right_bankbyte": "label>>16         ; CORRECT - ASAR bank byte",
        "wrong_segment": ".SEGMENT \"CODE\"  ; WRONG - ca65 syntax",
        "right_org": "org $008000          ; CORRECT - ASAR syntax",
        "wrong_byte": ".BYTE $42  ; WRONG - ca65 syntax",
        "right_byte": "db $42     ; CORRECT - ASAR syntax",
    },

    "wait_loops": """
; Wait for VBlank (ASAR style)
-
    LDA $4212
    AND #$80
    BEQ -
    ; Now in VBlank

; Wait for auto-joypad
WaitJoypad:
    LDA $4212
    AND #$01
    BNE WaitJoypad
""",

    "joypad_read": """
; Complete joypad read routine (ASAR)
ReadJoypad:
    PHP
    REP #$20
-   LDA $4212           ; Wait for auto-read
    AND #$0001
    BNE -
    LDA JoyCurrent      ; Save previous
    STA JoyPrevious
    LDA $4218           ; Read new state
    STA JoyCurrent
    EOR JoyPrevious     ; Calculate newly pressed
    AND JoyCurrent
    STA JoyPressed
    PLP
    RTS

JoyCurrent: dw $0000
JoyPrevious: dw $0000
JoyPressed: dw $0000
""",

    "complete_dma_example": """
; Full DMA routine - ROM to VRAM (ASAR syntax)
lorom

org $008000

!VMAIN    = $2115
!VMADDL   = $2116
!VMDATAL  = $2118
!DMAP0    = $4300
!BBAD0    = $4301
!A1T0L    = $4302
!A1B0     = $4304
!DAS0L    = $4305
!MDMAEN   = $420B

TransferTilesToVRAM:
    PHP
    REP #$20

    ; Set VRAM destination
    LDA #$4000          ; VRAM word address
    STA !VMADDL

    ; Configure DMA channel 0
    LDA #$1801          ; Mode 1 (2-reg), target $2118
    STA !DMAP0

    ; Source address
    LDA #TileData&$FFFF
    STA !A1T0L
    SEP #$20
    LDA #TileData>>16   ; Bank byte (use >>16, NOT ^)
    STA !A1B0

    ; Transfer size
    REP #$20
    LDA #$1000          ; 4KB of tiles
    STA !DAS0L

    ; Set increment mode and start
    SEP #$20
    LDA #$80
    STA !VMAIN
    LDA #$01
    STA !MDMAEN

    PLP
    RTS

TileData:
    db $00,$00,$00,$00  ; Placeholder tile data
""",
}


# =============================================================================
# DIN PATTERNS - Optimization before/after pairs
# =============================================================================

DIN_PATTERNS = {
    "basic": {
        "redundant_loads": [
            # (before, after, description)
            (
                "LDA #$00\nSTA $10\nLDA #$00\nSTA $11\nLDA #$00\nSTA $12",
                "STZ $10\nSTZ $11\nSTZ $12",
                "Replace LDA #$00 + STA with STZ"
            ),
            (
                "LDA $10\nSTA $20\nLDA $10\nSTA $21\nLDA $10\nSTA $22",
                "LDA $10\nSTA $20\nSTA $21\nSTA $22",
                "Eliminate redundant loads of same address"
            ),
            (
                "LDA #$01\nSTA $10\nLDA #$01\nSTA $11",
                "LDA #$01\nSTA $10\nSTA $11",
                "Reuse accumulator value"
            ),
            (
                "LDX #$00\nSTX $10\nLDX #$00\nSTX $11",
                "LDX #$00\nSTX $10\nSTX $11",
                "Reuse index register value"
            ),
        ],
        "register_mode": [
            (
                "SEP #$20\nSEP #$10",
                "SEP #$30",
                "Combine separate mode switches"
            ),
            (
                "REP #$20\nREP #$10",
                "REP #$30",
                "Combine 16-bit mode switches"
            ),
            (
                "SEP #$20\nLDA #$05\nSEP #$20",
                "SEP #$20\nLDA #$05",
                "Remove redundant mode switch"
            ),
        ],
        "branch_optimization": [
            (
                "CMP #$00\nBEQ label",
                "BEQ label",
                "CMP #$00 is redundant before BEQ"
            ),
            (
                "LDA $10\nCMP #$00\nBNE label",
                "LDA $10\nBNE label",
                "LDA sets zero flag automatically"
            ),
        ],
        "increment_decrement": [
            (
                "LDA $10\nCLC\nADC #$01\nSTA $10",
                "INC $10",
                "Use INC for adding 1"
            ),
            (
                "LDA $10\nSEC\nSBC #$01\nSTA $10",
                "DEC $10",
                "Use DEC for subtracting 1"
            ),
        ],
    },
    "intermediate": {
        "loop_optimization": [
            (
                "LDX #$10\nloop:\nLDA $1000,X\nSTA $2000,X\nINX\nCPX #$20\nBNE loop",
                "LDX #$0F\nloop:\nLDA $1000,X\nSTA $2000,X\nDEX\nBPL loop",
                "Count down to avoid CPX"
            ),
            (
                "LDY #$00\nloop:\nLDA ($10),Y\nSTA $2000,Y\nINY\nCPY #$10\nBNE loop",
                "LDY #$0F\nloop:\nLDA ($10),Y\nSTA $2000,Y\nDEY\nBPL loop",
                "Count down for zero-page indirect"
            ),
        ],
        "addressing_modes": [
            (
                "LDA #$7E\nPHA\nPLB\nLDA $0000\nPHK\nPLB",
                "LDA $7E0000",
                "Use long addressing instead of bank switch"
            ),
            (
                "LDA $10\nASL A\nTAX\nLDA table,X",
                "LDX $10\nLDA table,X\nLDA table+1,X",
                "Direct index for 16-bit table lookup"
            ),
        ],
        "16bit_operations": [
            (
                "SEP #$20\nLDA $10\nSTA $20\nLDA $11\nSTA $21",
                "REP #$20\nLDA $10\nSTA $20\nSEP #$20",
                "Use 16-bit mode for word copy"
            ),
            (
                "SEP #$20\nSTZ $10\nSTZ $11\nSTZ $12\nSTZ $13",
                "REP #$20\nSTZ $10\nSTZ $12\nSEP #$20",
                "Use 16-bit STZ for clearing"
            ),
        ],
        "stack_usage": [
            (
                "PHA\nPHX\nPHY\n; code\nPLY\nPLX\nPLA",
                "PHD\nTSC\nTCD\n; use direct page\nPLD",
                "Use direct page for temp storage"
            ),
        ],
    },
    "advanced": {
        "dma_setup": [
            (
                "LDA #$80\nSTA $2115\nLDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\n"
                "LDA #<src\nSTA $4302\nLDA #>src\nSTA $4303\nLDA #^src\nSTA $4304\n"
                "LDA #<size\nSTA $4305\nLDA #>size\nSTA $4306\nLDA #$01\nSTA $420B",
                "REP #$20\nLDA #$1801\nSTA $4300\nLDA #src\nSTA $4302\nLDA #size\nSTA $4305\n"
                "SEP #$20\nLDA #^src\nSTA $4304\nLDA #$80\nSTA $2115\nLDA #$01\nSTA $420B",
                "Use 16-bit stores for DMA setup"
            ),
        ],
        "hdma_optimization": [
            (
                "; Per-scanline HDMA table with repeated values\n"
                "db $01, $FF\ndb $01, $FF\ndb $01, $FF\ndb $01, $FE",
                "; HDMA repeat mode\ndb $83, $FF  ; 3 lines at $FF\ndb $01, $FE\ndb $00",
                "Use HDMA repeat mode for constant values"
            ),
        ],
        "unrolled_loops": [
            (
                "LDX #$08\nloop:\nLDA $1000,X\nSTA $2000,X\nDEX\nBPL loop",
                "LDA $1000\nSTA $2000\nLDA $1001\nSTA $2001\nLDA $1002\nSTA $2002\n"
                "LDA $1003\nSTA $2003\nLDA $1004\nSTA $2004\nLDA $1005\nSTA $2005\n"
                "LDA $1006\nSTA $2006\nLDA $1007\nSTA $2007\nLDA $1008\nSTA $2008",
                "Unroll small loops to eliminate branch overhead"
            ),
        ],
    },
    "expert": {
        "mode7_calc": [
            (
                "; Multiply by cosine table lookup\n"
                "LDA angle\nASL A\nTAX\nLDA cos_table,X\nSTA $211B\n"
                "LDA cos_table+1,X\nSTA $211C\nLDA pos_x\nSTA $211B\n"
                "LDA pos_x+1\nSTA $211C\nLDA $2134\nSTA result\nLDA $2135\nSTA result+1",
                "; Use hardware multiplier with signed extension\n"
                "REP #$30\nLDA angle\nAND #$00FF\nASL A\nTAX\nLDA cos_table,X\n"
                "SEP #$20\nSTA $211B\nXBA\nSTA $211C\nLDA pos_x\nSTA $211B\n"
                "LDA pos_x+1\nSTA $211C\nREP #$20\nLDA $2134\nSTA result\nSEP #$20",
                "Optimize mode 7 matrix calculation"
            ),
        ],
        "audio_streaming": [
            (
                "; Wait for APU ready\nwait:\nLDA $2140\nCMP #$AA\nBNE wait\n"
                "; Send byte\nLDA sample\nSTA $2140\n; Wait for echo\nwait2:\n"
                "LDA $2140\nCMP sample\nBNE wait2",
                "; Block transfer to APU\nLDA #<block\nSTA $4302\nLDA #>block\n"
                "STA $4303\nLDA #^block\nSTA $4304\nLDA #<size\nSTA $4305\n"
                "LDA #>size\nSTA $4306\n; Use IPL block transfer\nLDA #$CC\nSTA $2140",
                "Use APU IPL block transfer instead of byte-by-byte"
            ),
        ],
    },
}


# =============================================================================
# FARORE BUGS - Common bugs with diagnosis and fix
# =============================================================================

FARORE_BUGS = {
    "basic": {
        "mode_mismatch": [
            {
                "buggy": "LDA #$1234\nSTA $10",
                "issue": "16-bit immediate value with 8-bit accumulator mode",
                "fix": "REP #$20\nLDA #$1234\nSTA $10\nSEP #$20",
                "explanation": "The accumulator is in 8-bit mode but #$1234 is a 16-bit value. "
                              "Need REP #$20 to switch to 16-bit mode first.",
            },
            {
                "buggy": "SEP #$20\nLDA #$00\nSTA $10\nSTA $11",
                "issue": "Storing 8-bit value expecting 16-bit clear",
                "fix": "REP #$20\nLDA #$0000\nSTA $10\nSEP #$20",
                "explanation": "If $10-$11 need to be cleared as a 16-bit value, "
                              "must use 16-bit mode to write both bytes at once.",
            },
        ],
        "addressing_errors": [
            {
                "buggy": "LDA $7E0100",
                "issue": "Long address in bank $00 context",
                "fix": "LDA $7E0100",
                "explanation": "This is actually correct if in bank 00, but if the intent "
                              "is to read WRAM, the code should verify bank setup or use "
                              "explicit long addressing with proper bank byte.",
            },
            {
                "buggy": "LDA ($10)\nSTA $20",
                "issue": "Missing Y register for indirect indexed",
                "fix": "LDA ($10),Y\nSTA $20",
                "explanation": "65816 indirect addressing requires ,Y for indexed indirect. "
                              "Without it, this is stack-relative indirect which is different.",
            },
        ],
        "branch_range": [
            {
                "buggy": "BRA far_label  ; more than 128 bytes away",
                "issue": "Branch target out of range",
                "fix": "BRL far_label  ; or JMP far_label",
                "explanation": "BRA can only branch -128 to +127 bytes. For longer jumps, "
                              "use BRL (branch long) or JMP.",
            },
        ],
    },
    "intermediate": {
        "register_corruption": [
            {
                "buggy": "JSR subroutine\nSTX $10  ; X was modified in subroutine",
                "issue": "Register corrupted by subroutine call",
                "fix": "PHX\nJSR subroutine\nPLX\nSTX $10",
                "explanation": "Subroutines may modify registers. Preserve important values "
                              "on the stack before calling.",
            },
            {
                "buggy": "LDA $10\nJSL $7E8000\nSTA $20  ; A was corrupted by long call",
                "issue": "Accumulator corrupted by JSL",
                "fix": "PHA\nJSL $7E8000\nPLA\nLDA $10\nSTA $20",
                "explanation": "The JSL target routine modified A. Either preserve A before "
                              "the call or reload it after.",
            },
        ],
        "dma_issues": [
            {
                "buggy": "LDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\n"
                        "LDA #<src\nSTA $4302\nLDA #>src\nSTA $4303\n"
                        "LDA #<size\nSTA $4305\nLDA #>size\nSTA $4306\n"
                        "LDA #$01\nSTA $420B",
                "issue": "Missing source bank register $4304",
                "fix": "LDA #$01\nSTA $4300\nLDA #$18\nSTA $4301\n"
                      "LDA #<src\nSTA $4302\nLDA #>src\nSTA $4303\n"
                      "LDA #^src\nSTA $4304\n"
                      "LDA #<size\nSTA $4305\nLDA #>size\nSTA $4306\n"
                      "LDA #$01\nSTA $420B",
                "explanation": "DMA requires the source bank in $4304. Without it, the bank "
                              "defaults to whatever was last written, causing wrong data transfer.",
            },
        ],
        "stack_imbalance": [
            {
                "buggy": "subroutine:\nPHA\nPHX\n; ... code ...\nPLA\nRTS",
                "issue": "Stack imbalance - pushed X but only pulled A",
                "fix": "subroutine:\nPHA\nPHX\n; ... code ...\nPLX\nPLA\nRTS",
                "explanation": "Every PHA needs a matching PLA, and every PHX needs a PLX. "
                              "Stack must be balanced before RTS or the return address is wrong.",
            },
        ],
    },
    "advanced": {
        "nmi_race": [
            {
                "buggy": "; Main loop\nLDA $10\nINC A\nSTA $10\n"
                        "; NMI handler\nNMI:\nLDA $10\nSTA $2100\nRTI",
                "issue": "Race condition between main loop and NMI",
                "fix": "; Main loop\nSEI\nLDA $10\nINC A\nSTA $10\nCLI\n"
                      "; NMI handler\nNMI:\nLDA $10\nSTA $2100\nRTI",
                "explanation": "NMI can interrupt between LDA and STA in main loop, causing "
                              "the NMI to read a stale value. Use SEI/CLI to protect critical "
                              "sections, or use a double-buffer approach.",
            },
        ],
        "hdma_corruption": [
            {
                "buggy": "; Change HDMA table mid-frame\nLDA #<new_table\nSTA $4302",
                "issue": "Modifying HDMA registers while HDMA is active",
                "fix": "; Wait for VBLANK\nwait:\nLDA $4212\nBPL wait\n"
                      "LDA #<new_table\nSTA $4302",
                "explanation": "HDMA registers should only be modified during VBLANK. "
                              "Changing them mid-frame causes visual corruption.",
            },
        ],
    },
    "expert": {
        "save_corruption": [
            {
                "buggy": "; Save routine\nLDA $7EF340\nSTA $700000\n; ... more saves ...\n"
                        "; No checksum or validation",
                "issue": "Save data has no integrity check",
                "fix": "; Save routine with checksum\nLDX #$0000\nLDA #$0000\n"
                      "checksum_loop:\nCLC\nADC $7EF340,X\nINX\nINX\n"
                      "CPX #$0500\nBNE checksum_loop\nEOR #$5A5A\nSTA $7EF840\n"
                      "; Now save data...",
                "explanation": "Save data should include a checksum to detect corruption. "
                              "Calculate checksum of save area and store it for validation on load.",
            },
        ],
    },
}


# =============================================================================
# NAYRU HARDWARE - Hardware documentation and reference code
# =============================================================================

NAYRU_HARDWARE = {
    "link_state": {
        "description": "Link's state variables in WRAM",
        "addresses": {
            "x_coord": ("$7E0022", "Link X position (low byte)"),
            "x_coord_high": ("$7E0023", "Link X position (high byte)"),
            "y_coord": ("$7E0020", "Link Y position (low byte)"),
            "y_coord_high": ("$7E0021", "Link Y position (high byte)"),
            "direction": ("$7E002F", "Link facing direction (0=up,2=down,4=left,6=right)"),
            "speed": ("$7E0031", "Link movement speed"),
            "health": ("$7E036C", "Current health"),
            "max_health": ("$7E036D", "Maximum health"),
            "state": ("$7E005D", "Link action state"),
        },
        "example_read": "LDA $7E0022  ; Get Link X low byte\nLDA $7E0020  ; Get Link Y low byte",
        "example_write": "LDA #$80\nSTA $7E0022  ; Set Link X to $80",
    },
    "sprite_oam": {
        "description": "OAM (Object Attribute Memory) for sprites",
        "addresses": {
            "oam_buffer": ("$7E0800", "OAM buffer (512+32 bytes)"),
            "oam_x": ("$7E0800", "Sprite X position"),
            "oam_y": ("$7E0801", "Sprite Y position"),
            "oam_tile": ("$7E0802", "Tile number"),
            "oam_attr": ("$7E0803", "Attributes (palette, priority, flip)"),
        },
        "example": """
; Write a sprite to OAM buffer
LDX oam_index
LDA sprite_x
STA $0800,X     ; X position
LDA sprite_y
STA $0801,X     ; Y position
LDA tile_num
STA $0802,X     ; Tile
LDA #$30        ; Palette 3, priority 0
STA $0803,X     ; Attributes
""",
    },
    "ppu_registers": {
        "description": "PPU control registers",
        "registers": {
            "INIDISP": ("$2100", "Display control (brightness, force blank)"),
            "BGMODE": ("$2105", "BG mode and tile size"),
            "VMAIN": ("$2115", "VRAM address increment mode"),
            "VMADDL": ("$2116", "VRAM address low"),
            "VMADDH": ("$2117", "VRAM address high"),
            "VMDATAL": ("$2118", "VRAM data write low"),
            "VMDATAH": ("$2119", "VRAM data write high"),
            "CGADD": ("$2121", "CGRAM address"),
            "CGDATA": ("$2122", "CGRAM data write"),
        },
    },
    "dma_channels": {
        "description": "DMA channel registers",
        "pattern": {
            "DMAPn": ("$43n0", "DMA parameters"),
            "BBADn": ("$43n1", "B-bus address"),
            "A1TnL": ("$43n2", "A-bus address low"),
            "A1TnH": ("$43n3", "A-bus address high"),
            "A1Bn": ("$43n4", "A-bus bank"),
            "DASnL": ("$43n5", "Transfer size low"),
            "DASnH": ("$43n6", "Transfer size high"),
        },
        "example": """
; DMA transfer to VRAM
LDA #$80
STA $2115      ; VRAM increment mode
REP #$20
LDA #$6000
STA $2116      ; VRAM address
LDA #$1801
STA $4300      ; 2-register, write to $2118
LDA #source
STA $4302      ; Source address
LDA #$0800
STA $4305      ; Size (2048 bytes)
SEP #$20
LDA #^source
STA $4304      ; Source bank
LDA #$01
STA $420B      ; Start DMA channel 0
""",
    },
    "joypad": {
        "description": "Controller input registers",
        "addresses": {
            "JOY1L": ("$4218", "Joypad 1 low byte"),
            "JOY1H": ("$4219", "Joypad 1 high byte"),
            "JOY2L": ("$421A", "Joypad 2 low byte"),
            "JOY2H": ("$421B", "Joypad 2 high byte"),
        },
        "button_masks": {
            "A": "$80 (high byte)",
            "X": "$40 (high byte)",
            "L": "$20 (high byte)",
            "R": "$10 (high byte)",
            "B": "$80 (low byte)",
            "Y": "$40 (low byte)",
            "Select": "$20 (low byte)",
            "Start": "$10 (low byte)",
            "Up": "$08 (low byte)",
            "Down": "$04 (low byte)",
            "Left": "$02 (low byte)",
            "Right": "$01 (low byte)",
        },
        "example": """
; Read joypad and check for A button
LDA $4219      ; High byte
AND #$80       ; A button mask
BNE a_pressed
""",
    },
}


# =============================================================================
# NAYRU CODE TEMPLATES - Example code for generation tasks
# =============================================================================

NAYRU_TEMPLATES = {
    "basic": [
        {
            "task": "Read joypad input",
            "code": """
; Read joypad 1 into zero page
; Assumes auto-joypad is enabled ($4200 bit 0)
wait_joypad:
    LDA $4212
    AND #$01
    BNE wait_joypad   ; Wait for auto-read complete

    LDA $4218
    STA joy1_lo       ; Store low byte (directions + B/Y/Select/Start)
    LDA $4219
    STA joy1_hi       ; Store high byte (A/X/L/R)
    RTS
""",
        },
        {
            "task": "Simple delay loop",
            "code": """
; Delay for A frames (pass frame count in A)
delay_frames:
    STA delay_count
.loop:
    WAI               ; Wait for interrupt (NMI)
    DEC delay_count
    BNE .loop
    RTS
""",
        },
        {
            "task": "Copy memory block",
            "code": """
; Copy X bytes from (source) to (dest)
; Source/dest are 16-bit zero page pointers
copy_block:
    PHY
    LDY #$00
.loop:
    LDA (source),Y
    STA (dest),Y
    INY
    DEX
    BNE .loop
    PLY
    RTS
""",
        },
        {
            "task": "Set screen brightness",
            "code": """
; Set screen brightness (0-15 in A)
; 0 = off, 15 = full brightness
set_brightness:
    AND #$0F          ; Mask to valid range
    STA $2100         ; INIDISP
    RTS
""",
        },
    ],
    "intermediate": [
        {
            "task": "Update position based on input",
            "code": """
; Move Link based on D-pad input
; Uses $7E0022 (X) and $7E0020 (Y)
update_position:
    LDA joy1_lo

    ; Check right
    LSR A
    BCC .check_left
    LDA $7E0022
    CLC
    ADC #$02          ; Move right 2 pixels
    STA $7E0022
    BRA .check_vertical

.check_left:
    LSR A
    BCC .check_vertical
    LDA $7E0022
    SEC
    SBC #$02          ; Move left 2 pixels
    STA $7E0022

.check_vertical:
    LDA joy1_lo

    ; Check down
    AND #$04
    BEQ .check_up
    LDA $7E0020
    CLC
    ADC #$02          ; Move down 2 pixels
    STA $7E0020
    RTS

.check_up:
    LDA joy1_lo
    AND #$08
    BEQ .done
    LDA $7E0020
    SEC
    SBC #$02          ; Move up 2 pixels
    STA $7E0020
.done:
    RTS
""",
        },
        {
            "task": "Spawn sprite at coordinates",
            "code": """
; Spawn a new sprite at (spawn_x, spawn_y)
; Returns sprite slot in X, or carry set if no slots
spawn_sprite:
    ; Find free slot
    LDX #$00
.find_slot:
    LDA sprite_active,X
    BEQ .found
    INX
    CPX #$10          ; Max 16 sprites
    BNE .find_slot
    SEC               ; No free slots
    RTS

.found:
    LDA #$01
    STA sprite_active,X
    LDA spawn_x
    STA sprite_x,X
    LDA spawn_y
    STA sprite_y,X
    LDA spawn_type
    STA sprite_type,X
    CLC               ; Success
    RTS
""",
        },
    ],
    "advanced": [
        {
            "task": "DMA chain for VRAM update",
            "code": """
; Execute DMA chain from chain_ptr
; Chain format: [size_lo, size_hi, vram_lo, vram_hi, data...]
; Terminated by size=0
execute_dma_chain:
    LDY #$00
.loop:
    ; Get size
    LDA (chain_ptr),Y
    STA $4305
    INY
    LDA (chain_ptr),Y
    STA $4306
    ORA $4305
    BEQ .done         ; Size 0 = end of chain
    INY

    ; Get VRAM address
    LDA (chain_ptr),Y
    STA $2116
    INY
    LDA (chain_ptr),Y
    STA $2117
    INY

    ; Calculate data address
    TYA
    CLC
    ADC chain_ptr
    STA $4302
    LDA chain_ptr+1
    ADC #$00
    STA $4303
    LDA chain_ptr+2
    STA $4304

    ; Execute DMA
    LDA #$01
    STA $420B

    ; Advance past data
    TYA
    CLC
    ADC $4305
    TAY
    LDA #$00
    ADC $4306
    BEQ .loop         ; Continue if no overflow

.done:
    RTS
""",
        },
        {
            "task": "Palette fade routine",
            "code": """
; Fade palette to black over A frames
; Modifies palette buffer at $7EC000
fade_to_black:
    STA fade_frames

.frame_loop:
    WAI               ; Wait for NMI

    ; Process all 256 colors
    LDX #$00
.color_loop:
    ; Get color (BGR555)
    REP #$20
    LDA $7EC000,X

    ; Fade red (bits 0-4)
    PHA
    AND #$001F
    BEQ .red_done
    DEC A
.red_done:
    STA temp
    PLA

    ; Fade green (bits 5-9)
    PHA
    AND #$03E0
    BEQ .green_done
    SEC
    SBC #$0020
.green_done:
    ORA temp
    STA temp
    PLA

    ; Fade blue (bits 10-14)
    AND #$7C00
    BEQ .blue_done
    SEC
    SBC #$0400
.blue_done:
    ORA temp
    STA $7EC000,X
    SEP #$20

    INX
    INX
    CPX #$0200        ; 256 colors * 2 bytes
    BNE .color_loop

    DEC fade_frames
    BNE .frame_loop
    RTS
""",
        },
    ],
    "expert": [
        {
            "task": "Dialog box with text scrolling",
            "code": """
; Initialize dialog box
; Text pointer in dialog_text
init_dialog:
    ; Draw box frame
    JSR draw_dialog_frame

    ; Initialize state
    STZ dialog_char_x
    STZ dialog_char_y
    STZ dialog_delay
    LDA #$01
    STA dialog_active
    RTS

; Update dialog - call once per frame
update_dialog:
    LDA dialog_active
    BEQ .done

    ; Character delay
    DEC dialog_delay
    BPL .done
    LDA #$02          ; 2 frame delay between chars
    STA dialog_delay

    ; Get next character
    LDA (dialog_text)
    BEQ .end_dialog   ; Null terminator
    CMP #$FE
    BEQ .newline
    CMP #$FF
    BEQ .wait_button

    ; Draw character
    JSR draw_dialog_char

    ; Advance position
    INC dialog_char_x
    LDA dialog_char_x
    CMP #$18          ; 24 chars per line
    BCC .next_char
    STZ dialog_char_x
    INC dialog_char_y

.next_char:
    ; Advance text pointer
    INC dialog_text
    BNE .done
    INC dialog_text+1
.done:
    RTS

.newline:
    STZ dialog_char_x
    INC dialog_char_y
    BRA .next_char

.wait_button:
    LDA joy1_pressed
    AND #$80          ; A button
    BEQ .done
    BRA .next_char

.end_dialog:
    STZ dialog_active
    RTS
""",
        },
    ],
}


# =============================================================================
# VERAN CONCEPTS - Code examples needing explanation
# =============================================================================

VERAN_EXAMPLES = {
    "basic": [
        {
            "code": "LDA #$42\nSTA $7E0100",
            "concepts": ["immediate addressing", "absolute long addressing", "WRAM"],
        },
        {
            "code": "LDX #$10\nloop:\nDEX\nBNE loop",
            "concepts": ["loop", "index register", "branch on condition"],
        },
        {
            "code": "REP #$20\nLDA $10\nSEP #$20",
            "concepts": ["processor mode", "16-bit accumulator", "zero page"],
        },
        {
            "code": "CLC\nADC #$05",
            "concepts": ["carry flag", "addition", "immediate value"],
        },
        {
            "code": "PHA\nPHX\nJSR routine\nPLX\nPLA",
            "concepts": ["stack operations", "register preservation", "subroutine call"],
        },
    ],
    "intermediate": [
        {
            "code": """
LDA #$80
STA $2115      ; VMAIN
REP #$20
LDA #$6000
STA $2116      ; VMADDL/H
LDA #$1801
STA $4300      ; DMAP0 + BBAD0
""",
            "concepts": ["VRAM access", "DMA setup", "word writes to registers"],
        },
        {
            "code": """
LDA $4212
AND #$80
BEQ wait_vblank
""",
            "concepts": ["VBLANK detection", "hardware status register", "timing"],
        },
        {
            "code": """
LDA $10
ASL A
ASL A
CLC
ADC $10
TAX
LDA table,X
""",
            "concepts": ["multiply by 5", "table lookup", "index calculation"],
        },
    ],
    "advanced": [
        {
            "code": """
; HDMA gradient table
db $20, $00    ; 32 lines, color 0
db $10, $08    ; 16 lines, color 8
db $10, $10    ; 16 lines, color 16
db $00         ; End
""",
            "concepts": ["HDMA table format", "scanline effects", "color gradients"],
        },
        {
            "code": """
NMI:
    REP #$30
    PHA
    PHX
    PHY
    PHD
    PHB
    ; ... NMI code ...
    PLB
    PLD
    PLY
    PLX
    PLA
    RTI
""",
            "concepts": ["NMI handler", "register preservation", "interrupt return"],
        },
    ],
    "expert": [
        {
            "code": """
; Mode 7 matrix setup
LDA angle
ASL A
TAX
LDA m7_cos,X
STA $211B
STA $211D      ; A and D = cos
LDA m7_sin,X
STA $211C      ; B = sin
EOR #$FF
INC A
STA $211E      ; C = -sin
""",
            "concepts": ["Mode 7", "rotation matrix", "trigonometry", "PPU matrix"],
        },
    ],
}


# =============================================================================
# REAL PATTERNS FROM oracle-of-secrets (ALTTP/Zelda ROM hack)
# =============================================================================

ORACLE_PATTERNS = {
    "din": {
        "sprite_movement": (
            # Before: naive approach
            "LDA.w SprXSpeed, X\nCLC\nADC.w SprX, X\nSTA.w SprX, X",
            # After: sub-pixel precision from sprite_functions.asm
            """Sprite_MoveHoriz:
{
  LDA.w SprXSpeed, X : BEQ .no_velocity
    ASL #4 : CLC : ADC.w SprXRound, X : STA.w SprXRound, X
    LDY.b #$00
    LDA.w SprXSpeed, X
    PHP : LSR #4 : PLP : BPL ++
      ORA.b #$F0
      DEY
    ++
    ADC.w SprX, X : STA.w SprX, X
    TYA : ADC.w SprXH, X : STA.w SprXH, X
  .no_velocity
  RTL
}""",
            "Sub-pixel precision movement with carry propagation"
        ),
        "loop_unrolling": (
            # Before: tight loop
            "LDX #$FF\n.loop\n  LDA.w menu_frame, X\n  STA.w $1000, X\n  DEX\n  BPL .loop",
            # After: unrolled 8x for tilemap
            """REP #$30
LDX.w #$FE
.loop
  LDA.w menu_frame, X : STA.w $1000, X
  LDA.w menu_frame+$100, X : STA.w $1100, X
  LDA.w menu_frame+$200, X : STA.w $1200, X
  LDA.w menu_frame+$300, X : STA.w $1300, X
  DEX : DEX
BPL .loop""",
            "8x unrolled loop eliminates branch overhead"
        ),
        "speed_inversion": (
            # Before: multiply then subtract
            "LDA.w SprXSpeed, X\nSTA $00\nLDA #$00\nSEC\nSBC $00\nSTA.w SprXSpeed, X",
            # After: efficient two's complement
            "LDA.w SprXSpeed, X\nEOR.b #$FF : INC A\nSTA.w SprXSpeed, X",
            "Two's complement inversion (XOR + INC)"
        ),
    },
    "nayru": {
        "sprite_altitude": """
; Sprite_MoveAltitude - Z-axis movement with subpixel
; Maintains floating/bounce effect
Sprite_MoveAltitude:
{
  LDA.w SprTimerF, X : ASL #4
  CLC : ADC.w SprHeightS, X : STA.w SprHeightS, X
  LDA.w SprTimerF, X : PHP : LSR #4 : PLP : BPL .positive
    ORA.b #$F0
  .positive
  ADC.w SprHeight, X : STA.w SprHeight, X
  RTL
}""",
        "sprite_init": """
; Sprite Initialization - cache position for damage reset
Sprite_IceBlock_Prep:
{
  PHB : PHK : PLB
  ; Cache Sprite position
  LDA.w SprX, X : STA.w SprMiscD, X
  LDA.w SprY, X : STA.w SprMiscE, X
  LDA.w SprXH, X : STA.w SprMiscF, X
  LDA.w SprYH, X : STA.w SprMiscG, X
  STZ.w SprDefl, X
  LDA.w SprHitbox, X : ORA.b #$09 : STA.w SprHitbox, X
  PLB
  RTL
}""",
        "follower_direction": """
; Follower watches Link and updates head/body offsets
Follower_WatchLink:
{
  JSL Sprite_IsToRightOfPlayer : TYA : BEQ .right
    LDA.b #$40 : STA.w FollowerHeadOffset
    LDA.b #$60 : STA.w FollowerBodyOffset
    RTS
  .right
  LDA.b #$00 : STA.w FollowerHeadOffset
  LDA.b #$A0 : STA.w FollowerBodyOffset
  RTS
}""",
    },
    "farore": {
        "water_collision": {
            "buggy": """
; Water gate fill - missing collision update
WaterGate_FillComplete:
  STZ.b $1E
  STZ.b $1F
  JSL IrisSpotlight_ResetTable
  ; Player can now walk through water tiles!
  JML $01F3DA""",
            "issue": "Missing collision tilemap update after water fills",
            "fix": """
WaterGate_FillComplete_Hook:
{
  STZ.b $1E : STZ.b $1F
  JSL IrisSpotlight_ResetTable

  PHB : PHK : PLB
  SEP #$20
  LDA.b $A0  ; Current room

  CMP.b #$27 : BNE .check_room_25
    REP #$20
    LDA.w #WaterGate_Room27_Data : STA.b $00
    SEP #$20
    LDA.b #WaterGate_Room27_Data>>16 : STA.b $02
    JSR WaterGate_ApplyCollision
    JSR WaterGate_SetPersistenceFlag
    BRA .done

  .check_room_25
  CMP.b #$25 : BNE .done
    ; Apply room 25 collision...

  .done
  SEP #$30
  PLB
  JML $01F3DA
}""",
            "explanation": "Water fill must update $7F2000 collision tilemap and set SRAM persistence flag"
        },
        "palette_dispatch": {
            "buggy": """
; Palette switching without proper state
Palette_Update:
  LDA !CurrentMask
  CMP.b #$01 : BEQ .mask1
  CMP.b #$02 : BEQ .mask2
  ; Missing fallback case!
  RTL""",
            "issue": "Missing default case in mask comparison causes unpredictable palette",
            "fix": """
Palette_ArmorAndGloves:
{
  LDA !CurrentMask
  CMP.b #$01 : BEQ .deku_mask
  CMP.b #$02 : BEQ .zora_mask
  CMP.b #$03 : BEQ .wolf_mask
  JMP .original_sprite  ; Default fallback

  .deku_mask
  LDA.b #$35 : STA $BC
  JSL UpdateDekuPalette
  RTL

  .zora_mask
  LDA.b #$36 : STA $BC
  JSL UpdateZoraPalette
  RTL

  .wolf_mask
  LDA.b #$38 : STA $BC
  JSL UpdateWolfPalette
  RTL

  .original_sprite
  ; Original Link palette logic
  RTL
}""",
            "explanation": "State dispatch must always have a default case to prevent undefined behavior"
        },
    },
    "veran": {
        "minecart_docs": """
; =========================================================
; Minecart Sprite System Documentation
;
; Used in Goron Mines along with SwitchTrack and Mineswitch.
; Uses custom collision with Somaria track corner tiles.
;
; Cart States:
; - Inactive: Horizontal or vertical, waiting for player
; - Active: Moving in SprMiscB direction until:
;   1. Somaria Stop Tile - Halt and set next direction
;   2. Somaria Corner Track - Switch direction based on tile
;   3. Somaria Any Track - Switch based on player input
;   4. Dungeon Transition - Switch to follower mode
;
; Direction Values (SprMiscB):
; 0 = North, 1 = East, 2 = South, 3 = West
;
; Track System:
; - Up to 0x20 different subtypes possible
; - $0728+ stores which room each cart was left in
; - Allows cart persistence across room transitions
; =========================================================

North = $00
East  = $01
South = $02
West  = $03
!MinecartDirection = $0DE0  ; = SprMiscC
!MinecartTrackRoom = $0728  ; Room persistence
""",
        "sprite_props_docs": """
; Sprite Properties Memory Map
;
; $0E40 - Harmless/HVelocity/NbrTiles
;   Bit 7: Harmless (no damage)
;   Bit 6: Has velocity
;   Bits 0-5: Number of OAM tiles
;
; $0E50 - Sprite HP
; $0CD2 - Sprite Damage
;
; $0E60/$0F50 - Sprite Data
;   Bit 7: Death animation
;   Bit 6: Impervious to all
;   Bit 5: Small shadow
;   Bit 4: Has shadow
;   Bits 1-3: Palette
;
; $0F60 - Sprite Hitbox
;   Bit 7: Collision layer
;   Bit 6: Static (no movement)
;   Bit 5: Persist across rooms
;   Bits 0-4: Hitbox type
;
; $0BE0 - Prize/Interaction
;   Bit 7: Interaction type
;   Bit 6: Water sprite
;   Bit 5: Blockable
;   Bit 4: Sound effect
;   Bits 0-3: Prize drop
""",
    },
}


def get_oracle_pattern(domain: str, name: str | None = None) -> str | tuple | dict:
    """Get a pattern from the oracle-of-secrets collection."""
    import random

    patterns = ORACLE_PATTERNS.get(domain, {})
    if not patterns:
        return ""

    if name and name in patterns:
        return patterns[name]

    # Return random pattern
    key = random.choice(list(patterns.keys()))
    return patterns[key]


def get_din_pattern(difficulty: str, category: str | None = None) -> tuple[str, str, str]:
    """Get a random Din optimization pattern.

    Returns (before_code, after_code, description)
    """
    import random

    patterns = DIN_PATTERNS.get(difficulty, {})
    if not patterns:
        patterns = DIN_PATTERNS["basic"]

    if category and category in patterns:
        items = patterns[category]
    else:
        # Combine all categories
        items = []
        for cat_items in patterns.values():
            items.extend(cat_items)

    if not items:
        return ("", "", "")

    return random.choice(items)


def get_farore_bug(difficulty: str, category: str | None = None) -> dict:
    """Get a random Farore debugging example.

    Returns dict with: buggy, issue, fix, explanation
    """
    import random

    bugs = FARORE_BUGS.get(difficulty, {})
    if not bugs:
        bugs = FARORE_BUGS["basic"]

    if category and category in bugs:
        items = bugs[category]
    else:
        items = []
        for cat_items in bugs.values():
            items.extend(cat_items)

    if not items:
        return {"buggy": "", "issue": "", "fix": "", "explanation": ""}

    return random.choice(items)


def get_nayru_template(difficulty: str) -> dict:
    """Get a random Nayru code generation template.

    Returns dict with: task, code
    """
    import random

    templates = NAYRU_TEMPLATES.get(difficulty, NAYRU_TEMPLATES["basic"])
    return random.choice(templates)


def get_veran_example(difficulty: str) -> dict:
    """Get a random Veran code explanation example.

    Returns dict with: code, concepts
    """
    import random

    examples = VERAN_EXAMPLES.get(difficulty, VERAN_EXAMPLES["basic"])
    return random.choice(examples)


def get_asar_context(include_examples: bool = True) -> str:
    """Get ASAR syntax context for prompts.

    Args:
        include_examples: If True, include full code examples

    Returns:
        String with ASAR syntax rules and examples
    """
    parts = [
        "## ASAR Assembler Syntax (CRITICAL)",
        "",
        "You MUST use ASAR syntax, NOT ca65/cc65 syntax.",
        "",
        "### Address Operators",
        "- `label&$FFFF` - Get low 16 bits (NOT .LOWORD)",
        "- `label>>8` - Get high byte of low word",
        "- `label>>16` - Get bank byte (NOT .BANKBYTE, NOT ^label)",
        "- `LDA.l label` - Long addressing (24-bit)",
        "",
        "### Data Directives",
        "- `db $42` - Define byte (NOT .BYTE)",
        "- `dw $1234` - Define word (NOT .WORD)",
        "- `dl $123456` - Define long",
        "",
        "### Structure",
        "- Start with `lorom` or `hirom`",
        "- Use `org $address` for placement (NOT .SEGMENT)",
        "- Local labels: `.label` or `-`/`+` for anonymous",
        "- Defines: `!MyConst = $1234`",
        "",
        "### WRONG vs CORRECT",
        "```",
        "; WRONG:                CORRECT (ASAR):",
        ".LOWORD(src)      →    src&$FFFF",
        ".BANKBYTE(src)    →    src>>16",
        "^src              →    src>>16",
        ".SEGMENT \"CODE\"  →    org $008000",
        ".BYTE $42         →    db $42",
        "```",
    ]

    if include_examples:
        parts.extend([
            "",
            "### Example DMA Setup (ASAR)",
            "```asm",
            ASAR_SYNTAX["dma_setup_correct"].strip(),
            "```",
        ])

    return "\n".join(parts)


def get_hardware_context(topics: list[str]) -> str:
    """Build hardware context string for given topics."""
    context_parts = []

    for topic in topics:
        if topic in NAYRU_HARDWARE:
            hw = NAYRU_HARDWARE[topic]
            context_parts.append(f"## {topic.replace('_', ' ').title()}")
            context_parts.append(hw.get("description", ""))

            if "addresses" in hw:
                context_parts.append("\nAddresses:")
                for name, (addr, desc) in hw["addresses"].items():
                    context_parts.append(f"  {addr}: {desc}")

            if "registers" in hw:
                context_parts.append("\nRegisters:")
                for name, (addr, desc) in hw["registers"].items():
                    context_parts.append(f"  {addr}: {name} - {desc}")

            if "example" in hw:
                context_parts.append(f"\nExample:\n```\n{hw['example'].strip()}\n```")

    return "\n".join(context_parts)
