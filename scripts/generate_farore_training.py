#!/usr/bin/env python3
"""Generate debugging training data for Farore.

Farore specializes in finding and fixing bugs in 65816 assembly code.
Training data format: ChatML with bug identification and fix explanation.
"""

import json
from pathlib import Path

# Output path
OUTPUT = Path(__file__).parent.parent / "models" / "farore_debugging_training.jsonl"

SYSTEM_PROMPT = """You are Farore, a 65816 assembly debugging expert specializing in SNES/Super Famicom code.

Your expertise includes:
- Identifying bugs in 65816 assembly code
- Explaining why code fails or behaves unexpectedly
- Providing corrected code with explanations
- Understanding common pitfalls: mode mismatches, DMA issues, addressing errors

When debugging:
1. Identify the specific bug or issue
2. Explain WHY it's wrong
3. Provide the corrected code
4. Explain the fix"""

# Debugging examples: (buggy_code, user_question, correct_analysis)
DEBUGGING_EXAMPLES = [
    # === MODE MISMATCH BUGS ===
    (
        """LDA #$1234
STA $10""",
        "This code crashes. What's wrong?",
        """**Bug: 16-bit immediate with 8-bit accumulator**

The code attempts to load a 16-bit value (#$1234) but likely has an 8-bit accumulator (M flag set). This causes the assembler to interpret $34 as the operand and $12 as the next instruction opcode.

**Fix:**
```asm
REP #$20        ; Switch to 16-bit accumulator
LDA #$1234
STA $10
SEP #$20        ; Restore 8-bit if needed
```

Always use REP #$20 before 16-bit accumulator operations."""
    ),
    (
        """REP #$20
LDA $10
AND #$FF
STA $10
SEP #$20""",
        "I want to clear the high byte but this doesn't work right.",
        """**Bug: Wrong mask for 16-bit mode**

In 16-bit mode, AND #$FF keeps the LOW byte and clears the HIGH byte. But you're storing back to $10, which in 16-bit mode writes to both $10 and $11.

If you only want to affect $10:
```asm
SEP #$20        ; 8-bit mode
LDA $10
; Already 8-bit, no masking needed
STA $10
```

Or to truly clear high byte of a 16-bit value:
```asm
REP #$20
LDA $10
AND #$00FF      ; Keep low byte, clear high byte
STA $10
SEP #$20
```

The original code IS correct for clearing high byte - verify $11 isn't being unexpectedly modified."""
    ),
    (
        """LDX #$0100
loop:
DEX
BNE loop""",
        "Loop runs wrong number of times.",
        """**Bug: 16-bit index with potentially 8-bit mode**

If X flag is set (8-bit index), LDX #$0100 loads only $00, making the loop execute 256 times (wrapping from 0 to FF, then to 0).

**Fix:**
```asm
REP #$10        ; 16-bit index registers
LDX #$0100      ; X = 256
loop:
DEX
BNE loop        ; Loops 256 times correctly
SEP #$10        ; Restore 8-bit if needed
```

Always check the X flag (bit 4 of P register) matches your intended index size."""
    ),

    # === DMA BUGS ===
    (
        """LDA #$01
STA $4300       ; DMA mode
LDA #$18
STA $4301       ; Destination: VRAM data
LDA #$00
STA $4302       ; Source low
LDA #$80
STA $4303       ; Source high
LDA #$00
STA $4305       ; Size low
LDA #$10
STA $4306       ; Size high
LDA #$01
STA $420B       ; Start DMA""",
        "DMA transfer doesn't work, nothing appears in VRAM.",
        """**Bug: Missing source bank register ($4304)**

You set $4302-$4303 (source address) but forgot $4304 (source bank). The bank defaults to 0 or garbage.

**Fix:**
```asm
LDA #$01
STA $4300       ; DMA mode: A->B, increment
LDA #$18
STA $4301       ; Destination: $2118 (VRAM data)
LDA #$00
STA $4302       ; Source address low
LDA #$80
STA $4303       ; Source address high
LDA #$7E        ; Source BANK - THIS WAS MISSING
STA $4304
LDA #$00
STA $4305       ; Size low ($1000 bytes)
LDA #$10
STA $4306       ; Size high
LDA #$01
STA $420B       ; Enable DMA channel 0
```

Also verify VRAM address is set via $2116-$2117 before the transfer."""
    ),
    (
        """LDA #$80
STA $2115       ; VRAM increment mode
REP #$20
LDA #$0000
STA $2116       ; VRAM address
SEP #$20
LDA #$01
STA $420B       ; Start DMA""",
        "VRAM DMA only transfers half the data.",
        """**Bug: Wrong DMA mode for word transfers**

$2115 = $80 means increment after writing to $2119 (high byte), but your DMA is probably targeting $2118 only.

For proper word-at-a-time VRAM DMA:

**Fix:**
```asm
LDA #$80
STA $2115       ; Increment on high byte write

; DMA setup for word transfer
LDA #$01        ; Mode: 2 registers write once (word)
STA $4300       ; This writes to $2118 then $2119
LDA #$18
STA $4301       ; Base destination $2118

; ... rest of DMA setup ...
LDA #$01
STA $420B
```

Use DMA mode $01 (2 regs write once) for VRAM word transfers, not mode $00."""
    ),
    (
        """STZ $420C       ; Disable HDMA
LDA #$02
STA $4300       ; DMA mode
LDA #$22
STA $4301       ; Destination: CGRAM
LDA.l source
STA $4302
LDA.l source+1
STA $4303
LDA.l source+2
STA $4304
LDA #$00
STA $4305
LDA #$02        ; 512 bytes
STA $4306
LDA #$01
STA $420B""",
        "CGRAM colors are wrong/garbage.",
        """**Bug: Wrong DMA mode for CGRAM**

DMA mode $02 means "write twice to same register" but CGRAM ($2122) expects sequential byte writes. Mode $02 writes the same byte twice.

**Fix:**
```asm
STZ $420C       ; Disable HDMA
STZ $2121       ; CGRAM address = 0 (important!)

LDA #$00        ; Mode 0: 1 register write once
STA $4300
LDA #$22        ; Destination: $2122 (CGRAM data)
STA $4301
; ... source setup ...
LDA #$00
STA $4305
LDA #$02        ; 512 bytes (256 colors x 2 bytes)
STA $4306
LDA #$01
STA $420B
```

Use mode $00 for CGRAM. Also set CGRAM address ($2121) before transfer!"""
    ),

    # === ADDRESSING MODE BUGS ===
    (
        """LDA $7E0010
STA $12""",
        "Value isn't being loaded correctly.",
        """**Bug: Using absolute long in bank 00 without .l suffix**

If assembled in bank $00-$3F, `LDA $7E0010` might be interpreted as absolute addressing to $0010 (mirrored), not long addressing to bank $7E.

**Fix:**
```asm
LDA.l $7E0010   ; Explicit long addressing (4 bytes)
STA $12
```

Or use direct bank setup:
```asm
PHB
LDA #$7E
PHA
PLB             ; Set data bank to $7E
LDA $0010       ; Now reads from $7E:0010
STA $12
PLB             ; Restore bank
```

The .l suffix forces 24-bit long addressing."""
    ),
    (
        """LDA ($10)
STA $20""",
        "Indirect load gives wrong value.",
        """**Bug: Missing Y index or wrong indirect mode**

`LDA ($10)` is indirect LONG in 65816 - it reads a 24-bit pointer from $10-$12. You probably want indirect with Y index.

**Fix - if you want [$10] to point to an address:**
```asm
LDA ($10),Y     ; Indirect indexed - reads 16-bit ptr from $10-$11
                ; Then adds Y and loads from (ptr+Y) in current bank
STA $20
```

**Fix - if you want true indirect (no index):**
```asm
LDA ($10)       ; 65816 indirect long - reads 24-bit from $10-$12
; Make sure $10-$12 contain valid 24-bit address
```

**Fix - for direct indirect in bank 0:**
```asm
LDY #$00
LDA ($10),Y     ; Common pattern for indirect without offset
STA $20
```"""
    ),
    (
        """loop:
    LDA $1000,X
    STA $2000,X
    INX
    CPX #$100
    BNE loop""",
        "Only copies 256 bytes, should copy more.",
        """**Bug: 8-bit index register wraps at 256**

If X is in 8-bit mode (X flag set), it wraps from $FF to $00, never reaching $100.

**Fix:**
```asm
REP #$10        ; 16-bit index registers
LDX #$0000
loop:
    LDA $1000,X
    STA $2000,X
    INX
    CPX #$0100  ; Now can compare to values > 255
    BNE loop
SEP #$10        ; Restore if needed
```

For large copies, consider using MVN/MVP block move instructions instead."""
    ),

    # === STACK BUGS ===
    (
        """PHA
PHX
JSR subroutine
PLX
PLA
RTS""",
        "Registers corrupted after JSR.",
        """**Bug: Stack imbalance in subroutine or mode mismatch**

If the subroutine uses different register sizes than the caller, PHA/PLA will push/pull wrong byte counts.

**Fix - ensure consistent modes:**
```asm
PHP             ; Save processor status (includes M/X flags)
REP #$30        ; 16-bit A and X/Y
PHA             ; Pushes 2 bytes
PHX             ; Pushes 2 bytes
JSR subroutine
PLX             ; Must be 16-bit to pull 2 bytes
PLA
PLP             ; Restore original modes
RTS
```

**Or document register sizes:**
```asm
; Assumes 8-bit A, 16-bit X
SEP #$20
REP #$10
PHA             ; 1 byte
PHX             ; 2 bytes
JSR subroutine  ; Subroutine must maintain this!
PLX
PLA
RTS
```

Subroutine must use same modes OR save/restore P register."""
    ),
    (
        """JSL LongRoutine
RTS""",
        "Game crashes after returning from long subroutine.",
        """**Bug: Using RTS instead of RTL after JSL**

JSL pushes 3 bytes (24-bit return address). RTS only pulls 2 bytes, leaving the stack corrupted.

**Fix in calling code:**
```asm
JSL LongRoutine
RTL             ; If caller is also long
; OR
JSR ShortRoutine  ; Use JSR if staying in same bank
RTS
```

**Fix in subroutine (LongRoutine):**
```asm
LongRoutine:
    ; ... code ...
    RTL         ; Return LONG - pulls 3 bytes
```

JSL/RTL are pairs. JSR/RTS are pairs. Never mix them."""
    ),
    (
        """PHB
PHK
PLB
; ... code using current bank ...
PLB
RTS""",
        "Bank register is wrong after this code.",
        """**Bug: Stack order - PLB pops in wrong order**

PHB pushes old bank, PHK pushes program bank, but PLB pulls in LIFO order - you're restoring the program bank to B, not the original!

**Fix:**
```asm
PHB             ; Save data bank (1)
PHK             ; Push program bank (2)
PLB             ; Pop (2) into B - now B = K
; ... code using current program bank as data bank ...
PLB             ; Pop (1) - restores original bank
RTS
```

Or more clearly:
```asm
PHB             ; Save original bank
LDA #$7E        ; Load desired bank
PHA
PLB             ; Set data bank to $7E
; ... code ...
PLB             ; Restore original (from first PHB)
RTS
```

Stack is LIFO - last pushed is first popped."""
    ),

    # === INTERRUPT BUGS ===
    (
        """LDA #$81
STA $4200       ; Enable NMI and joypad
; ... rest of init ...
CLI""",
        "NMI handler never runs.",
        """**Bug: NMI is controlled by $4200, not the I flag**

CLI clears the Interrupt disable flag, which controls IRQ, not NMI. NMI fires regardless of I flag if enabled in $4200.

Common causes of NMI not firing:

**1. VBlank hasn't started yet:**
```asm
LDA #$81
STA $4200       ; Enable NMI
; Wait for first vblank
-   LDA $4212
    BPL -       ; Wait for vblank flag
```

**2. NMI vector not set correctly:**
Check your vector table at $00:FFEA (native) or $00:FFFA (emulation).

**3. Force blank is on:**
```asm
LDA #$0F        ; Full brightness, no force blank
STA $2100       ; NMI fires during vblank even with this
```

**4. Reading $4210 clears NMI flag:**
```asm
NMI:
    LDA $4210   ; Acknowledge NMI (required!)
    ; ... handler code ...
    RTI
```"""
    ),
    (
        """IRQ:
    PHA
    LDA $4211       ; Acknowledge IRQ
    ; do stuff
    PLA
    RTI""",
        "IRQ handler corrupts registers.",
        """**Bug: Not saving all registers or status**

IRQ can fire at any time. You must preserve ALL registers including processor status.

**Fix:**
```asm
IRQ:
    REP #$30        ; Ensure consistent mode for saves
    PHA
    PHX
    PHY
    PHB
    PHD

    SEP #$20        ; 8-bit for register access
    LDA $4211       ; Acknowledge IRQ

    ; ... handler code ...
    ; Restore modes before your code if needed

    REP #$30        ; Match the mode used for push
    PLD
    PLB
    PLY
    PLX
    PLA
    RTI

; Note: 65816 RTI restores P register automatically,
; but you must match push/pull sizes!
```

Alternative - save P first:
```asm
IRQ:
    PHP             ; Save status FIRST
    REP #$30
    PHA
    PHX
    PHY
    ; ...
    PLY
    PLX
    PLA
    PLP             ; Restore status
    RTI
```"""
    ),

    # === BRANCH BUGS ===
    (
        """    LDA $10
    BEQ far_target
    ; ... lots of code ...
far_target:
    RTS""",
        "Assembler error: branch out of range.",
        """**Bug: Branch target too far**

Branch instructions (BEQ, BNE, BCC, etc.) use 8-bit signed offset: -128 to +127 bytes.

**Fix - use long branch:**
```asm
    LDA $10
    BNE +           ; Invert condition, skip the jump
    JMP far_target  ; JMP has no range limit
+
    ; ... lots of code ...
far_target:
    RTS
```

**Or use BRL (Branch Relative Long) in 65816:**
```asm
    LDA $10
    BNE +
    BRL far_target  ; 16-bit offset: -32768 to +32767
+
    ; ... lots of code ...
far_target:
    RTS
```

BRL is preferred over JMP when you want relocatable code."""
    ),
    (
        """loop:
    DEX
    BNE loop
    DEY
    BNE loop""",
        "Outer loop runs wrong number of times.",
        """**Bug: Both branches go to same target**

When inner loop ends (X=0), you DEY and branch to `loop`, but `loop` starts with DEX, not reloading X.

**Fix:**
```asm
outer:
    LDX #$10        ; Reset inner counter each outer iteration
inner:
    DEX
    BNE inner       ; Inner loop on X
    DEY
    BNE outer       ; Outer loop on Y - goes back to LDX
```

Or using nested structure:
```asm
    LDY #$05        ; Outer count
outer:
    LDX #$10        ; Inner count (reset each time)
inner:
    ; ... inner loop body ...
    DEX
    BNE inner
    ; ... outer loop body ...
    DEY
    BNE outer
```"""
    ),

    # === COMPARISON BUGS ===
    (
        """LDA $10
CMP #$80
BCC less_than""",
        "Comparison doesn't work for negative numbers.",
        """**Bug: Using unsigned comparison for signed values**

CMP does unsigned comparison. $80-$FF are treated as 128-255, not -128 to -1.

**Fix - for signed comparison:**
```asm
LDA $10
SEC
SBC #$80        ; Subtract comparison value
BMI less_than   ; Branch if result is negative (N flag)
```

**Or use EOR trick for signed compare to specific value:**
```asm
LDA $10
EOR #$80        ; Flip sign bit
CMP #$80        ; Compare (now unsigned works for signed)
BCC less_than
```

**Or compare to zero (common case):**
```asm
LDA $10
BMI negative    ; N flag set if bit 7 is 1 (negative)
BPL positive
```"""
    ),
    (
        """REP #$20
LDA $10
CMP #$1000
BEQ equal""",
        "16-bit compare never matches.",
        """**Bug: Possibly comparing wrong bytes**

In 16-bit mode, LDA $10 loads from $10 AND $11. Make sure both bytes are what you expect.

**Debug steps:**

1. Verify 16-bit mode is active:
```asm
REP #$20        ; Clear M flag
.A16            ; Tell assembler (if using asar)
LDA $10         ; Loads $10 (low) and $11 (high)
CMP #$1000
BEQ equal
```

2. Check byte order (little-endian):
- $10 should contain $00 (low byte of $1000)
- $11 should contain $10 (high byte of $1000)

3. Common mistake - storing in wrong order:
```asm
; WRONG:
LDA #$10
STA $10
LDA #$00
STA $11
; This stores $0010, not $1000!

; RIGHT:
REP #$20
LDA #$1000
STA $10         ; Stores $00 to $10, $10 to $11
SEP #$20
```"""
    ),

    # === REGISTER PRESERVATION BUGS ===
    (
        """LDA #$01
STA $4016       ; Strobe joypad
LDA $4016       ; Read bit
AND #$01
ORA $10         ; Combine with existing
STA $10""",
        "Joypad reading only gets first button.",
        """**Bug: Missing latch clear and loop**

Joypad reading requires: strobe high, strobe low, then read 8 times.

**Fix:**
```asm
    LDA #$01
    STA $4016       ; Strobe high
    STZ $4016       ; Strobe low - latch data

    LDX #$08        ; 8 buttons
    STZ $10         ; Clear result
-   LDA $4016       ; Read one bit
    LSR             ; Shift bit 0 into carry
    ROL $10         ; Roll carry into result
    DEX
    BNE -

; $10 now has: AXLR----UDLR (standard order)
```

Each read of $4016 returns one button bit in bit 0. You need to read 8 times (or 16 for extended controllers)."""
    ),
    (
        """LDA $10
ASL
ASL
ASL             ; Multiply by 8
TAX
LDA table,X
STA $20""",
        "Wrong table entry is loaded.",
        """**Bug: Table index might exceed 8-bit range or mode mismatch**

If $10 can be >= 32, then $10 * 8 >= 256, overflowing 8-bit A.

**Fix for 8-bit safe range:**
```asm
SEP #$20
REP #$10        ; 16-bit index!
LDA $10
AND #$1F        ; Limit to 0-31 (result 0-248)
ASL
ASL
ASL
TAX             ; X needs to be 16-bit for values > 255
LDA table,X
STA $20
```

**Fix for larger ranges:**
```asm
REP #$30        ; 16-bit A and X
LDA $10
AND #$00FF      ; Ensure clean 16-bit value
ASL
ASL
ASL             ; Now can handle up to 8191
TAX
SEP #$20        ; 8-bit for table load if entries are bytes
LDA table,X
STA $20
```"""
    ),

    # === DIRECT PAGE BUGS ===
    (
        """LDA #$00
STA $00
LDA #$7E
STA $01
; Use $00-$01 as pointer
LDA ($00)""",
        "Indirect load gets wrong data.",
        """**Bug: Direct page might not be at $0000**

65816 has a moveable direct page. LDA ($00) uses D register + $00, not absolute $0000.

**Fix - ensure direct page is correct:**
```asm
PHD             ; Save current DP
LDA #$0000
TCD             ; Set DP to $0000

LDA #$00
STA $00
LDA #$7E
STA $01
LDA ($00)       ; Now definitely uses $0000-$0001

PLD             ; Restore DP
```

**Or use absolute addressing for the pointer setup:**
```asm
LDA #$00
STA.w $0000     ; .w forces absolute, ignores DP
LDA #$7E
STA.w $0001
; Then set DP if needed for indirect
LDA #$0000
TCD
LDA ($00)
```"""
    ),
    (
        """LDA #$10
TCD             ; Direct page at $0010
LDA $00         ; Should be $10 in RAM
STA $02         ; Should be $12 in RAM""",
        "Writes going to wrong addresses.",
        """**Bug: DP offset confusion**

When D=$0010, LDA $00 accesses $0010, LDA $02 accesses $0012. This is correct!

But if you ALSO access $10-$12 thinking they're separate:
```asm
LDA $10         ; This is $0010 + $10 = $0020, not $0010!
```

**Common confusion:**
```asm
; With D=$0010:
LDA $00   ; Reads $0010 - "DP zero"
LDA $10   ; Reads $0020 - NOT $0010!
LDA $100  ; Reads $0110 (DP + $100) if <$200, else absolute
```

**Fix - be explicit about what you want:**
```asm
LDA #$0010
TCD
LDA $00         ; Reads RAM $0010
LDA.w $0010     ; Reads RAM $0010 (absolute, ignores DP)
LDA $10         ; Reads RAM $0020!
```

With DP offset, all DP references shift by that amount."""
    ),

    # === HARDWARE TIMING BUGS ===
    (
        """LDA #$0F
STA $2100       ; Screen on
LDA #$80
STA $2115
REP #$20
LDA #$1000
STA $2116       ; VRAM address
LDA.l source
STA $2118       ; Write VRAM""",
        "VRAM writes are corrupted or wrong location.",
        """**Bug: Writing VRAM outside of VBlank/Force Blank**

VRAM can only be written during VBlank or Force Blank ($2100 bit 7 set). Writing during active display corrupts data.

**Fix:**
```asm
; Option 1: Use force blank
LDA #$80
STA $2100       ; Force blank ON (bit 7)

LDA #$80
STA $2115
REP #$20
LDA #$1000
STA $2116
LDA.l source
STA $2118       ; Safe - screen is blanked

SEP #$20
LDA #$0F
STA $2100       ; Screen back on

; Option 2: Wait for VBlank
LDA #$0F
STA $2100       ; Screen on
-   LDA $4212
    AND #$80
    BEQ -       ; Wait for vblank (bit 7 of $4212)

; Now safe to write VRAM briefly
```

Always use force blank for large VRAM updates."""
    ),
    (
        """LDA #$01
STA $212C       ; Enable BG1
STA $2105       ; Mode 1
LDA #$00
STA $2107       ; BG1 tilemap""",
        "Background doesn't appear.",
        """**Bug: Missing BG setup and/or VRAM data**

Background display requires multiple steps. Common issues:

**Fix - complete BG setup:**
```asm
; 1. Set BG mode
LDA #$01
STA $2105       ; Mode 1 (8x8 tiles)

; 2. Set tilemap address (in VRAM words)
LDA #$00        ; VRAM $0000 (bits 2-6 = addr/1024, >> 1)
STA $2107       ; BG1 tilemap at $0000

; 3. Set tile data address
LDA #$01        ; VRAM $1000 (bits 0-3 = addr/4096, >> 1)
STA $210B       ; BG1/BG2 tile data

; 4. Enable on main screen
LDA #$01
STA $212C       ; BG1 on main screen

; 5. Make sure screen is on
LDA #$0F
STA $2100       ; Full brightness

; 6. Don't forget: you need actual tile and tilemap
;    data uploaded to VRAM at those addresses!
```

Most common issue: no tilemap/tile data in VRAM."""
    ),

    # === MVN/MVP BUGS ===
    (
        """REP #$30
LDX #source
LDY #dest
LDA #$0100
MVN $7E,$7E""",
        "Block move copies wrong amount.",
        """**Bug: MVN length is off-by-one**

MVN/MVP copy (A+1) bytes, not A bytes. LDA #$0100 copies 257 bytes!

**Fix:**
```asm
REP #$30
LDX #source     ; Source address (16-bit)
LDY #dest       ; Dest address (16-bit)
LDA #$00FF      ; Copy $100 (256) bytes: length-1!
MVN $7E,$7E     ; MVN destbank, srcbank

; After MVN:
; X = source + count
; Y = dest + count
; A = $FFFF
; DB = dest bank
```

Also note the operand order is DESTINATION, SOURCE (opposite of memcpy)."""
    ),
    (
        """REP #$30
LDX #$1000
LDY #$1010
LDA #$00FF
MVN $7E,$7E     ; Copy 256 bytes forward""",
        "Overlapping copy corrupts data.",
        """**Bug: Using MVN for overlapping regions where dest > source**

MVN copies forward (low to high). For overlapping regions where dest > source, use MVP.

**Fix:**
```asm
; Source: $1000-$10FF
; Dest:   $1010-$110F (overlaps!)

; Use MVP (moves backward, high to low)
REP #$30
LDX #$10FF      ; Source END
LDY #$110F      ; Dest END
LDA #$00FF      ; 256 bytes (minus 1)
MVP $7E,$7E     ; Copies backward, safe for this overlap

; MVP decrements X and Y
; For overlap where dest < source, MVN is fine
```

Rule: dest > source = MVP, dest < source = MVN, no overlap = either."""
    ),

    # === ACCUMULATOR SIZE BUGS ===
    (
        """REP #$20
LDA #$1234
SEP #$20
PHA
REP #$20
PLA
STA $10""",
        "Only low byte saved/restored correctly.",
        """**Bug: PHA/PLA size depends on M flag at time of execution**

In 8-bit mode, PHA pushes 1 byte. In 16-bit mode, PLA pulls 2 bytes (garbage in high byte).

**Fix - consistent sizing:**
```asm
REP #$20
LDA #$1234
PHA             ; Push 2 bytes (16-bit)
; ... other code ...
REP #$20        ; MUST be 16-bit to pull 2 bytes
PLA
STA $10
```

**Or save/restore with explicit size:**
```asm
REP #$20
LDA #$1234
STA $10         ; Just store to memory instead
; ... other code ...
REP #$20
LDA $10         ; Reload
```

**Or use XBA for byte swapping:**
```asm
REP #$20
LDA #$1234
XBA             ; A = $3412
SEP #$20
PHA             ; Push $12 (now low byte)
XBA             ; A = $xx34
PHA             ; Push $34
; ... later ...
PLA             ; $34
XBA
PLA             ; $12
XBA             ; A = $1234 (in 16-bit mode)
```"""
    ),

    # === ZERO FLAG BUGS ===
    (
        """LDA $10
AND #$0F
BEQ zero
LDA #$01
STA $20
zero:
RTS""",
        "Code after BEQ always runs, even when result is zero.",
        """**Bug: No bug shown - verify your code flow**

Your code looks correct. If AND #$0F produces 0, BEQ branches to `zero`, skipping the STA.

**Potential issues:**

1. **Hidden .A8/.A16 mismatch:**
```asm
REP #$20        ; 16-bit
LDA $10         ; Loads $10 AND $11
AND #$0F        ; Only affects low byte, high byte may be non-zero!
BEQ zero        ; Won't branch if $11 != 0
```

2. **Check what's at `zero` label:**
```asm
zero:           ; Falls through here
    RTS         ; If $20 isn't touched, that's expected

; Maybe you meant:
    LDA $10
    AND #$0F
    BNE not_zero    ; Branch if NOT zero
    RTS             ; Return early if zero
not_zero:
    LDA #$01
    STA $20
    RTS
```

3. **Use .A8 directive for assembler:**
```asm
SEP #$20
.A8             ; Tell asar we're 8-bit
LDA $10
AND #$0F
BEQ zero
```"""
    ),
]

def generate_training_data():
    """Generate ChatML formatted training data."""
    examples = []

    for buggy_code, question, analysis in DEBUGGING_EXAMPLES:
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"```asm\n{buggy_code}\n```\n\n{question}"},
                {"role": "assistant", "content": analysis}
            ]
        }
        examples.append(example)

    return examples


def main():
    examples = generate_training_data()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} debugging examples")
    print(f"Output: {OUTPUT}")

    # Category breakdown
    categories = {
        "Mode mismatch": 3,
        "DMA issues": 3,
        "Addressing bugs": 3,
        "Stack bugs": 3,
        "Interrupt bugs": 2,
        "Branch bugs": 2,
        "Comparison bugs": 2,
        "Register preservation": 2,
        "Direct page bugs": 2,
        "Hardware timing": 2,
        "MVN/MVP bugs": 2,
        "Accumulator size": 1,
        "Zero flag": 1,
    }
    print("\nCategories:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
