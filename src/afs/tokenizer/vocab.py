"""65816 Assembly vocabulary definitions.

Comprehensive vocabulary for SNES/65816 assembly including:
- All valid opcodes
- Addressing mode markers
- Registers
- Assembler directives (asar-compatible)
- Special tokens
"""

# =============================================================================
# 65816 Opcodes (all valid mnemonics)
# =============================================================================

OPCODES = {
    # Load/Store
    "LDA", "LDX", "LDY", "STA", "STX", "STY", "STZ",
    # Transfer
    "TAX", "TAY", "TXA", "TYA", "TXS", "TSX", "TCD", "TDC", "TCS", "TSC", "XBA",
    # Stack
    "PHA", "PHX", "PHY", "PHP", "PHB", "PHD", "PHK",
    "PLA", "PLX", "PLY", "PLP", "PLB", "PLD",
    "PEA", "PEI", "PER",
    # Arithmetic
    "ADC", "SBC", "INC", "INX", "INY", "DEC", "DEX", "DEY",
    # Logical
    "AND", "ORA", "EOR", "BIT",
    # Shift/Rotate
    "ASL", "LSR", "ROL", "ROR",
    # Compare
    "CMP", "CPX", "CPY",
    # Branch
    "BRA", "BEQ", "BNE", "BCC", "BCS", "BMI", "BPL", "BVC", "BVS", "BRL",
    # Jump/Call
    "JMP", "JML", "JSR", "JSL", "RTS", "RTL", "RTI",
    # Flag
    "SEC", "CLC", "SEI", "CLI", "SED", "CLD", "CLV",
    "SEP", "REP", "XCE",
    # Misc
    "NOP", "WDM", "STP", "WAI", "BRK", "COP",
    # Block Move
    "MVN", "MVP",
}

# Opcode size suffixes (asar/wla-dx style)
OPCODE_SUFFIXES = {".b", ".w", ".l"}

# =============================================================================
# Registers
# =============================================================================

REGISTERS = {
    "A",    # Accumulator
    "X",    # X index
    "Y",    # Y index
    "S",    # Stack pointer
    "P",    # Processor status
    "DB",   # Data bank
    "DP",   # Direct page
    "PB",   # Program bank
}

# =============================================================================
# Assembler Directives (asar-compatible)
# =============================================================================

DIRECTIVES = {
    # Data definition
    "db", "dw", "dl", "dd",
    "byte", "word", "long",
    ".db", ".dw", ".dl", ".dd",

    # Origin/positioning
    "org", "base", "skip", "align",
    ".org", ".base",

    # Sections
    "freecode", "freedata", "prot", "autoclean",

    # Includes
    "incsrc", "incbin", "include",

    # Macros
    "macro", "endmacro", "%",

    # Conditionals
    "if", "else", "elseif", "endif",
    "ifdef", "ifndef",

    # Labels/Symbols
    "pushpc", "pullpc",

    # Tables
    "table", "cleartable",

    # Print/Assert
    "print", "assert", "error", "warn",

    # Math functions
    "read1", "read2", "read3", "read4",
    "canread", "canread1", "canread2", "canread3", "canread4",

    # Arch
    "arch", "65816", "spc700",
}

# =============================================================================
# Addressing Mode Tokens
# =============================================================================

# These are tokenized as units to preserve addressing semantics
ADDR_TOKENS = {
    "#",      # Immediate prefix
    ",X",     # X indexed
    ",x",
    ",Y",     # Y indexed
    ",y",
    ",S",     # Stack relative
    ",s",
    "(",      # Indirect open
    ")",      # Indirect close
    "[",      # Long indirect open
    "]",      # Long indirect close
}

# =============================================================================
# Special Tokens
# =============================================================================

SPECIAL_TOKENS = {
    # Standard transformer tokens
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,

    # Assembly-specific
    "[LABEL]": 5,      # Label placeholder
    "[COMMENT]": 6,    # Comment marker
    "[NEWLINE]": 7,    # Line break
    "[INDENT]": 8,     # Indentation
    "[HEX]": 9,        # Hex literal marker
    "[DEC]": 10,       # Decimal literal marker
    "[BIN]": 11,       # Binary literal marker
}

# =============================================================================
# Build Complete Vocabulary
# =============================================================================

def build_base_vocab() -> dict[str, int]:
    """Build the base vocabulary with fixed token IDs.

    Returns:
        Dictionary mapping tokens to IDs.
    """
    vocab = {}
    idx = 0

    # Special tokens first (fixed positions)
    for token, token_id in SPECIAL_TOKENS.items():
        vocab[token] = token_id
        idx = max(idx, token_id + 1)

    # Opcodes (uppercase)
    for op in sorted(OPCODES):
        vocab[op] = idx
        idx += 1
        # Also add with suffixes
        for suffix in OPCODE_SUFFIXES:
            vocab[op + suffix] = idx
            idx += 1

    # Registers
    for reg in sorted(REGISTERS):
        vocab[reg] = idx
        idx += 1

    # Directives
    for directive in sorted(DIRECTIVES):
        vocab[directive] = idx
        idx += 1

    # Addressing tokens
    for addr_tok in sorted(ADDR_TOKENS):
        if addr_tok not in vocab:
            vocab[addr_tok] = idx
            idx += 1

    # Common punctuation
    for punct in [":", ";", ",", "$", "%", ".", "+", "-", "*", "/", "=", "<", ">"]:
        if punct not in vocab:
            vocab[punct] = idx
            idx += 1

    # Hex digits (for composing addresses)
    for digit in "0123456789ABCDEFabcdef":
        if digit not in vocab:
            vocab[digit] = idx
            idx += 1

    return vocab


# Pre-built vocabulary
BASE_VOCAB = build_base_vocab()
VOCAB_SIZE = len(BASE_VOCAB)

# Reverse mapping
ID_TO_TOKEN = {v: k for k, v in BASE_VOCAB.items()}
