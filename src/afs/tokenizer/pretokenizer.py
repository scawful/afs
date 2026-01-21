"""Pre-tokenizer for 65816 assembly.

Splits assembly code into semantic units before vocabulary lookup.
Preserves:
- Opcodes as single tokens
- Addresses with their prefixes ($xxxx, #$xx)
- Labels with colons
- Comments as units
- Indexed addressing (,X ,Y)
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .vocab import DIRECTIVES, OPCODES


@dataclass
class Token:
    """A pre-tokenized token with metadata."""
    text: str
    type: str  # opcode, address, label, comment, register, directive, literal, punct, unknown
    start: int  # Character offset in original text
    end: int


class AssemblyPreTokenizer:
    """Pre-tokenizer that splits assembly into semantic units."""

    # Regex patterns for different token types (order matters!)
    PATTERNS = [
        # Comments (capture entire comment including ;)
        (r';[^\n]*', 'comment'),

        # Labels (identifier followed by :)
        (r'[A-Za-z_][A-Za-z0-9_]*:', 'label'),

        # Local labels (.identifier or @identifier)
        (r'[.@][A-Za-z_][A-Za-z0-9_]*:?', 'label'),

        # Address labels with bank prefix (#_BBAAAA:)
        (r'#_[0-9A-Fa-f]{6}:', 'label'),

        # Immediate hex (#$xx, #$xxxx, #$xxxxxx)
        (r'#\$[0-9A-Fa-f]+', 'immediate'),

        # Immediate decimal (#123)
        (r'#[0-9]+', 'immediate'),

        # Long address ($xxxxxx)
        (r'\$[0-9A-Fa-f]{6}', 'address_long'),

        # Absolute address ($xxxx)
        (r'\$[0-9A-Fa-f]{4}', 'address_abs'),

        # Direct page address ($xx)
        (r'\$[0-9A-Fa-f]{2}(?![0-9A-Fa-f])', 'address_dp'),

        # Generic hex (for data bytes)
        (r'\$[0-9A-Fa-f]+', 'hex'),

        # Binary literal (%01010101)
        (r'%[01]+', 'binary'),

        # Decimal number
        (r'\b[0-9]+\b', 'decimal'),

        # Indexed addressing (must capture ,X ,Y ,S as unit)
        (r',\s*[XYSxys](?![A-Za-z0-9_])', 'index'),

        # Opcode with suffix (LDA.b, STA.w, etc.)
        (r'\b(' + '|'.join(OPCODES) + r')\.[bwlBWL]\b', 'opcode'),

        # Plain opcode
        (r'\b(' + '|'.join(OPCODES) + r')\b', 'opcode'),

        # Directive
        (r'\.?(' + '|'.join(d.lstrip('.') for d in DIRECTIVES) + r')\b', 'directive'),

        # Identifier (for labels, macro names, etc.)
        (r'[A-Za-z_][A-Za-z0-9_]*', 'identifier'),

        # Brackets for indirect addressing
        (r'[\(\)\[\]]', 'bracket'),

        # Punctuation
        (r'[,:;]', 'punct'),

        # Operators
        (r'[+\-*/=<>|&^~!]', 'operator'),

        # Whitespace (preserve for structure)
        (r'[ \t]+', 'whitespace'),

        # Newline
        (r'\n', 'newline'),

        # Anything else
        (r'.', 'unknown'),
    ]

    def __init__(self, preserve_whitespace: bool = False):
        """Initialize pre-tokenizer.

        Args:
            preserve_whitespace: If True, include whitespace tokens.
        """
        self.preserve_whitespace = preserve_whitespace

        # Compile combined pattern
        pattern_parts = [f'(?P<{name}_{i}>{pattern})'
                        for i, (pattern, name) in enumerate(self.PATTERNS)]
        self.combined_pattern = re.compile('|'.join(pattern_parts), re.IGNORECASE)

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize assembly text into semantic units.

        Args:
            text: Assembly source code.

        Returns:
            List of Token objects.
        """
        tokens = []
        pos = 0

        for match in self.combined_pattern.finditer(text):
            # Find which pattern matched
            for name, value in match.groupdict().items():
                if value is not None:
                    # Extract type from pattern name (e.g., "opcode_5" -> "opcode")
                    token_type = name.rsplit('_', 1)[0]

                    # Skip whitespace if not preserving
                    if token_type == 'whitespace' and not self.preserve_whitespace:
                        continue

                    tokens.append(Token(
                        text=value,
                        type=token_type,
                        start=match.start(),
                        end=match.end(),
                    ))
                    break

        return tokens

    def tokenize_line(self, line: str) -> list[Token]:
        """Tokenize a single line of assembly.

        Convenience method that handles a single line.
        """
        return self.tokenize(line)

    def iter_tokens(self, text: str) -> Iterator[Token]:
        """Iterate over tokens without building full list."""
        for match in self.combined_pattern.finditer(text):
            for name, value in match.groupdict().items():
                if value is not None:
                    token_type = name.rsplit('_', 1)[0]
                    if token_type == 'whitespace' and not self.preserve_whitespace:
                        continue
                    yield Token(
                        text=value,
                        type=token_type,
                        start=match.start(),
                        end=match.end(),
                    )
                    break


def normalize_token(token: Token) -> str:
    """Normalize a token for vocabulary lookup.

    - Opcodes: uppercase (LDA, STA)
    - Addresses: preserve structure, normalize case ($7F00 -> $7F00)
    - Labels: keep as-is for BPE
    - Comments: optionally truncate or mask

    Args:
        token: Token to normalize.

    Returns:
        Normalized token string.
    """
    if token.type == 'opcode':
        return token.text.upper()

    if token.type in ('address_long', 'address_abs', 'address_dp', 'hex', 'immediate'):
        # Uppercase hex digits
        return token.text.upper()

    if token.type == 'index':
        # Normalize to uppercase, remove space
        return ',' + token.text.replace(',', '').strip().upper()

    if token.type == 'bracket':
        return token.text

    return token.text


def split_address(token: Token) -> list[str]:
    """Split an address token into prefix + digits for fine-grained encoding.

    Useful if you want the model to learn digit patterns.

    Examples:
        $7F00 -> ['$', '7', 'F', '0', '0']
        #$0F -> ['#', '$', '0', 'F']
    """
    text = token.text
    parts = []

    i = 0
    while i < len(text):
        if text[i] in '#$%':
            parts.append(text[i])
            i += 1
        elif text[i].isalnum():
            parts.append(text[i].upper())
            i += 1
        else:
            parts.append(text[i])
            i += 1

    return parts
