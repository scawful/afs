"""Entity extraction from 65816 assembly code.

Extracts ALTTP memory addresses, labels, and other entities from assembly
code and links them to the knowledge base in alttp_addresses.py.

Used to populate the kg_entities field in TrainingSample for knowledge-aware
training and evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .alttp_addresses import (
    ALTTP_ADDRESSES,
    AddressCategory,
    AddressInfo,
    lookup_by_address,
)


@dataclass
class ExtractedEntity:
    """An entity extracted from assembly code."""

    address: str  # Raw address string, e.g., "$7EF36C", "$0D00", "$22"
    name: str  # Resolved name from knowledge base, or generated name
    category: str  # AddressCategory value or "unknown"
    confidence: float  # 0.0-1.0, higher = more certain about mapping
    line_number: int  # 1-indexed line where entity was found
    context: str = ""  # Surrounding code context
    is_known: bool = False  # True if matched in knowledge base
    info: AddressInfo | None = None  # Full AddressInfo if matched


@dataclass
class ExtractionResult:
    """Result of entity extraction from code."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # Code labels found
    unknown_addresses: list[str] = field(default_factory=list)  # Addresses not in KB
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def known_count(self) -> int:
        """Count of entities matched in knowledge base."""
        return sum(1 for e in self.entities if e.is_known)

    @property
    def coverage(self) -> float:
        """Ratio of known entities to total."""
        if not self.entities:
            return 0.0
        return self.known_count / len(self.entities)

    def entity_names(self) -> list[str]:
        """Get list of entity names for kg_entities field."""
        return [e.name for e in self.entities if e.is_known]


class EntityExtractor:
    """Extract ALTTP entities from 65816 assembly code."""

    # Address patterns in assembly code
    # Long address: $XXXXXX (6 hex digits with bank)
    # Absolute address: $XXXX (4 hex digits)
    # Direct page: $XX (2 hex digits)
    # Also handles indexed: $XXXX,X  $XXXX,Y  ($XX),Y  etc.
    ADDRESS_PATTERN = re.compile(
        r"\$([0-9A-Fa-f]{2,6})"  # $ followed by 2-6 hex digits
        r"(?:\s*[,+]\s*[XYS])?"  # Optional indexed addressing
        r"(?:\s*\))?",  # Optional closing paren for indirect
        re.IGNORECASE,
    )

    # Label definition pattern: Label: or Label_Name:
    LABEL_PATTERN = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*:",
        re.MULTILINE,
    )

    # Common SNES hardware registers
    HARDWARE_REGISTERS = {
        "2100": ("INIDISP", AddressCategory.HARDWARE),
        "2101": ("OBSEL", AddressCategory.HARDWARE),
        "2102": ("OAMADDL", AddressCategory.HARDWARE),
        "2103": ("OAMADDH", AddressCategory.HARDWARE),
        "2104": ("OAMDATA", AddressCategory.HARDWARE),
        "2105": ("BGMODE", AddressCategory.HARDWARE),
        "2106": ("MOSAIC", AddressCategory.HARDWARE),
        "210D": ("BG1HOFS", AddressCategory.HARDWARE),
        "210E": ("BG1VOFS", AddressCategory.HARDWARE),
        "2115": ("VMAIN", AddressCategory.HARDWARE),
        "2116": ("VMADDL", AddressCategory.HARDWARE),
        "2117": ("VMADDH", AddressCategory.HARDWARE),
        "2118": ("VMDATAL", AddressCategory.HARDWARE),
        "2119": ("VMDATAH", AddressCategory.HARDWARE),
        "2121": ("CGADD", AddressCategory.HARDWARE),
        "2122": ("CGDATA", AddressCategory.HARDWARE),
        "2140": ("APUIO0", AddressCategory.SOUND),
        "2141": ("APUIO1", AddressCategory.SOUND),
        "2142": ("APUIO2", AddressCategory.SOUND),
        "2143": ("APUIO3", AddressCategory.SOUND),
        "4200": ("NMITIMEN", AddressCategory.HARDWARE),
        "4210": ("RDNMI", AddressCategory.HARDWARE),
        "4211": ("TIMEUP", AddressCategory.HARDWARE),
        "4212": ("HVBJOY", AddressCategory.HARDWARE),
        "4218": ("JOY1L", AddressCategory.HARDWARE),
        "4219": ("JOY1H", AddressCategory.HARDWARE),
        "420B": ("MDMAEN", AddressCategory.HARDWARE),
        "420C": ("HDMAEN", AddressCategory.HARDWARE),
    }

    def __init__(self, include_hardware: bool = True):
        """Initialize extractor.

        Args:
            include_hardware: Whether to include hardware register matches
        """
        self.include_hardware = include_hardware
        self._build_address_lookup()

    def _build_address_lookup(self) -> None:
        """Build reverse lookup table from addresses to info."""
        self._address_to_info: dict[tuple[str, int], list[tuple[str, AddressInfo]]] = {}

        for name, info in ALTTP_ADDRESSES.items():
            bank = info.bank.upper()

            if isinstance(info.address, int):
                key = (bank, info.address)
                if key not in self._address_to_info:
                    self._address_to_info[key] = []
                self._address_to_info[key].append((name, info))

    def extract(self, code: str) -> ExtractionResult:
        """Extract all entities from assembly code.

        Args:
            code: Assembly code text

        Returns:
            ExtractionResult with entities, labels, and stats
        """
        result = ExtractionResult()
        lines = code.split("\n")

        # Extract labels
        for match in self.LABEL_PATTERN.finditer(code):
            label = match.group(1)
            if label not in result.labels:
                result.labels.append(label)

        # Extract addresses
        seen_addresses: set[str] = set()

        for line_num, line in enumerate(lines, start=1):
            # Skip comment-only lines
            stripped = line.strip()
            if stripped.startswith(";") or not stripped:
                continue

            for match in self.ADDRESS_PATTERN.finditer(line):
                addr_hex = match.group(1).upper()

                # Skip if we've seen this exact address string
                if addr_hex in seen_addresses:
                    continue
                seen_addresses.add(addr_hex)

                entity = self._resolve_address(addr_hex, line_num, line)
                if entity:
                    result.entities.append(entity)
                    if not entity.is_known:
                        addr_str = f"${addr_hex}"
                        if addr_str not in result.unknown_addresses:
                            result.unknown_addresses.append(addr_str)

        # Compute stats
        result.stats = {
            "total_entities": len(result.entities),
            "known_entities": result.known_count,
            "unknown_addresses": len(result.unknown_addresses),
            "labels": len(result.labels),
            "coverage": result.coverage,
        }

        return result

    def _resolve_address(
        self, addr_hex: str, line_num: int, context: str
    ) -> ExtractedEntity | None:
        """Resolve an address hex string to an entity.

        Args:
            addr_hex: Hex address without $ prefix (e.g., "7EF36C", "0D00", "22")
            line_num: Line number where found
            context: Line of code for context

        Returns:
            ExtractedEntity or None if not resolvable
        """
        addr_hex = addr_hex.upper()
        addr_len = len(addr_hex)

        # Parse address based on length
        if addr_len == 6:
            # Long address: BBXXXX (bank + offset)
            bank = addr_hex[:2]
            offset = int(addr_hex[2:], 16)
        elif addr_len == 4:
            # Absolute address: assume bank 7E for WRAM range, 00 otherwise
            offset = int(addr_hex, 16)
            # WRAM mirror at 0000-1FFF is bank 7E
            if offset < 0x2000:
                bank = "7E"
            elif 0x2100 <= offset <= 0x21FF or 0x4200 <= offset <= 0x44FF:
                # Hardware registers
                bank = "00"
            else:
                bank = "7E"  # Default assumption for ALTTP
        elif addr_len == 2:
            # Direct page address
            offset = int(addr_hex, 16)
            bank = "7E"  # Direct page is typically WRAM
        else:
            return None

        # Try to find in knowledge base
        matches = lookup_by_address(offset, bank)

        if matches:
            # Use first match (most specific)
            name, info = matches[0]
            return ExtractedEntity(
                address=f"${addr_hex}",
                name=name,
                category=info.category.value,
                confidence=1.0 if len(matches) == 1 else 0.8,
                line_number=line_num,
                context=context.strip(),
                is_known=True,
                info=info,
            )

        # Check hardware registers
        if self.include_hardware and addr_hex in self.HARDWARE_REGISTERS:
            name, category = self.HARDWARE_REGISTERS[addr_hex]
            return ExtractedEntity(
                address=f"${addr_hex}",
                name=name,
                category=category.value,
                confidence=1.0,
                line_number=line_num,
                context=context.strip(),
                is_known=True,
                info=None,
            )

        # Check 4-digit variant for hardware
        if self.include_hardware and addr_len == 4:
            short_addr = addr_hex
            if short_addr in self.HARDWARE_REGISTERS:
                name, category = self.HARDWARE_REGISTERS[short_addr]
                return ExtractedEntity(
                    address=f"${addr_hex}",
                    name=name,
                    category=category.value,
                    confidence=1.0,
                    line_number=line_num,
                    context=context.strip(),
                    is_known=True,
                    info=None,
                )

        # Unknown address - still track it
        return ExtractedEntity(
            address=f"${addr_hex}",
            name=f"addr_{addr_hex}",
            category="unknown",
            confidence=0.0,
            line_number=line_num,
            context=context.strip(),
            is_known=False,
            info=None,
        )

    def validate_entities(
        self, entities: list[ExtractedEntity], code: str
    ) -> dict[str, Any]:
        """Validate extracted entities make sense in context.

        Checks for:
        - Addressing mode consistency (e.g., direct page vs absolute)
        - Read/write consistency with instruction

        Args:
            entities: Previously extracted entities
            code: Original assembly code

        Returns:
            Validation results dict
        """
        issues: list[str] = []
        validated = 0

        for entity in entities:
            if not entity.is_known or not entity.info:
                continue

            # Check read/write consistency
            context_upper = entity.context.upper()
            info = entity.info

            # LDA, LDX, LDY, etc. = read
            is_read = any(
                op in context_upper for op in ["LDA", "LDX", "LDY", "CMP", "CPX", "CPY"]
            )
            # STA, STX, STY, etc. = write
            is_write = any(
                op in context_upper for op in ["STA", "STX", "STY", "STZ", "INC", "DEC"]
            )

            if is_read and info.read_write == "w":
                issues.append(
                    f"Line {entity.line_number}: Reading from write-only {entity.name}"
                )
            elif is_write and info.read_write == "r":
                issues.append(
                    f"Line {entity.line_number}: Writing to read-only {entity.name}"
                )
            else:
                validated += 1

        return {
            "validated": validated,
            "total": len([e for e in entities if e.is_known]),
            "issues": issues,
            "is_valid": len(issues) == 0,
        }


def extract_entities(code: str) -> list[str]:
    """Convenience function to extract entity names from code.

    Args:
        code: Assembly code text

    Returns:
        List of known entity names for kg_entities field
    """
    extractor = EntityExtractor()
    result = extractor.extract(code)
    return result.entity_names()


def extract_with_stats(code: str) -> tuple[list[str], dict[str, Any]]:
    """Extract entities with statistics.

    Args:
        code: Assembly code text

    Returns:
        Tuple of (entity_names, stats_dict)
    """
    extractor = EntityExtractor()
    result = extractor.extract(code)
    return result.entity_names(), result.stats
