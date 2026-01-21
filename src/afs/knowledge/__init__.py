"""Knowledge base for ALTTP and SNES assembly."""

from .alttp_addresses import (
    ALTTP_ADDRESSES,
    INVENTORY_ADDRESSES,
    LINK_STATE_ADDRESSES,
    SPRITE_TABLES,
    WRAM_ADDRESSES,
    AddressCategory,
    AddressInfo,
    format_address_reference,
    get_address_info,
    get_addresses_by_category,
    lookup_by_address,
)
from .entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractionResult,
    extract_entities,
    extract_with_stats,
)

__all__ = [
    # Address data
    "ALTTP_ADDRESSES",
    "SPRITE_TABLES",
    "LINK_STATE_ADDRESSES",
    "INVENTORY_ADDRESSES",
    "WRAM_ADDRESSES",
    "AddressCategory",
    "AddressInfo",
    # Address utilities
    "get_address_info",
    "get_addresses_by_category",
    "format_address_reference",
    "lookup_by_address",
    # Entity extraction
    "EntityExtractor",
    "ExtractedEntity",
    "ExtractionResult",
    "extract_entities",
    "extract_with_stats",
]
