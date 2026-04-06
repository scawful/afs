"""Knowledge graph primitives plus optional extension-owned domain knowledge."""

from .adapters.personal_adapter import PersonalKnowledgeGraph
from .graph_core import (
    EdgeType,
    GraphConstraint,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeType,
)

__all__ = [
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "GraphConstraint",
    "KnowledgeGraph",
    "PersonalKnowledgeGraph",
]

try:  # pragma: no cover - compatibility path
    from afs_ext.knowledge.alttp_addresses import (
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
    from afs_ext.knowledge.entity_extractor import (
        EntityExtractor,
        ExtractedEntity,
        ExtractionResult,
        extract_entities,
        extract_with_stats,
    )

    __all__.extend(
        [
            "ALTTP_ADDRESSES",
            "SPRITE_TABLES",
            "LINK_STATE_ADDRESSES",
            "INVENTORY_ADDRESSES",
            "WRAM_ADDRESSES",
            "AddressCategory",
            "AddressInfo",
            "get_address_info",
            "get_addresses_by_category",
            "format_address_reference",
            "lookup_by_address",
            "EntityExtractor",
            "ExtractedEntity",
            "ExtractionResult",
            "extract_entities",
            "extract_with_stats",
        ]
    )
except Exception:
    pass
