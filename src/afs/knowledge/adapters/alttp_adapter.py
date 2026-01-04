"""ALTTP-specific knowledge graph adapter.

Wraps ALTTP_ADDRESSES and provides domain-specific graph operations
for 65816 assembly validation and context retrieval.
"""

import re
from typing import Optional
from ..graph_core import KnowledgeGraph, GraphNode, GraphEdge, GraphConstraint


class ALTTPNodeType:
    """ALTTP-specific node types."""
    ADDRESS = "address"
    ROUTINE = "routine"
    REGISTER = "register"
    SPRITE = "sprite"
    ITEM = "item"
    ROOM = "room"


class ALTTPEdgeType:
    """ALTTP-specific edge types."""
    READS = "reads"
    WRITES = "writes"
    CALLS = "calls"
    JUMPS_TO = "jumps_to"
    USES_SPRITE = "uses_sprite"
    IN_ROOM = "in_room"


class ALTTPKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph specialized for ALTTP/65816 assembly."""

    def __init__(self):
        super().__init__()
        self._address_patterns = {}  # Quick lookup by address
        self._load_alttp_addresses()
        self._add_default_constraints()

    def _load_alttp_addresses(self) -> None:
        """Load ALTTP addresses into graph nodes."""
        try:
            from ..alttp_addresses import ALTTP_ADDRESSES, AddressCategory
        except ImportError:
            return

        for name, info in ALTTP_ADDRESSES.items():
            # Create node for each address
            node = GraphNode(
                id=f"addr_{name}",
                name=info.name,
                node_type=ALTTPNodeType.ADDRESS,
                properties={
                    "address": info.full_address,
                    "raw_address": info.address,
                    "bank": info.bank,
                    "size": info.size,
                    "category": info.category.value,
                    "description": info.description,
                    "read_write": info.read_write,
                    "notes": info.notes,
                }
            )
            self.add_node(node)

            # Index by address for quick lookup
            if isinstance(info.address, int):
                full_addr = f"${info.bank}{info.address:04X}"
                self._address_patterns[full_addr.upper()] = node.id
                # Also index short form for direct page
                if info.address < 0x2000:
                    short = f"${info.address:02X}" if info.address < 0x100 else f"${info.address:04X}"
                    self._address_patterns[short.upper()] = node.id

        # Add edges between related addresses
        self._add_relationship_edges()

    def _add_relationship_edges(self) -> None:
        """Add edges between related addresses."""
        # Link state addresses are related
        link_nodes = list(self.find_nodes(property_filter={"category": "link_state"}))
        for i, node1 in enumerate(link_nodes):
            for node2 in link_nodes[i+1:]:
                self.add_edge(GraphEdge(
                    source_id=node1.id,
                    target_id=node2.id,
                    edge_type="related_to",
                    weight=0.5,
                ))

    def _add_default_constraints(self) -> None:
        """Add default validation constraints for 65816 code."""
        # Constraint: Don't write to read-only addresses
        self.add_constraint(GraphConstraint(
            name="read_only_check",
            description="Cannot write to read-only addresses",
            validator=lambda ctx: self._validate_read_only(ctx),
            severity="error",
        ))

        # Constraint: Size mismatch detection
        self.add_constraint(GraphConstraint(
            name="size_check",
            description="Operation size should match address size",
            validator=lambda ctx: self._validate_size(ctx),
            severity="warning",
        ))

    def _validate_read_only(self, code: str) -> bool:
        """Check if code tries to write to read-only addresses."""
        # Find STA/STX/STY/STZ operations
        write_pattern = r'(STA|STX|STY|STZ)\s+(\$[0-9A-Fa-f]+)'
        for match in re.finditer(write_pattern, code, re.IGNORECASE):
            addr = match.group(2).upper()
            node_id = self._address_patterns.get(addr)
            if node_id:
                node = self.get_node(node_id)
                if node and node.properties.get("read_write") == "r":
                    return False
        return True

    def _validate_size(self, code: str) -> bool:
        """Check for size mismatches (simplified)."""
        # This is a simplified check - real validation would need mode tracking
        return True

    def get_context_for_prompt(self, query: str, max_entities: int = 10) -> str:
        """Get relevant ALTTP context for a prompt."""
        context_parts = []

        # Extract addresses mentioned in query
        addr_pattern = r'\$[0-9A-Fa-f]{2,6}'
        mentioned = set(re.findall(addr_pattern, query, re.IGNORECASE))

        # Get nodes for mentioned addresses
        relevant_nodes = []
        for addr in mentioned:
            node_id = self._address_patterns.get(addr.upper())
            if node_id:
                node = self.get_node(node_id)
                if node:
                    relevant_nodes.append(node)

        # Also search by keywords
        keywords = ["link", "sprite", "health", "inventory", "coordinate", "room"]
        for keyword in keywords:
            if keyword in query.lower():
                for node in self.find_nodes(name_pattern=keyword):
                    if node not in relevant_nodes:
                        relevant_nodes.append(node)
                    if len(relevant_nodes) >= max_entities:
                        break

        # Build context string
        if relevant_nodes:
            context_parts.append("Relevant ALTTP addresses:")
            for node in relevant_nodes[:max_entities]:
                props = node.properties
                context_parts.append(
                    f"- {props.get('address')}: {node.name} - {props.get('description')}"
                )

        return "\n".join(context_parts)

    def validate_output(self, output: str) -> list[tuple[bool, str]]:
        """Validate generated assembly against ALTTP constraints."""
        results = self.validate_constraints(output)

        # Additional ALTTP-specific checks
        # Check for valid addressing modes
        if re.search(r'LDA\s+#\$[0-9A-Fa-f]{5,}', output):
            results.append((False, "Immediate value too large for LDA"))

        # Check for known problematic patterns
        if "STA $2100" in output.upper() and "SEI" not in output.upper():
            results.append((False, "Writing to PPU register $2100 without disabling interrupts"))

        return results

    def lookup_address(self, address: str) -> Optional[GraphNode]:
        """Look up an address and return its node."""
        node_id = self._address_patterns.get(address.upper())
        if node_id:
            return self.get_node(node_id)
        return None

    def get_addresses_for_task(self, task_type: str) -> list[GraphNode]:
        """Get addresses relevant to a specific task type."""
        category_map = {
            "movement": "link_state",
            "health": "link_state",
            "inventory": "inventory",
            "sprite": "sprite",
            "graphics": "graphics",
        }

        category = category_map.get(task_type.lower())
        if category:
            return list(self.find_nodes(property_filter={"category": category}))
        return []

    def generate_asm_header(self) -> str:
        """Generate asar-compatible header with address defines."""
        lines = [
            "; ALTTP Address Definitions",
            "; Generated from ALTTPKnowledgeGraph",
            "",
        ]

        for node in self._nodes.values():
            if node.node_type == ALTTPNodeType.ADDRESS:
                props = node.properties
                addr = props.get("address", "")
                name = node.name.upper().replace(" ", "_").replace("'", "")
                lines.append(f"!{name} = {addr}")

        return "\n".join(lines)
