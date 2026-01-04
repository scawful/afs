"""Domain-agnostic knowledge graph framework.

Provides abstract base classes for knowledge graphs that can be
specialized for different domains (Zelda/ALTTP, Avatar/Personal).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Iterator, Callable, Any
import json
from pathlib import Path


class NodeType(Enum):
    """Base node types - extend for domain-specific types."""
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATION = "relation"


class EdgeType(Enum):
    """Base edge types - extend for domain-specific types."""
    REFERENCES = "references"
    CONTAINS = "contains"
    CALLS = "calls"
    DEPENDS_ON = "depends_on"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    name: str
    node_type: str
    properties: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphNode":
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=data["node_type"],
            properties=data.get("properties", {}),
        )


@dataclass
class GraphEdge:
    """An edge connecting two nodes."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["edge_type"],
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
        )


@dataclass
class GraphConstraint:
    """A constraint that can be validated against graph operations."""
    name: str
    description: str
    validator: Callable[[Any], bool]
    severity: str = "error"  # error, warning, info

    def validate(self, context: Any) -> tuple[bool, str]:
        """Validate the constraint, return (passed, message)."""
        try:
            if self.validator(context):
                return True, ""
            return False, f"Constraint '{self.name}' violated: {self.description}"
        except Exception as e:
            return False, f"Constraint '{self.name}' error: {e}"


class KnowledgeGraph(ABC):
    """Abstract base class for knowledge graphs."""

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[str]] = {}  # node_id -> [neighbor_ids]
        self._reverse_adjacency: dict[str, list[str]] = {}  # node_id -> [predecessor_ids]
        self._constraints: list[GraphConstraint] = []

    # Node operations
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.id not in self._reverse_adjacency:
            self._reverse_adjacency[node.id] = []

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def find_nodes(
        self,
        node_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        property_filter: Optional[dict] = None,
    ) -> Iterator[GraphNode]:
        """Find nodes matching criteria."""
        for node in self._nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if name_pattern and name_pattern.lower() not in node.name.lower():
                continue
            if property_filter:
                match = all(
                    node.properties.get(k) == v
                    for k, v in property_filter.items()
                )
                if not match:
                    continue
            yield node

    # Edge operations
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self._edges.append(edge)
        if edge.source_id not in self._adjacency:
            self._adjacency[edge.source_id] = []
        self._adjacency[edge.source_id].append(edge.target_id)

        if edge.target_id not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_id] = []
        self._reverse_adjacency[edge.target_id].append(edge.source_id)

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> Iterator[GraphEdge]:
        """Get edges matching criteria."""
        for edge in self._edges:
            if source_id and edge.source_id != source_id:
                continue
            if target_id and edge.target_id != target_id:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            yield edge

    def get_neighbors(self, node_id: str) -> list[GraphNode]:
        """Get all nodes connected from this node."""
        neighbor_ids = self._adjacency.get(node_id, [])
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def get_predecessors(self, node_id: str) -> list[GraphNode]:
        """Get all nodes that connect to this node."""
        pred_ids = self._reverse_adjacency.get(node_id, [])
        return [self._nodes[pid] for pid in pred_ids if pid in self._nodes]

    # Subgraph extraction
    def extract_subgraph(
        self,
        root_ids: list[str],
        max_depth: int = 2,
        edge_types: Optional[list[str]] = None,
    ) -> "KnowledgeGraph":
        """Extract a subgraph starting from root nodes."""
        visited = set()
        to_visit = [(rid, 0) for rid in root_ids]
        subgraph = self.__class__()

        while to_visit:
            node_id, depth = to_visit.pop(0)
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)

            node = self.get_node(node_id)
            if node:
                subgraph.add_node(node)

            if depth < max_depth:
                for edge in self.get_edges(source_id=node_id):
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    subgraph.add_edge(edge)
                    if edge.target_id not in visited:
                        to_visit.append((edge.target_id, depth + 1))

        return subgraph

    # Constraint management
    def add_constraint(self, constraint: GraphConstraint) -> None:
        """Add a validation constraint."""
        self._constraints.append(constraint)

    def validate_constraints(self, context: Any) -> list[tuple[bool, str]]:
        """Validate all constraints against context."""
        return [c.validate(context) for c in self._constraints]

    # Serialization
    def to_dict(self) -> dict:
        """Serialize graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def save(self, path: Path) -> None:
        """Save graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: Path) -> None:
        """Load graph from JSON file."""
        with open(path) as f:
            data = json.load(f)

        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()

        for node_data in data.get("nodes", []):
            self.add_node(GraphNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            self.add_edge(GraphEdge.from_dict(edge_data))

    # Abstract methods for domain-specific behavior
    @abstractmethod
    def get_context_for_prompt(self, query: str, max_entities: int = 10) -> str:
        """Get relevant context from graph for a prompt."""
        pass

    @abstractmethod
    def validate_output(self, output: str) -> list[tuple[bool, str]]:
        """Validate generated output against graph constraints."""
        pass

    # Statistics
    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        node_types = {}
        edge_types = {}

        for node in self._nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        for edge in self._edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "node_types": node_types,
            "edge_types": edge_types,
        }
