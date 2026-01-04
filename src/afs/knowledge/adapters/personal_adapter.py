"""Personal knowledge graph adapter for Avatar models.

Provides domain-specific graph operations for personal context,
preferences, and persona consistency.
"""

import re
from typing import Optional
from pathlib import Path
from ..graph_core import KnowledgeGraph, GraphNode, GraphEdge, GraphConstraint


class PersonalNodeType:
    """Personal-specific node types."""
    FACT = "fact"
    PREFERENCE = "preference"
    MEMORY = "memory"
    STYLE = "style"
    TOPIC = "topic"
    PERSON = "person"
    PROJECT = "project"


class PersonalEdgeType:
    """Personal-specific edge types."""
    LIKES = "likes"
    DISLIKES = "dislikes"
    KNOWS = "knows"
    WORKS_ON = "works_on"
    RELATED_TO = "related_to"
    REMINDS_OF = "reminds_of"


class PersonalKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph specialized for personal context and persona."""

    def __init__(self):
        super().__init__()
        self._style_rules: list[str] = []
        self._add_default_constraints()

    def _add_default_constraints(self) -> None:
        """Add default validation constraints for persona consistency."""
        self.add_constraint(GraphConstraint(
            name="persona_consistency",
            description="Output should match persona style",
            validator=lambda ctx: self._validate_persona(ctx),
            severity="warning",
        ))

        self.add_constraint(GraphConstraint(
            name="factual_consistency",
            description="Output should not contradict known facts",
            validator=lambda ctx: self._validate_facts(ctx),
            severity="error",
        ))

    def _validate_persona(self, output: str) -> bool:
        """Check if output matches persona style."""
        # Check against style rules
        for rule in self._style_rules:
            if rule.startswith("avoid:"):
                pattern = rule[6:].strip()
                if re.search(pattern, output, re.IGNORECASE):
                    return False
            elif rule.startswith("prefer:"):
                # Soft preference, don't fail
                pass
        return True

    def _validate_facts(self, output: str) -> bool:
        """Check if output contradicts known facts."""
        # Get all fact nodes
        for node in self.find_nodes(node_type=PersonalNodeType.FACT):
            # Check for direct contradictions (simplified)
            fact = node.properties.get("value", "")
            negation = node.properties.get("negation", "")
            if negation and negation.lower() in output.lower():
                return False
        return True

    def add_fact(self, name: str, value: str, category: str = "general") -> GraphNode:
        """Add a personal fact to the graph."""
        node = GraphNode(
            id=f"fact_{name.lower().replace(' ', '_')}",
            name=name,
            node_type=PersonalNodeType.FACT,
            properties={
                "value": value,
                "category": category,
            }
        )
        self.add_node(node)
        return node

    def add_preference(
        self,
        name: str,
        sentiment: str,  # "like", "dislike", "neutral"
        strength: float = 0.5,
        context: str = "",
    ) -> GraphNode:
        """Add a preference to the graph."""
        node = GraphNode(
            id=f"pref_{name.lower().replace(' ', '_')}",
            name=name,
            node_type=PersonalNodeType.PREFERENCE,
            properties={
                "sentiment": sentiment,
                "strength": strength,
                "context": context,
            }
        )
        self.add_node(node)
        return node

    def add_style_rule(self, rule: str) -> None:
        """Add a writing style rule.

        Rules can be:
        - "avoid: pattern" - avoid matching pattern
        - "prefer: pattern" - prefer using pattern
        - "tone: casual|formal|technical"
        """
        self._style_rules.append(rule)

    def add_memory(
        self,
        content: str,
        timestamp: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> GraphNode:
        """Add a memory/experience to the graph."""
        import hashlib
        mem_id = hashlib.md5(content.encode()).hexdigest()[:8]

        node = GraphNode(
            id=f"mem_{mem_id}",
            name=content[:50] + "..." if len(content) > 50 else content,
            node_type=PersonalNodeType.MEMORY,
            properties={
                "content": content,
                "timestamp": timestamp,
                "importance": importance,
                "tags": tags or [],
            }
        )
        self.add_node(node)
        return node

    def get_context_for_prompt(self, query: str, max_entities: int = 10) -> str:
        """Get relevant personal context for a prompt."""
        context_parts = []

        # Search for relevant facts
        relevant_nodes = []

        # Extract keywords from query
        words = set(query.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "can", "do", "i", "you"}
        keywords = words - stop_words

        # Find matching nodes
        for node in self._nodes.values():
            name_lower = node.name.lower()
            props_str = str(node.properties).lower()

            # Check if any keyword matches
            for keyword in keywords:
                if keyword in name_lower or keyword in props_str:
                    relevant_nodes.append(node)
                    break

            if len(relevant_nodes) >= max_entities:
                break

        # Build context string
        if relevant_nodes:
            context_parts.append("Relevant personal context:")
            for node in relevant_nodes:
                if node.node_type == PersonalNodeType.FACT:
                    context_parts.append(f"- {node.name}: {node.properties.get('value')}")
                elif node.node_type == PersonalNodeType.PREFERENCE:
                    sentiment = node.properties.get("sentiment")
                    context_parts.append(f"- {sentiment}s: {node.name}")
                elif node.node_type == PersonalNodeType.MEMORY:
                    context_parts.append(f"- Memory: {node.properties.get('content', '')[:100]}")

        # Add style rules if any
        if self._style_rules:
            context_parts.append("\nStyle guidelines:")
            for rule in self._style_rules[:5]:
                context_parts.append(f"- {rule}")

        return "\n".join(context_parts)

    def validate_output(self, output: str) -> list[tuple[bool, str]]:
        """Validate generated output against personal constraints."""
        return self.validate_constraints(output)

    def load_from_notes(self, notes_dir: Path) -> int:
        """Load personal facts from markdown notes."""
        count = 0
        if not notes_dir.exists():
            return count

        for md_file in notes_dir.glob("**/*.md"):
            with open(md_file) as f:
                content = f.read()

            # Extract facts from frontmatter or content
            # This is a simplified parser
            lines = content.split("\n")
            for line in lines:
                if line.startswith("- ") and ":" in line:
                    parts = line[2:].split(":", 1)
                    if len(parts) == 2:
                        self.add_fact(
                            name=parts[0].strip(),
                            value=parts[1].strip(),
                            category=md_file.stem,
                        )
                        count += 1

        return count

    def export_persona_prompt(self) -> str:
        """Export a system prompt describing the persona."""
        lines = ["You are a personal AI assistant with the following context:"]

        # Add facts
        facts = list(self.find_nodes(node_type=PersonalNodeType.FACT))
        if facts:
            lines.append("\nKnown facts:")
            for node in facts[:20]:
                lines.append(f"- {node.name}: {node.properties.get('value')}")

        # Add preferences
        prefs = list(self.find_nodes(node_type=PersonalNodeType.PREFERENCE))
        if prefs:
            lines.append("\nPreferences:")
            for node in prefs[:10]:
                sentiment = node.properties.get("sentiment")
                lines.append(f"- {sentiment}s {node.name}")

        # Add style rules
        if self._style_rules:
            lines.append("\nCommunication style:")
            for rule in self._style_rules:
                lines.append(f"- {rule}")

        return "\n".join(lines)
