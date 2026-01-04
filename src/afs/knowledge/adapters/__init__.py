"""Domain-specific knowledge graph adapters."""

from .alttp_adapter import ALTTPKnowledgeGraph
from .personal_adapter import PersonalKnowledgeGraph

__all__ = ["ALTTPKnowledgeGraph", "PersonalKnowledgeGraph"]
