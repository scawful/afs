"""Domain-specific knowledge graph adapters."""

from .personal_adapter import PersonalKnowledgeGraph

__all__ = ["PersonalKnowledgeGraph"]

try:  # pragma: no cover - compatibility path
    from afs_scawful.knowledge.adapters.alttp_adapter import ALTTPKnowledgeGraph

    __all__.append("ALTTPKnowledgeGraph")
except Exception:
    pass
