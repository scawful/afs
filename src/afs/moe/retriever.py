"""RAG retriever for 65816 assembly knowledge bases."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document."""

    id: str
    text: str
    score: float
    source: str  # e.g., "alttp", "oracle"


@dataclass
class KnowledgeBase:
    """A loaded knowledge base with embeddings."""

    name: str
    index_path: Path
    embeddings_dir: Path

    # Loaded data
    index: dict[str, str] = field(default_factory=dict)  # id -> embedding file
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)

    def load(self) -> None:
        """Load the knowledge base index and embeddings."""
        if not self.index_path.exists():
            logger.warning(f"Index not found: {self.index_path}")
            return

        with open(self.index_path) as f:
            self.index = json.load(f)

        logger.info(f"Loaded {len(self.index)} entries from {self.name}")

    def load_embedding(self, doc_id: str) -> tuple[np.ndarray, str] | None:
        """Load a single embedding by ID."""
        if doc_id in self.embeddings:
            return self.embeddings[doc_id], self.texts.get(doc_id, "")

        filename = self.index.get(doc_id)
        if not filename:
            return None

        embedding_file = self.embeddings_dir / filename
        if not embedding_file.exists():
            return None

        with open(embedding_file) as f:
            data = json.load(f)

        embedding = np.array(data.get("embedding", []), dtype=np.float32)
        text = data.get("text_preview", "")

        # Cache
        self.embeddings[doc_id] = embedding
        self.texts[doc_id] = text

        return embedding, text

    def load_all_embeddings(self) -> None:
        """Load all embeddings into memory for fast retrieval."""
        for doc_id in self.index:
            self.load_embedding(doc_id)
        logger.info(f"Loaded {len(self.embeddings)} embeddings for {self.name}")


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    knowledge_bases: list[Path] = field(default_factory=list)
    top_k: int = 5
    min_score: float = 0.3
    preload_embeddings: bool = False  # Load all into memory at startup

    @classmethod
    def default(cls) -> RetrieverConfig:
        """Default config using ~/.context/knowledge/"""
        knowledge_root = Path.home() / ".context" / "knowledge"
        return cls(
            knowledge_bases=[
                knowledge_root / "alttp",
                knowledge_root / "oracle-of-secrets",
            ]
        )


class Retriever:
    """Retrieves relevant context from knowledge bases using embeddings."""

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ):
        """Initialize retriever.

        Args:
            config: Retriever configuration
            embed_fn: Function to embed query text. If None, uses keyword matching.
        """
        self.config = config or RetrieverConfig.default()
        self.embed_fn = embed_fn
        self.knowledge_bases: list[KnowledgeBase] = []

        self._load_knowledge_bases()

    def _load_knowledge_bases(self) -> None:
        """Load all configured knowledge bases."""
        for kb_path in self.config.knowledge_bases:
            if not kb_path.exists():
                logger.warning(f"Knowledge base not found: {kb_path}")
                continue

            kb = KnowledgeBase(
                name=kb_path.name,
                index_path=kb_path / "embedding_index.json",
                embeddings_dir=kb_path / "embeddings",
            )
            kb.load()

            if self.config.preload_embeddings:
                kb.load_all_embeddings()

            self.knowledge_bases.append(kb)

        logger.info(f"Loaded {len(self.knowledge_bases)} knowledge bases")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _keyword_search(self, query: str, kb: KnowledgeBase, top_k: int) -> list[RetrievalResult]:
        """Fallback keyword-based search when no embedding function available."""
        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for doc_id in kb.index:
            # Load text if not cached
            if doc_id not in kb.texts:
                kb.load_embedding(doc_id)

            text = kb.texts.get(doc_id, "")
            text_lower = text.lower()

            # Simple keyword scoring
            score = 0.0
            for term in query_terms:
                if term in text_lower:
                    score += 1.0
                if term in doc_id.lower():
                    score += 0.5

            if score > 0:
                # Normalize by query length
                score = score / len(query_terms)
                results.append(RetrievalResult(
                    id=doc_id,
                    text=text,
                    score=score,
                    source=kb.name,
                ))

        # Sort by score and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _embedding_search(
        self,
        query_embedding: np.ndarray,
        kb: KnowledgeBase,
        top_k: int
    ) -> list[RetrievalResult]:
        """Search using embedding similarity."""
        results = []

        for doc_id in kb.index:
            result = kb.load_embedding(doc_id)
            if result is None:
                continue

            embedding, text = result
            if len(embedding) == 0:
                continue

            score = self._cosine_similarity(query_embedding, embedding)

            if score >= self.config.min_score:
                results.append(RetrievalResult(
                    id=doc_id,
                    text=text,
                    score=score,
                    source=kb.name,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query text
            top_k: Number of results to return (default from config)

        Returns:
            List of retrieval results sorted by relevance
        """
        top_k = top_k or self.config.top_k
        all_results: list[RetrievalResult] = []

        # Get query embedding if embedding function available
        query_embedding = None
        if self.embed_fn is not None:
            try:
                query_embedding = self.embed_fn(query)
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to keyword: {e}")

        for kb in self.knowledge_bases:
            if query_embedding is not None:
                results = self._embedding_search(query_embedding, kb, top_k)
            else:
                results = self._keyword_search(query, kb, top_k)
            all_results.extend(results)

        # Re-sort combined results and take top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as context string for LLM."""
        if not results:
            return ""

        lines = ["## Relevant Context\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**[{r.source}] {r.id}** (relevance: {r.score:.2f})")
            lines.append(f"{r.text}\n")

        return "\n".join(lines)


def create_ollama_embed_fn(
    model: str = "nomic-embed-text",
    host: str = "http://localhost:11435",
) -> Callable[[str], np.ndarray]:
    """Create an embedding function using Ollama."""
    import httpx

    def embed(text: str) -> np.ndarray:
        response = httpx.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    return embed
