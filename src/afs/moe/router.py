"""MoE Router for 65816 assembly expert models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, TYPE_CHECKING

import httpx

from .classifier import IntentClassifier, QueryIntent, ClassificationResult
from .router_trainer import HybridRouter, RouterTrainingConfig

if TYPE_CHECKING:
    from .retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for an expert model."""

    name: str
    model_id: str
    intent: QueryIntent
    description: str
    host: str = "http://localhost:11435"  # Windows Ollama via tunnel
    temperature: float = 0.5
    top_p: float = 0.85


@dataclass
class RouterConfig:
    """MoE router configuration."""

    experts: list[ExpertConfig] = field(default_factory=list)
    fallback_model: str = "qwen2.5-coder:7b"
    fallback_host: str = "http://localhost:11435"
    enable_ensemble: bool = False
    log_routing: bool = True
    enable_rag: bool = True
    rag_top_k: int = 3
    embedding_model: str = "embeddinggemma:latest"
    embedding_host: str = "http://localhost:11435"

    # Router type: "keyword" or "hybrid"
    router_type: str = "hybrid"
    router_model_path: str | None = None  # Path to trained hybrid model

    @classmethod
    def default(cls) -> "RouterConfig":
        """Create default configuration with din/nayru experts."""
        return cls(
            experts=[
                ExpertConfig(
                    name="din",
                    model_id="din-v2:latest",
                    intent=QueryIntent.OPTIMIZATION,
                    description="65816 assembly optimization specialist",
                ),
                ExpertConfig(
                    name="nayru",
                    model_id="nayru-v5:latest",
                    intent=QueryIntent.GENERATION,
                    description="65816 assembly code generation",
                ),
                ExpertConfig(
                    name="farore",
                    model_id="farore-v1:latest",
                    intent=QueryIntent.DEBUGGING,
                    description="65816 assembly debugging specialist",
                    temperature=0.4,
                ),
            ]
        )


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    expert: ExpertConfig | None
    classification: ClassificationResult
    reason: str


@dataclass
class GenerationResult:
    """Result from model generation."""

    content: str
    model: str
    expert_name: str | None
    classification: ClassificationResult
    tokens_generated: int = 0
    retrieved_context: list["RetrievalResult"] = field(default_factory=list)


class MoERouter:
    """Mixture of Experts router for 65816 assembly models."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        retriever: "Retriever | None" = None,
    ):
        """Initialize router with configuration.

        Args:
            config: Router configuration
            retriever: Optional retriever for RAG. If None and enable_rag=True,
                      a default retriever will be created.
        """
        self.config = config or RouterConfig.default()
        self._client: httpx.AsyncClient | None = None
        self._retriever = retriever

        # Initialize classifier based on router_type
        if self.config.router_type == "hybrid":
            try:
                from pathlib import Path
                self.classifier = HybridRouter()
                model_path = None
                if self.config.router_model_path:
                    model_path = Path(self.config.router_model_path)
                self.classifier.load(model_path)
                logger.info("Using hybrid router (learned + keyword)")
            except Exception as e:
                logger.warning(f"Failed to load hybrid router: {e}, falling back to keyword")
                self.classifier = IntentClassifier()
        else:
            self.classifier = IntentClassifier()
            logger.info("Using keyword-based router")

        # Lazy-load retriever if RAG enabled
        if self.config.enable_rag and self._retriever is None:
            try:
                from .retriever import Retriever, RetrieverConfig, create_ollama_embed_fn

                # Create embedding function for semantic search
                embed_fn = None
                try:
                    embed_fn = create_ollama_embed_fn(
                        model=self.config.embedding_model,
                        host=self.config.embedding_host,
                    )
                    logger.info(
                        f"Initialized embedding function with {self.config.embedding_model}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create embed_fn, using keyword fallback: {e}")

                self._retriever = Retriever(
                    config=RetrieverConfig.default(),
                    embed_fn=embed_fn,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize retriever: {e}")
                self._retriever = None

        # Build intent -> expert mapping
        self._expert_map: dict[QueryIntent, ExpertConfig] = {
            expert.intent: expert for expert in self.config.experts
        }

    async def __aenter__(self) -> "MoERouter":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def route(self, query: str) -> RoutingDecision:
        """Route query to appropriate expert based on intent classification."""
        classification = self.classifier.classify(query)

        if self.config.log_routing:
            logger.info(
                f"Classified query as {classification.intent.value} "
                f"(confidence={classification.confidence:.2f})"
            )

        expert = self._expert_map.get(classification.intent)

        if expert:
            reason = f"Routed to {expert.name} for {classification.intent.value}"
        else:
            reason = f"No expert for {classification.intent.value}, using fallback"

        return RoutingDecision(
            expert=expert,
            classification=classification,
            reason=reason,
        )

    def retrieve_context(self, query: str) -> list["RetrievalResult"]:
        """Retrieve relevant context for a query."""
        if not self._retriever or not self.config.enable_rag:
            return []

        try:
            return self._retriever.retrieve(query, top_k=self.config.rag_top_k)
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    async def generate(
        self,
        query: str,
        system_prompt: str | None = None,
        stream: bool = False,
        use_rag: bool | None = None,
    ) -> GenerationResult | AsyncIterator[str]:
        """Generate response using routed expert model.

        Args:
            query: The user query
            system_prompt: Optional system prompt override
            stream: Whether to stream the response
            use_rag: Override RAG setting (None = use config default)
        """
        if not self._client:
            raise RuntimeError("Router not initialized. Use async context manager.")

        decision = self.route(query)

        # Retrieve context if RAG enabled
        # Skip RAG for GENERATION intent - ALTTP-specific context hurts general code gen
        use_rag = use_rag if use_rag is not None else self.config.enable_rag
        skip_rag_intents = {QueryIntent.GENERATION}
        retrieved = []
        context_str = ""
        if use_rag and self._retriever and decision.classification.intent not in skip_rag_intents:
            retrieved = self.retrieve_context(query)
            if retrieved:
                context_str = self._retriever.format_context(retrieved)
                if self.config.log_routing:
                    logger.info(f"Retrieved {len(retrieved)} context documents")
        elif use_rag and decision.classification.intent in skip_rag_intents:
            if self.config.log_routing:
                logger.info(f"Skipping RAG for {decision.classification.intent.value} intent")

        if decision.expert:
            model_id = decision.expert.model_id
            host = decision.expert.host
            expert_name = decision.expert.name
            temperature = decision.expert.temperature
            top_p = decision.expert.top_p
        else:
            model_id = self.config.fallback_model
            host = self.config.fallback_host
            expert_name = None
            temperature = 0.7
            top_p = 0.9

        if self.config.log_routing:
            logger.info(f"Using model {model_id} from {host}")

        # Inject context into prompt if available
        augmented_query = query
        if context_str:
            augmented_query = f"{context_str}\n\n## Query\n{query}"

        payload = {
            "model": model_id,
            "prompt": augmented_query,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stream:
            return self._stream_generate(host, payload, expert_name, decision.classification)
        else:
            result = await self._batch_generate(host, payload, expert_name, decision.classification)
            result.retrieved_context = retrieved
            return result

    async def _batch_generate(
        self,
        host: str,
        payload: dict,
        expert_name: str | None,
        classification: ClassificationResult,
    ) -> GenerationResult:
        """Non-streaming generation."""
        response = await self._client.post(
            f"{host}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            content=data.get("response", ""),
            model=payload["model"],
            expert_name=expert_name,
            classification=classification,
            tokens_generated=data.get("eval_count", 0),
        )

    async def _stream_generate(
        self,
        host: str,
        payload: dict,
        expert_name: str | None,
        classification: ClassificationResult,
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        async with self._client.stream(
            "POST",
            f"{host}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if chunk := data.get("response"):
                        yield chunk

    async def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[str]:
        """Chat completion using routed expert model."""
        if not self._client:
            raise RuntimeError("Router not initialized. Use async context manager.")

        # Extract last user message for classification
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        decision = self.route(last_user_msg)

        if decision.expert:
            model_id = decision.expert.model_id
            host = decision.expert.host
            expert_name = decision.expert.name
        else:
            model_id = self.config.fallback_model
            host = self.config.fallback_host
            expert_name = None

        if self.config.log_routing:
            logger.info(f"Chat using model {model_id}")

        payload = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }

        if stream:
            return self._stream_chat(host, payload, expert_name, decision.classification)
        else:
            return await self._batch_chat(host, payload, expert_name, decision.classification)

    async def _batch_chat(
        self,
        host: str,
        payload: dict,
        expert_name: str | None,
        classification: ClassificationResult,
    ) -> GenerationResult:
        """Non-streaming chat."""
        response = await self._client.post(
            f"{host}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            content=data.get("message", {}).get("content", ""),
            model=payload["model"],
            expert_name=expert_name,
            classification=classification,
            tokens_generated=data.get("eval_count", 0),
        )

    async def _stream_chat(
        self,
        host: str,
        payload: dict,
        expert_name: str | None,
        classification: ClassificationResult,
    ) -> AsyncIterator[str]:
        """Streaming chat."""
        async with self._client.stream(
            "POST",
            f"{host}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if chunk := data.get("message", {}).get("content"):
                        yield chunk

    def list_experts(self) -> list[ExpertConfig]:
        """List configured experts."""
        return list(self.config.experts)

    def get_expert(self, intent: QueryIntent) -> ExpertConfig | None:
        """Get expert for specific intent."""
        return self._expert_map.get(intent)
