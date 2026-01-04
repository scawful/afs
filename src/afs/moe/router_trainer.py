"""Learned MoE Router for task→expert classification.

Replaces keyword-based routing with a trained classifier.
Uses lightweight embeddings + classification for low-latency routing.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .classifier import QueryIntent, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class RoutingExample:
    """A single training example for the router."""

    instruction: str
    expert: QueryIntent
    confidence: float = 1.0
    source: str = ""  # e.g., "benchmark", "curriculum", "manual"

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "expert": self.expert.value,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingExample":
        return cls(
            instruction=data["instruction"],
            expert=QueryIntent(data["expert"]),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
        )


@dataclass
class RouterTrainingConfig:
    """Configuration for router training."""

    # Model settings
    embedding_dim: int = 384  # Size of instruction embeddings
    hidden_dim: int = 128     # Hidden layer size
    num_classes: int = 4      # din, nayru, farore, veran

    # Training settings
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.1

    # Paths
    model_path: Path = field(default_factory=lambda: Path("~/.cache/afs/router_model.pkl").expanduser())
    training_data_path: Path = field(default_factory=lambda: Path("~/.cache/afs/router_training.jsonl").expanduser())


class EmbeddingFunction:
    """Abstract embedding function interface."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning (n, dim) array."""
        raise NotImplementedError


class TFIDFEmbedding(EmbeddingFunction):
    """Simple TF-IDF based embeddings for fast, local routing."""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vectorizer = None
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vectorizer on training texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using fitted TF-IDF."""
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedding not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()

    def save(self, path: Path) -> None:
        """Save vectorizer to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path) -> None:
        """Load vectorizer from disk."""
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


class SentenceTransformerEmbedding(EmbeddingFunction):
    """Sentence transformer embeddings for semantic routing."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using sentence transformer."""
        self._load_model()
        return self._model.encode(texts, show_progress_bar=False)


class LearnedRouter:
    """Learned classifier for MoE routing.

    Architecture:
        Instruction → Embedding → MLP → Expert Probabilities

    Can use either:
    - TF-IDF embeddings (fast, no external deps)
    - Sentence transformer (more accurate, requires model)
    """

    INTENT_MAP = {
        0: QueryIntent.OPTIMIZATION,  # din
        1: QueryIntent.GENERATION,    # nayru
        2: QueryIntent.DEBUGGING,     # farore
        3: QueryIntent.GENERAL,       # veran/fallback
    }

    def __init__(
        self,
        config: RouterTrainingConfig | None = None,
        embedding: EmbeddingFunction | None = None,
    ):
        self.config = config or RouterTrainingConfig()
        self.embedding = embedding or TFIDFEmbedding()
        self._classifier = None
        self._trained = False

    def _create_classifier(self, input_dim: int):
        """Create the MLP classifier."""
        from sklearn.neural_network import MLPClassifier

        self._classifier = MLPClassifier(
            hidden_layer_sizes=(self.config.hidden_dim,),
            activation="relu",
            solver="adam",
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=self.config.validation_split,
        )

    def train(self, examples: list[RoutingExample]) -> dict[str, float]:
        """Train the router on labeled examples."""
        if len(examples) < 10:
            raise ValueError(f"Need at least 10 examples, got {len(examples)}")

        examples = list(examples)
        random.shuffle(examples)

        texts = [ex.instruction for ex in examples]
        labels = [list(self.INTENT_MAP.values()).index(ex.expert) for ex in examples]

        if isinstance(self.embedding, TFIDFEmbedding):
            self.embedding.fit(texts)

        X = self.embedding.embed(texts)
        y = np.array(labels)

        self._create_classifier(X.shape[1])
        self._classifier.fit(X, y)
        self._trained = True

        train_acc = self._classifier.score(X, y)
        val_loss = self._classifier.loss_

        logger.info(f"Training complete: acc={train_acc:.3f}, loss={val_loss:.4f}")

        return {
            "train_accuracy": train_acc,
            "final_loss": val_loss,
            "n_samples": len(examples),
            "n_iterations": self._classifier.n_iter_,
        }

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query to an expert intent."""
        if not self._trained:
            raise RuntimeError("Router not trained. Call train() first.")

        X = self.embedding.embed([query])
        probs = self._classifier.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        intent = self.INTENT_MAP[pred_class]

        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            matched_patterns=[f"learned_class_{pred_class}"],
        )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        """Classify multiple queries efficiently."""
        if not self._trained:
            raise RuntimeError("Router not trained. Call train() first.")

        X = self.embedding.embed(queries)
        probs = self._classifier.predict_proba(X)

        results = []
        for i, query in enumerate(queries):
            pred_class = int(np.argmax(probs[i]))
            confidence = float(probs[i][pred_class])
            intent = self.INTENT_MAP[pred_class]
            results.append(ClassificationResult(
                intent=intent,
                confidence=confidence,
                matched_patterns=[f"learned_class_{pred_class}"],
            ))
        return results

    def save(self, path: Path | None = None) -> Path:
        """Save trained router to disk."""
        path = path or self.config.model_path
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "classifier": self._classifier,
            "config": self.config,
            "intent_map": self.INTENT_MAP,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        if isinstance(self.embedding, TFIDFEmbedding):
            embed_path = path.with_suffix(".embed.pkl")
            self.embedding.save(embed_path)

        logger.info(f"Saved router to {path}")
        return path

    def load(self, path: Path | None = None) -> None:
        """Load trained router from disk."""
        path = path or self.config.model_path

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self._classifier = model_data["classifier"]
        self.config = model_data.get("config", self.config)
        self._trained = True

        if isinstance(self.embedding, TFIDFEmbedding):
            embed_path = path.with_suffix(".embed.pkl")
            if embed_path.exists():
                self.embedding.load(embed_path)

        logger.info(f"Loaded router from {path}")


class HybridRouter:
    """Hybrid router combining learned model + keyword rules.

    Uses learned model as primary, with keyword boosting for
    high-confidence pattern matches. Best of both approaches.
    """

    def __init__(
        self,
        config: RouterTrainingConfig | None = None,
        keyword_boost: float = 0.3,
    ):
        self.config = config or RouterTrainingConfig()
        self.keyword_boost = keyword_boost
        self._learned: LearnedRouter | None = None
        self._keyword: "IntentClassifier" = None

    def load(self, path: Path | None = None) -> None:
        """Load learned model and initialize keyword classifier."""
        from .classifier import IntentClassifier

        self._learned = LearnedRouter(config=self.config)
        self._learned.load(path)
        self._keyword = IntentClassifier()

    def classify(self, query: str) -> ClassificationResult:
        """Classify using hybrid approach.

        1. Get learned model prediction
        2. Get keyword classifier prediction
        3. Boost learned confidence if keyword agrees
        4. Override if keyword has very high confidence
        """
        if not self._learned or not self._keyword:
            raise RuntimeError("Router not loaded. Call load() first.")

        learned = self._learned.classify(query)
        keyword = self._keyword.classify(query)

        # If keyword classifier has high confidence (>0.5), use it
        if keyword.confidence > 0.5 and keyword.intent != QueryIntent.GENERAL:
            return ClassificationResult(
                intent=keyword.intent,
                confidence=min(1.0, keyword.confidence + 0.1),
                matched_patterns=keyword.matched_patterns + ["keyword_override"],
            )

        # If both agree, boost confidence
        if learned.intent == keyword.intent:
            return ClassificationResult(
                intent=learned.intent,
                confidence=min(1.0, learned.confidence + self.keyword_boost),
                matched_patterns=learned.matched_patterns + keyword.matched_patterns,
            )

        # If keyword found patterns but learned disagrees, use keyword for debugging
        if keyword.intent == QueryIntent.DEBUGGING and keyword.matched_patterns:
            return ClassificationResult(
                intent=keyword.intent,
                confidence=keyword.confidence,
                matched_patterns=keyword.matched_patterns + ["keyword_debug_boost"],
            )

        # Otherwise use learned
        return learned


def generate_training_data() -> list[RoutingExample]:
    """Generate comprehensive training data for router."""
    examples = []

    # =========================================================================
    # DIN (Optimization) - ~100 examples
    # =========================================================================
    din_templates = [
        # Direct optimization requests
        "optimize this code", "optimize this routine", "optimize this loop",
        "optimize this for fewer cycles", "optimize this for size",
        "optimize this assembly", "optimize this 65816 code",
        # Performance keywords
        "make this faster", "make this more efficient", "make this smaller",
        "speed up this code", "speed up this routine",
        "improve performance", "improve efficiency",
        # Reduction keywords
        "reduce cycles", "reduce size", "reduce bytes",
        "reduce the cycle count", "reduce code size",
        "shrink this", "shrink this routine",
        # Specific techniques
        "use STZ instead", "use hardware multiplier", "unroll this loop",
        "eliminate redundant loads", "combine mode switches",
        "use 16-bit mode", "use MVN/MVP",
        # Questions
        "can this be faster?", "can this be smaller?",
        "is there a better way?", "how can I optimize this?",
        "what optimizations apply here?",
        # With code context
        "optimize: LDA STA patterns", "optimize: loop with CPX",
        "optimize this DMA setup", "optimize this sprite routine",
        "optimize this HDMA table", "optimize this OAM update",
    ]

    # Variations with prefixes/suffixes
    din_prefixes = ["", "please ", "can you ", "help me ", "I need to "]
    din_suffixes = ["", " for SNES", " for 65816", " in assembly"]

    for template in din_templates:
        for prefix in din_prefixes[:2]:  # Limit combinations
            for suffix in din_suffixes[:2]:
                examples.append(RoutingExample(
                    instruction=f"{prefix}{template}{suffix}",
                    expert=QueryIntent.OPTIMIZATION,
                    source="generated",
                ))

    # =========================================================================
    # NAYRU (Generation) - ~100 examples
    # =========================================================================
    nayru_templates = [
        # Write/create/generate
        "write code to", "write a routine to", "write assembly for",
        "create a function to", "create code for", "create a routine that",
        "generate code for", "generate a routine to", "generate assembly that",
        "implement", "implement a", "implement code for",
        # Specific tasks
        "read joypad", "read controller", "read input",
        "copy memory", "transfer data", "move bytes",
        "fade screen", "fade to black", "fade palette",
        "spawn sprite", "create sprite", "add sprite",
        "play sound", "trigger sfx", "play music",
        "save game", "load save", "write SRAM",
        "DMA transfer", "VRAM update", "tilemap write",
        "NMI handler", "interrupt routine", "VBLANK code",
        "collision detection", "hitbox check", "sprite collision",
        # Full phrases
        "write code to read the joypad",
        "create a sprite spawning routine",
        "implement DMA transfer to VRAM",
        "generate a delay loop",
        "write a memory copy function",
        "implement controller input handling",
        "create a palette loading routine",
    ]

    nayru_prefixes = ["", "please ", "I need ", "can you ", "help me "]

    for template in nayru_templates:
        for prefix in nayru_prefixes[:2]:
            examples.append(RoutingExample(
                instruction=f"{prefix}{template}",
                expert=QueryIntent.GENERATION,
                source="generated",
            ))

    # =========================================================================
    # FARORE (Debugging) - ~100 examples
    # =========================================================================
    farore_templates = [
        # Problem indicators
        "crashes", "freezes", "hangs", "locks up",
        "doesn't work", "not working", "broken",
        "wrong value", "wrong result", "incorrect",
        "corrupted", "glitched", "garbled",
        # Debug actions
        "debug", "fix", "diagnose", "troubleshoot",
        "find the bug", "find the error", "find the problem",
        "what's wrong", "why doesn't", "why does this",
        # Specific issues
        "DMA doesn't transfer", "sprite not appearing",
        "screen corruption", "palette wrong",
        "NMI timing issue", "race condition",
        "stack overflow", "infinite loop",
        "works in emulator not hardware",
        "save data corrupted", "SRAM not saving",
        # Full phrases
        "why is my code crashing?",
        "this DMA transfer doesn't work",
        "debug this sprite routine",
        "my code freezes the game",
        "find the bug in this NMI handler",
        "why is the screen corrupted?",
        "this loop never exits",
        "fix this stack overflow",
        "my palette isn't loading correctly",
        "why does this crash on real hardware?",
        "this code works in emulator but not console",
        "find the race condition in this code",
        "my save data gets corrupted",
        "this code hangs during VBLANK",
    ]

    for template in farore_templates:
        examples.append(RoutingExample(
            instruction=template,
            expert=QueryIntent.DEBUGGING,
            source="generated",
        ))

    # Add "why" questions
    why_templates = [
        "why does this crash", "why is this slow",
        "why doesn't this work", "why is the screen wrong",
        "why is the data corrupted", "why does this freeze",
    ]
    for template in why_templates:
        examples.append(RoutingExample(
            instruction=template,
            expert=QueryIntent.DEBUGGING,
            source="generated",
        ))

    # =========================================================================
    # VERAN/GENERAL (Explanation) - ~80 examples
    # =========================================================================
    veran_templates = [
        # Explain requests
        "explain", "explain this", "explain this code",
        "explain what this does", "explain how this works",
        "describe", "describe this", "describe this routine",
        # What questions
        "what does this do", "what does this code do",
        "what is this", "what are these registers",
        "what does LDA mean", "what does STA do",
        # How questions (understanding, not implementation)
        "how does this work", "how does this routine work",
        "how is this implemented",
        # Documentation
        "document this", "add comments", "annotate this code",
        # Specific explanations
        "explain DMA", "explain HDMA", "explain Mode 7",
        "explain the NMI handler", "explain sprite OAM",
        "explain REP and SEP", "explain addressing modes",
        "what does $2100 do", "what is VRAM",
        "explain the difference between BRL and JMP",
        # Full phrases
        "explain what this code does",
        "what does LDA $7E0020 mean?",
        "describe this DMA setup",
        "how does this HDMA table work?",
        "explain the Mode 7 matrix setup",
        "what are these registers for?",
        "describe the NMI handler structure",
        "explain how sprite OAM works",
    ]

    for template in veran_templates:
        examples.append(RoutingExample(
            instruction=template,
            expert=QueryIntent.GENERAL,  # Veran maps to GENERAL
            source="generated",
        ))

    # =========================================================================
    # Code examples with explicit intent
    # =========================================================================
    code_with_intent = [
        # Optimization - code that needs optimization
        ("LDA #$00\nSTA $10\nLDA #$00\nSTA $11", QueryIntent.OPTIMIZATION),
        ("SEP #$20\nSEP #$10", QueryIntent.OPTIMIZATION),
        ("LDA $10\nCLC\nADC #$01\nSTA $10", QueryIntent.OPTIMIZATION),
        ("CMP #$00\nBEQ label", QueryIntent.OPTIMIZATION),
        ("optimize this:\nLDX #$10\nloop:\nDEX\nCPX #$00\nBNE loop", QueryIntent.OPTIMIZATION),
        # Generation - tasks to implement
        ("write code to read $4218 joypad", QueryIntent.GENERATION),
        ("implement: wait for VBLANK", QueryIntent.GENERATION),
        ("create: copy 16 bytes from $1000 to $2000", QueryIntent.GENERATION),
        # Debugging - broken code
        ("this crashes: JSR $8000", QueryIntent.DEBUGGING),
        ("bug: LDA ($10) returns wrong value", QueryIntent.DEBUGGING),
        ("doesn't work: STA $2100", QueryIntent.DEBUGGING),
        ("freezes: loop: BRA loop", QueryIntent.DEBUGGING),
    ]

    for code, intent in code_with_intent:
        examples.append(RoutingExample(
            instruction=code,
            expert=intent,
            source="code_example",
        ))

    logger.info(f"Generated {len(examples)} training examples")
    return examples


def save_training_data(examples: list[RoutingExample], path: Path) -> None:
    """Save training examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def load_training_data(path: Path) -> list[RoutingExample]:
    """Load training examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(RoutingExample.from_dict(json.loads(line)))
    return examples


def train_router(
    output_path: Path | None = None,
    additional_data_path: Path | None = None,
) -> LearnedRouter:
    """Train a router from generated + optional additional data.

    Args:
        output_path: Where to save the trained model
        additional_data_path: Optional JSONL with more examples

    Returns:
        Trained LearnedRouter
    """
    # Generate base training data
    examples = generate_training_data()

    # Add additional data if provided
    if additional_data_path and additional_data_path.exists():
        additional = load_training_data(additional_data_path)
        examples.extend(additional)
        logger.info(f"Added {len(additional)} examples from {additional_data_path}")

    # Create and train router
    config = RouterTrainingConfig()
    if output_path:
        config.model_path = output_path

    router = LearnedRouter(config=config)
    metrics = router.train(examples)

    # Save
    router.save()

    print(f"\nTraining Results:")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Accuracy: {metrics['train_accuracy']:.1%}")
    print(f"  Loss: {metrics['final_loss']:.4f}")
    print(f"  Iterations: {metrics['n_iterations']}")
    print(f"  Model: {config.model_path}")

    return router


# CLI interface
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        router = train_router()

        # Test
        test_queries = [
            "optimize this loop",
            "write code for DMA",
            "why does this crash?",
            "explain this code",
        ]

        print("\nTest Classifications:")
        for query in test_queries:
            result = router.classify(query)
            print(f"  {result.intent.value:12} ({result.confidence:.2f}) | {query}")
    else:
        print("Usage: python -m afs.moe.router_trainer train")
