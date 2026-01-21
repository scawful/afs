"""Evaluation runner for MoE router and expert models."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..router import MoERouter

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"


@dataclass
class EvalExample:
    """A single evaluation example."""

    id: str
    query: str
    expected_intent: str
    expected_contains: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    """Result of evaluating a single example."""

    example: EvalExample
    actual_intent: str
    confidence: float
    response: str
    response_time_ms: float
    intent_correct: bool
    contains_score: float  # 0.0 to 1.0, fraction of expected_contains found
    tokens_generated: int = 0


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    total_examples: int
    intent_accuracy: float
    avg_confidence: float
    avg_contains_score: float
    avg_response_time_ms: float
    by_intent: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_examples": self.total_examples,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_contains_score": round(self.avg_contains_score, 4),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "by_intent": self.by_intent,
        }


class EvalRunner:
    """Runs evaluations on MoE router and expert models."""

    def __init__(
        self,
        router: MoERouter | None = None,
        datasets_dir: Path | None = None,
    ):
        """Initialize eval runner.

        Args:
            router: MoE router instance. If None, creates default.
            datasets_dir: Directory containing eval datasets.
        """
        self.router = router or MoERouter()
        self.datasets_dir = datasets_dir or DATASETS_DIR

    def load_dataset(self, intent: str) -> list[EvalExample]:
        """Load evaluation dataset for an intent."""
        dataset_file = self.datasets_dir / f"{intent}_eval.jsonl"
        if not dataset_file.exists():
            logger.warning(f"Dataset not found: {dataset_file}")
            return []

        examples = []
        with open(dataset_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(EvalExample(
                        id=data["id"],
                        query=data["query"],
                        expected_intent=data["expected_intent"],
                        expected_contains=data.get("expected_contains", []),
                        notes=data.get("notes", ""),
                    ))

        logger.info(f"Loaded {len(examples)} examples for {intent}")
        return examples

    def load_all_datasets(self) -> list[EvalExample]:
        """Load all evaluation datasets."""
        all_examples = []
        for intent in ["optimization", "generation", "debugging"]:
            all_examples.extend(self.load_dataset(intent))
        return all_examples

    def eval_routing_only(self, examples: list[EvalExample]) -> list[EvalResult]:
        """Evaluate routing decisions only (no generation)."""
        results = []

        for example in examples:
            start = time.perf_counter()
            decision = self.router.route(example.query)
            elapsed_ms = (time.perf_counter() - start) * 1000

            actual_intent = decision.classification.intent.value

            results.append(EvalResult(
                example=example,
                actual_intent=actual_intent,
                confidence=decision.classification.confidence,
                response="",
                response_time_ms=elapsed_ms,
                intent_correct=actual_intent == example.expected_intent,
                contains_score=0.0,
            ))

        return results

    async def eval_with_generation(
        self,
        examples: list[EvalExample],
        use_rag: bool = False,
    ) -> list[EvalResult]:
        """Evaluate routing and generation."""
        results = []

        async with MoERouter(self.router.config) as router:
            for example in examples:
                start = time.perf_counter()

                # Route
                decision = router.route(example.query)
                actual_intent = decision.classification.intent.value

                # Generate
                try:
                    gen_result = await router.generate(
                        example.query,
                        use_rag=use_rag,
                    )
                    response = gen_result.content
                    tokens = gen_result.tokens_generated
                except Exception as e:
                    logger.error(f"Generation failed for {example.id}: {e}")
                    response = f"ERROR: {e}"
                    tokens = 0

                elapsed_ms = (time.perf_counter() - start) * 1000

                # Score contains
                contains_score = self._score_contains(
                    response, example.expected_contains
                )

                results.append(EvalResult(
                    example=example,
                    actual_intent=actual_intent,
                    confidence=decision.classification.confidence,
                    response=response,
                    response_time_ms=elapsed_ms,
                    intent_correct=actual_intent == example.expected_intent,
                    contains_score=contains_score,
                    tokens_generated=tokens,
                ))

                # Progress
                logger.info(
                    f"[{example.id}] {actual_intent} "
                    f"({'OK' if actual_intent == example.expected_intent else 'WRONG'}) "
                    f"contains={contains_score:.2f} time={elapsed_ms:.0f}ms"
                )

        return results

    def _score_contains(self, response: str, expected: list[str]) -> float:
        """Score how many expected patterns are in the response."""
        if not expected:
            return 1.0  # No requirements = pass

        response_upper = response.upper()
        found = sum(1 for pattern in expected if pattern.upper() in response_upper)
        return found / len(expected)

    def compute_metrics(self, results: list[EvalResult]) -> EvalMetrics:
        """Compute aggregate metrics from results."""
        if not results:
            return EvalMetrics(
                total_examples=0,
                intent_accuracy=0.0,
                avg_confidence=0.0,
                avg_contains_score=0.0,
                avg_response_time_ms=0.0,
            )

        total = len(results)
        correct = sum(1 for r in results if r.intent_correct)

        # By intent breakdown
        by_intent: dict[str, dict] = {}
        for intent in ["optimization", "generation", "debugging"]:
            intent_results = [r for r in results if r.example.expected_intent == intent]
            if intent_results:
                intent_correct = sum(1 for r in intent_results if r.intent_correct)
                by_intent[intent] = {
                    "total": len(intent_results),
                    "correct": intent_correct,
                    "accuracy": round(intent_correct / len(intent_results), 4),
                    "avg_contains": round(
                        sum(r.contains_score for r in intent_results) / len(intent_results),
                        4
                    ),
                    "avg_time_ms": round(
                        sum(r.response_time_ms for r in intent_results) / len(intent_results),
                        2
                    ),
                }

        return EvalMetrics(
            total_examples=total,
            intent_accuracy=correct / total,
            avg_confidence=sum(r.confidence for r in results) / total,
            avg_contains_score=sum(r.contains_score for r in results) / total,
            avg_response_time_ms=sum(r.response_time_ms for r in results) / total,
            by_intent=by_intent,
        )

    def print_results(
        self,
        results: list[EvalResult],
        metrics: EvalMetrics,
        verbose: bool = False,
    ) -> None:
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("MoE EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nTotal examples: {metrics.total_examples}")
        print(f"Intent accuracy: {metrics.intent_accuracy:.1%}")
        print(f"Avg confidence: {metrics.avg_confidence:.2f}")
        print(f"Avg contains score: {metrics.avg_contains_score:.1%}")
        print(f"Avg response time: {metrics.avg_response_time_ms:.0f}ms")

        print("\n--- By Intent ---")
        for intent, data in metrics.by_intent.items():
            print(f"\n{intent.upper()}:")
            print(f"  Accuracy: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
            print(f"  Contains: {data['avg_contains']:.1%}")
            print(f"  Avg time: {data['avg_time_ms']:.0f}ms")

        # Show failures
        failures = [r for r in results if not r.intent_correct]
        if failures:
            print("\n--- Routing Failures ---")
            for r in failures:
                print(f"  [{r.example.id}] Expected {r.example.expected_intent}, got {r.actual_intent}")
                if verbose:
                    print(f"    Query: {r.example.query[:60]}...")

        # Show low contains scores
        low_contains = [r for r in results if r.contains_score < 0.5 and r.example.expected_contains]
        if low_contains and verbose:
            print("\n--- Low Contains Scores ---")
            for r in low_contains[:5]:
                print(f"  [{r.example.id}] {r.contains_score:.1%}")
                print(f"    Expected: {r.example.expected_contains}")
                print(f"    Response: {r.response[:100]}...")

        print("\n" + "=" * 60)


def run_routing_eval() -> EvalMetrics:
    """Run routing-only evaluation."""
    runner = EvalRunner()
    examples = runner.load_all_datasets()
    results = runner.eval_routing_only(examples)
    metrics = runner.compute_metrics(results)
    runner.print_results(results, metrics)
    return metrics


async def run_full_eval(use_rag: bool = False) -> EvalMetrics:
    """Run full evaluation with generation."""
    runner = EvalRunner()
    examples = runner.load_all_datasets()
    results = await runner.eval_with_generation(examples, use_rag=use_rag)
    metrics = runner.compute_metrics(results)
    runner.print_results(results, metrics, verbose=True)
    return metrics
