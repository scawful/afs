"""Evaluation utilities for AFS training data and models.

Provides tools for:
- Automated evaluation metrics (syntax, quality, diversity)
- Human evaluation workflow management
- A/B testing and comparison utilities
"""

from .harness import (
    ComparisonResult,
    EvaluationHarness,
    EvaluationResult,
    compare_datasets,
    evaluate_samples,
)
from .human import (
    HumanEvalBatch,
    HumanEvalTask,
    HumanEvaluationManager,
    create_eval_batch,
    import_eval_results,
)
from .semantic_eval import (
    CPUState,
    ExecutionResult,
    SemanticEvalConfig,
    SemanticEvaluator,
    SemanticScore,
    create_semantic_evaluator,
)

__all__ = [
    # Harness
    "EvaluationResult",
    "ComparisonResult",
    "EvaluationHarness",
    "evaluate_samples",
    "compare_datasets",
    # Human evaluation
    "HumanEvalTask",
    "HumanEvalBatch",
    "HumanEvaluationManager",
    "create_eval_batch",
    "import_eval_results",
    # Semantic evaluation
    "CPUState",
    "ExecutionResult",
    "SemanticScore",
    "SemanticEvalConfig",
    "SemanticEvaluator",
    "create_semantic_evaluator",
]
