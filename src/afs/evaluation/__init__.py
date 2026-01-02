"""Evaluation utilities for AFS training data and models.

Provides tools for:
- Automated evaluation metrics (syntax, quality, diversity)
- Human evaluation workflow management
- A/B testing and comparison utilities
"""

from .harness import (
    EvaluationResult,
    ComparisonResult,
    EvaluationHarness,
    evaluate_samples,
    compare_datasets,
)
from .human import (
    HumanEvalTask,
    HumanEvalBatch,
    HumanEvaluationManager,
    create_eval_batch,
    import_eval_results,
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
]
