"""Model comparison and evaluation framework.

This package provides tools for comprehensive side-by-side evaluation of
different model versions and variants.
"""

from .framework import (
    ComparisonMode,
    ComparisonReport,
    ComparisonResult,
    ModelComparator,
    ModelResponse,
    ResponseScorer,
    ScoreDimension,
    ScoredResponse,
    BasicScorer,
    StatisticalTester,
)

__all__ = [
    "ComparisonMode",
    "ComparisonReport",
    "ComparisonResult",
    "ModelComparator",
    "ModelResponse",
    "ResponseScorer",
    "ScoreDimension",
    "ScoredResponse",
    "BasicScorer",
    "StatisticalTester",
]
