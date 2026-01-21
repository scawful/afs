"""Cost optimization and analysis system for training.

This module provides tools to minimize training costs while maximizing model quality.

Features:
- GPU price tracking from vast.ai
- Cost per sample/epoch analysis
- Optimization recommendations
- Budget management and alerts
- Cost forecasting
"""

from .analyzer import BudgetAlert, CostAnalyzer, TrainingCostReport, TrainingMetrics
from .optimizer import CostOptimizer, OptimizationRecommendation
from .tracker import GPUPrice, GPUPriceTracker, PriceAlert, PriceHistory

__all__ = [
    "GPUPrice",
    "GPUPriceTracker",
    "PriceHistory",
    "PriceAlert",
    "TrainingMetrics",
    "TrainingCostReport",
    "BudgetAlert",
    "CostAnalyzer",
    "CostOptimizer",
    "OptimizationRecommendation",
]
