"""Continuous learning system for self-improving models.

Feeds production usage back into training through:
- Usage logging to SQLite database
- Quality-scored training data generation
- Automatic retraining triggers
- A/B testing of model versions
- Full feedback loop integration

Built on top of afs.feedback for production inference tracking.
"""

from .ab_test import (
    ABTestConfig,
    ABTestManager,
    ABTestResult,
    ModelStatus,
    ModelVersion,
    TrafficSplit,
)
from .generator import DataGeneratorConfig, GenerationResult, TrainingDataGenerator
from .logger import UsageDatabase, UsageLogger, UsageRecord
from .loop import ContinuousLearningLoop, LoopConfig, LoopStatus
from .trigger import (
    AutoRetrainer,
    RetrainTrigger,
    TriggerConfig,
    TriggerResult,
    TriggerType,
)

__all__ = [
    # Usage tracking
    "UsageLogger",
    "UsageRecord",
    "UsageDatabase",
    # Data generation
    "TrainingDataGenerator",
    "DataGeneratorConfig",
    "GenerationResult",
    # Retraining triggers
    "RetrainTrigger",
    "TriggerConfig",
    "TriggerType",
    "TriggerResult",
    "AutoRetrainer",
    # A/B testing
    "ABTestManager",
    "ABTestConfig",
    "ABTestResult",
    "ModelVersion",
    "ModelStatus",
    "TrafficSplit",
    # Main loop
    "ContinuousLearningLoop",
    "LoopConfig",
    "LoopStatus",
]
