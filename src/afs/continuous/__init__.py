"""Continuous learning system for self-improving models.

Feeds production usage back into training through:
- Usage logging to SQLite database
- Quality-scored training data generation
- Automatic retraining triggers
- A/B testing of model versions
- Full feedback loop integration

Built on top of afs.feedback for production inference tracking.
"""

from .logger import UsageLogger, UsageRecord, UsageDatabase
from .generator import TrainingDataGenerator, DataGeneratorConfig, GenerationResult
from .trigger import (
    RetrainTrigger,
    TriggerConfig,
    TriggerType,
    TriggerResult,
    AutoRetrainer,
)
from .ab_test import (
    ABTestManager,
    ABTestConfig,
    ABTestResult,
    ModelVersion,
    ModelStatus,
    TrafficSplit,
)
from .loop import ContinuousLearningLoop, LoopConfig, LoopStatus

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
