"""Training pipeline integrations using AFS Phase 2 features.

Provides:
- Session replay → training data extraction
- Router dataset generation from agent capabilities
- Pre-training freshness gate
"""

from .freshness_gate import (
    FreshnessGateConfig,
    FreshnessReport,
    MountReadiness,
    check_training_readiness,
)
from .session_source import (
    ExtractionResult,
    SessionExtractionConfig,
    extract_from_sessions,
    extract_samples_from_timeline,
)

__all__ = [
    "FreshnessGateConfig",
    "FreshnessReport",
    "MountReadiness",
    "check_training_readiness",
    "ExtractionResult",
    "SessionExtractionConfig",
    "extract_from_sessions",
    "extract_samples_from_timeline",
]
