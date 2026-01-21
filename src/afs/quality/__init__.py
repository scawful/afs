"""Training data quality analysis tools.

Provides comprehensive analysis and improvement suggestions for training datasets:
- Dataset statistics (size, diversity, distributions)
- Quality metrics (clarity, correctness, code quality)
- Bias detection (gender, cultural, technical)
- Duplicate and anomaly detection
- Actionable improvement recommendations

This module integrates with the training pipeline to ensure datasets
meet quality standards before being used for fine-tuning.

See also:
- afs.training: Training utilities and data conversion
- afs.training.scoring: Quality scoring implementation
- afs.generators: Training data generation
"""

from .analyzer import (
    DatasetAnalyzer,
    DatasetStatistics,
    QualityReport,
    analyze_dataset,
)
from .metrics import (
    QualityMetrics,
    InstructionClarity,
    OutputCorrectness,
    DuplicateDetector,
    AnomalyDetector,
)
from .bias import (
    BiasAnalyzer,
    BiasReport,
    GenderBiasDetector,
    CulturalBiasDetector,
    TechnicalBiasDetector,
    detect_biases,
)

__all__ = [
    # Analyzer
    "DatasetAnalyzer",
    "DatasetStatistics",
    "QualityReport",
    "analyze_dataset",
    # Metrics
    "QualityMetrics",
    "InstructionClarity",
    "OutputCorrectness",
    "DuplicateDetector",
    "AnomalyDetector",
    # Bias
    "BiasAnalyzer",
    "BiasReport",
    "GenderBiasDetector",
    "CulturalBiasDetector",
    "TechnicalBiasDetector",
    "detect_biases",
]
