"""Active learning utilities for iterative training data improvement.

Provides tools for:
- Uncertainty-based sample selection
- Curriculum learning (easy -> hard progression)
- Priority queue for annotation workflow
- Iterative model improvement
"""

from .sampler import (
    UncertaintySampler,
    CurriculumManager,
    CurriculumStage,
    sample_by_uncertainty,
    get_curriculum_samples,
)
from .priority_queue import (
    PriorityQueue,
    QueueItem,
    create_queue,
    get_next_batch,
)

__all__ = [
    # Samplers
    "UncertaintySampler",
    "CurriculumManager",
    "CurriculumStage",
    "sample_by_uncertainty",
    "get_curriculum_samples",
    # Priority Queue
    "PriorityQueue",
    "QueueItem",
    "create_queue",
    "get_next_batch",
]
