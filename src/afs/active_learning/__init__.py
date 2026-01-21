"""Active learning utilities for iterative training data improvement.

Provides tools for:
- Uncertainty-based sample selection
- Curriculum learning (easy -> hard progression)
- Priority queue for annotation workflow
- Iterative model improvement
"""

from .priority_queue import (
    PriorityQueue,
    QueueItem,
    create_queue,
    get_next_batch,
)
from .sampler import (
    CurriculumManager,
    CurriculumStage,
    UncertaintySampler,
    get_curriculum_samples,
    sample_by_uncertainty,
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
