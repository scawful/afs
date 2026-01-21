"""Priority queue for annotation workflow.

Manages a persistent queue of samples prioritized by:
- Uncertainty (ELECTRA score near 0.5)
- Quality score
- Domain coverage
- Recency
"""

from __future__ import annotations

import heapq
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afs.generators.base import TrainingSample
    from afs.training.scoring import QualityScorer


class ItemStatus(str, Enum):
    """Status of a queue item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


@dataclass
class QueueItem:
    """An item in the priority queue."""

    item_id: str
    sample_id: str
    priority: float  # Higher = more important

    # Sample data
    instruction: str
    output: str
    domain: str

    # Scores
    quality_score: float = 0.0
    electra_score: float = 0.0
    uncertainty: float = 0.0

    # Status
    status: ItemStatus = ItemStatus.PENDING
    added_at: str = ""
    reviewed_at: str | None = None
    reviewer: str | None = None

    # Review result
    human_rating: float | None = None
    review_notes: str = ""

    def __lt__(self, other: QueueItem) -> bool:
        """Compare for heap ordering (higher priority = comes first)."""
        return self.priority > other.priority

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "sample_id": self.sample_id,
            "priority": self.priority,
            "instruction": self.instruction,
            "output": self.output,
            "domain": self.domain,
            "quality_score": self.quality_score,
            "electra_score": self.electra_score,
            "uncertainty": self.uncertainty,
            "status": self.status.value,
            "added_at": self.added_at,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "human_rating": self.human_rating,
            "review_notes": self.review_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueItem:
        return cls(
            item_id=data["item_id"],
            sample_id=data["sample_id"],
            priority=data["priority"],
            instruction=data["instruction"],
            output=data["output"],
            domain=data.get("domain", "unknown"),
            quality_score=data.get("quality_score", 0.0),
            electra_score=data.get("electra_score", 0.0),
            uncertainty=data.get("uncertainty", 0.0),
            status=ItemStatus(data.get("status", "pending")),
            added_at=data.get("added_at", ""),
            reviewed_at=data.get("reviewed_at"),
            reviewer=data.get("reviewer"),
            human_rating=data.get("human_rating"),
            review_notes=data.get("review_notes", ""),
        )

    @classmethod
    def from_sample(
        cls,
        sample: TrainingSample,
        priority: float | None = None,
    ) -> QueueItem:
        """Create queue item from training sample."""
        electra = sample._metadata.get("quality_components", {}).get("electra", 0.5)
        uncertainty = 1.0 - abs(electra - 0.5)

        # Default priority: uncertainty + quality
        if priority is None:
            priority = uncertainty * 0.6 + sample.quality_score * 0.4

        return cls(
            item_id=str(uuid.uuid4()),
            sample_id=sample.sample_id,
            priority=priority,
            instruction=sample.instruction,
            output=sample.output,
            domain=sample.domain,
            quality_score=sample.quality_score,
            electra_score=electra,
            uncertainty=uncertainty,
            added_at=datetime.now().isoformat(),
        )


@dataclass
class PriorityQueueConfig:
    """Configuration for priority queue."""

    # Priority weights
    uncertainty_weight: float = 0.5
    quality_weight: float = 0.3
    recency_weight: float = 0.1
    domain_balance_weight: float = 0.1

    # Queue limits
    max_pending: int = 1000
    max_in_progress: int = 100


class PriorityQueue:
    """Priority queue for annotation workflow.

    Maintains a persistent heap of samples ordered by priority.
    Supports adding, retrieving, and marking items as reviewed.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        config: PriorityQueueConfig | None = None,
    ):
        """Initialize queue.

        Args:
            storage_path: Path for persistence
            config: Queue configuration
        """
        self.storage_path = storage_path
        self.config = config or PriorityQueueConfig()

        self._heap: list[QueueItem] = []
        self._items: dict[str, QueueItem] = {}  # item_id -> item
        self._domain_counts: dict[str, int] = {}

        if storage_path and storage_path.exists():
            self._load()

    def add(
        self,
        samples: list[TrainingSample],
        scorer: QualityScorer | None = None,
    ) -> int:
        """Add samples to the queue.

        Args:
            samples: Samples to add
            scorer: Optional scorer to compute priorities

        Returns:
            Number of items added
        """
        # Score if needed
        if scorer:
            scorer.score_batch(samples, update_samples=True)

        added = 0
        for sample in samples:
            # Skip if already in queue
            if any(item.sample_id == sample.sample_id for item in self._items.values()):
                continue

            # Compute priority
            priority = self._compute_priority(sample)

            item = QueueItem.from_sample(sample, priority=priority)
            self._items[item.item_id] = item
            heapq.heappush(self._heap, item)

            self._domain_counts[sample.domain] = self._domain_counts.get(sample.domain, 0) + 1
            added += 1

        # Enforce max pending
        while len(self._heap) > self.config.max_pending:
            removed = heapq.heappop(self._heap)
            del self._items[removed.item_id]
            self._domain_counts[removed.domain] -= 1

        self._save()
        return added

    def _compute_priority(self, sample: TrainingSample) -> float:
        """Compute priority for a sample."""
        c = self.config

        # Uncertainty component
        electra = sample._metadata.get("quality_components", {}).get("electra", 0.5)
        uncertainty = 1.0 - abs(electra - 0.5)

        # Quality component (prefer mid-quality for review)
        quality = sample.quality_score

        # Domain balance (boost underrepresented domains)
        domain_count = self._domain_counts.get(sample.domain, 0)
        total = sum(self._domain_counts.values()) or 1
        domain_ratio = domain_count / total
        domain_balance = 1.0 - domain_ratio  # Higher for rare domains

        # Combine
        priority = (
            c.uncertainty_weight * uncertainty
            + c.quality_weight * quality
            + c.domain_balance_weight * domain_balance
        )

        return priority

    def get_batch(self, n: int) -> list[QueueItem]:
        """Get next n items for review.

        Items are marked as in_progress.

        Args:
            n: Number of items to get

        Returns:
            List of queue items
        """
        result = []
        pending_items = [
            item for item in self._heap
            if item.status == ItemStatus.PENDING
        ]

        # Re-heapify to get highest priority pending items
        heapq.heapify(pending_items)

        for _ in range(min(n, len(pending_items))):
            if not pending_items:
                break

            item = heapq.heappop(pending_items)
            item.status = ItemStatus.IN_PROGRESS
            result.append(item)

        self._save()
        return result

    def mark_reviewed(
        self,
        item_ids: list[str],
        ratings: dict[str, float] | None = None,
        reviewer: str | None = None,
    ) -> int:
        """Mark items as reviewed.

        Args:
            item_ids: IDs of items to mark
            ratings: Optional dict of item_id -> human rating
            reviewer: Optional reviewer name

        Returns:
            Number of items updated
        """
        ratings = ratings or {}
        updated = 0

        for item_id in item_ids:
            if item_id in self._items:
                item = self._items[item_id]
                item.status = ItemStatus.REVIEWED
                item.reviewed_at = datetime.now().isoformat()
                item.reviewer = reviewer

                if item_id in ratings:
                    item.human_rating = ratings[item_id]

                updated += 1

        self._save()
        return updated

    def mark_rejected(self, item_ids: list[str]) -> int:
        """Mark items as rejected (low quality, skip).

        Args:
            item_ids: IDs of items to reject

        Returns:
            Number of items updated
        """
        updated = 0

        for item_id in item_ids:
            if item_id in self._items:
                self._items[item_id].status = ItemStatus.REJECTED
                self._items[item_id].reviewed_at = datetime.now().isoformat()
                updated += 1

        self._save()
        return updated

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        by_status = {status.value: 0 for status in ItemStatus}
        for item in self._items.values():
            by_status[item.status.value] += 1

        ratings = [
            item.human_rating for item in self._items.values()
            if item.human_rating is not None
        ]

        return {
            "total_items": len(self._items),
            "by_status": by_status,
            "by_domain": dict(self._domain_counts),
            "ratings": {
                "count": len(ratings),
                "mean": sum(ratings) / len(ratings) if ratings else 0,
                "min": min(ratings) if ratings else 0,
                "max": max(ratings) if ratings else 0,
            },
        }

    def clear_reviewed(self) -> int:
        """Remove all reviewed/rejected items.

        Returns:
            Number of items removed
        """
        to_remove = [
            item_id for item_id, item in self._items.items()
            if item.status in (ItemStatus.REVIEWED, ItemStatus.REJECTED)
        ]

        for item_id in to_remove:
            item = self._items.pop(item_id)
            self._domain_counts[item.domain] -= 1
            if self._domain_counts[item.domain] <= 0:
                del self._domain_counts[item.domain]

        # Rebuild heap
        self._heap = list(self._items.values())
        heapq.heapify(self._heap)

        self._save()
        return len(to_remove)

    def get_reviewed_samples(self) -> list[dict[str, Any]]:
        """Get all reviewed samples with their ratings.

        Returns:
            List of dicts with sample_id, human_rating, etc.
        """
        return [
            {
                "sample_id": item.sample_id,
                "human_rating": item.human_rating,
                "review_notes": item.review_notes,
                "reviewer": item.reviewer,
                "reviewed_at": item.reviewed_at,
            }
            for item in self._items.values()
            if item.status == ItemStatus.REVIEWED
        ]

    def _save(self) -> None:
        """Save queue to storage."""
        if not self.storage_path:
            return

        data = {
            "items": [item.to_dict() for item in self._items.values()],
            "domain_counts": self._domain_counts,
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load queue from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self._items = {}
        for item_data in data.get("items", []):
            item = QueueItem.from_dict(item_data)
            self._items[item.item_id] = item

        self._domain_counts = data.get("domain_counts", {})

        # Rebuild heap
        self._heap = list(self._items.values())
        heapq.heapify(self._heap)


def create_queue(
    samples: list[TrainingSample],
    storage_path: Path | None = None,
    scorer: QualityScorer | None = None,
) -> PriorityQueue:
    """Create a new priority queue from samples.

    Args:
        samples: Initial samples
        storage_path: Optional persistence path
        scorer: Optional quality scorer

    Returns:
        Initialized PriorityQueue
    """
    queue = PriorityQueue(storage_path=storage_path)
    queue.add(samples, scorer=scorer)
    return queue


def get_next_batch(
    queue_path: Path,
    n: int = 10,
) -> list[QueueItem]:
    """Get next batch from an existing queue.

    Args:
        queue_path: Path to queue storage
        n: Number of items to get

    Returns:
        List of queue items
    """
    queue = PriorityQueue(storage_path=queue_path)
    return queue.get_batch(n)
