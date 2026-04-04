"""Pre-training freshness gate.

Checks that knowledge/context backing training data is sufficiently
fresh before allowing a training run to proceed. Prevents training
on stale data that may have drifted from the current codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from afs.context_index import ContextSQLiteIndex
from afs.manager import AFSManager
from afs.models import MountType

logger = logging.getLogger(__name__)

# Mounts that typically back training data
DEFAULT_TRAINING_MOUNTS = [
    MountType.KNOWLEDGE,
    MountType.TOOLS,
    MountType.HISTORY,
    MountType.SCRATCHPAD,
]


@dataclass
class FreshnessGateConfig:
    """Configuration for the pre-training freshness gate."""

    min_score: float = 0.3
    decay_hours: float = 168.0  # 1 week
    mount_types: list[MountType] = field(
        default_factory=lambda: list(DEFAULT_TRAINING_MOUNTS)
    )
    block_on_failure: bool = True


@dataclass
class MountReadiness:
    """Readiness status for a single mount."""

    mount_type: str
    score: float
    status: str  # "ready", "stale", "missing"
    file_count: int = 0
    stale_files: int = 0


@dataclass
class FreshnessReport:
    """Report from the pre-training freshness gate."""

    ready: bool
    overall_score: float
    mounts: list[MountReadiness]
    blocked_mounts: list[str]
    warnings: list[str]
    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "overall_score": round(self.overall_score, 3),
            "mounts": [
                {
                    "mount_type": m.mount_type,
                    "score": round(m.score, 3),
                    "status": m.status,
                    "file_count": m.file_count,
                    "stale_files": m.stale_files,
                }
                for m in self.mounts
            ],
            "blocked_mounts": self.blocked_mounts,
            "warnings": self.warnings,
            "config": self.config,
            "timestamp": self.timestamp,
        }


def check_training_readiness(
    context_path: Path,
    *,
    config: FreshnessGateConfig | None = None,
    afs_config: Any = None,
) -> FreshnessReport:
    """Check whether context is fresh enough for training.

    Runs freshness_scores() on the context index and evaluates each
    mount against the configured minimum score threshold.

    Returns a FreshnessReport indicating whether training should proceed.
    """
    cfg = config or FreshnessGateConfig()

    manager = AFSManager(config=afs_config)
    index = ContextSQLiteIndex(manager, context_path)

    scores = index.freshness_scores(
        mount_types=cfg.mount_types,
        decay_hours=cfg.decay_hours,
        threshold=0.0,
    )

    mount_scores = scores.get("mount_scores", {})
    file_details = scores.get("files", {})

    mounts: list[MountReadiness] = []
    blocked: list[str] = []
    warnings: list[str] = []
    score_values: list[float] = []

    for mount_type in cfg.mount_types:
        key = mount_type.value
        score = mount_scores.get(key)

        if score is None:
            mounts.append(MountReadiness(
                mount_type=key,
                score=0.0,
                status="missing",
            ))
            warnings.append(f"Mount '{key}' not found in index")
            continue

        files = file_details.get(key, [])
        file_count = len(files)
        stale_count = sum(
            1 for f in files
            if f.get("score", 1.0) < cfg.min_score
        )

        if score < cfg.min_score:
            status = "stale"
            blocked.append(key)
        else:
            status = "ready"

        mounts.append(MountReadiness(
            mount_type=key,
            score=score,
            status=status,
            file_count=file_count,
            stale_files=stale_count,
        ))
        score_values.append(score)

    overall = sum(score_values) / len(score_values) if score_values else 0.0
    ready = len(blocked) == 0 if cfg.block_on_failure else True

    if blocked:
        warnings.append(
            f"Stale mounts blocking training: {', '.join(blocked)}. "
            f"Run 'afs index rebuild --path <workspace>' to refresh."
        )

    return FreshnessReport(
        ready=ready,
        overall_score=overall,
        mounts=mounts,
        blocked_mounts=blocked,
        warnings=warnings,
        config={
            "min_score": cfg.min_score,
            "decay_hours": cfg.decay_hours,
            "block_on_failure": cfg.block_on_failure,
            "mount_types": [m.value for m in cfg.mount_types],
        },
    )
