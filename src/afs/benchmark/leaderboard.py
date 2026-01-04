"""Leaderboard system for tracking benchmark results over time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""

    model: str
    domain: str
    pass_rate: float
    total_items: int
    metrics: dict[str, float]
    timestamp: str
    version: str = "1.0"

    @classmethod
    def from_result(cls, result: "BenchmarkResult") -> "LeaderboardEntry":
        """Create from a BenchmarkResult."""
        from .base import BenchmarkResult
        return cls(
            model=result.model,
            domain=result.domain,
            pass_rate=result.pass_rate,
            total_items=result.total_items,
            metrics=result.metrics,
            timestamp=result.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "domain": self.domain,
            "pass_rate": self.pass_rate,
            "total_items": self.total_items,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeaderboardEntry":
        """Create from dictionary."""
        return cls(
            model=data["model"],
            domain=data["domain"],
            pass_rate=data["pass_rate"],
            total_items=data["total_items"],
            metrics=data.get("metrics", {}),
            timestamp=data["timestamp"],
            version=data.get("version", "1.0"),
        )


@dataclass
class ComparisonResult:
    """Result of comparing two models."""

    baseline_model: str
    candidate_model: str
    domain: str
    metric: str
    baseline_value: float
    candidate_value: float
    improvement: float
    is_significant: bool
    effect_size: float

    def summary(self) -> str:
        """Generate summary string."""
        direction = "better" if self.improvement > 0 else "worse"
        return (
            f"{self.candidate_model} vs {self.baseline_model} on {self.domain}/{self.metric}:\n"
            f"  Baseline: {self.baseline_value:.3f}\n"
            f"  Candidate: {self.candidate_value:.3f}\n"
            f"  Improvement: {self.improvement:+.1%} ({direction})\n"
            f"  Effect Size: {self.effect_size:.2f}"
        )


class LeaderboardManager:
    """Manages leaderboard tracking and history."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_path = self.results_dir / "leaderboard.json"
        self.history_path = self.results_dir / "history.jsonl"
        self._leaderboard: dict[str, dict[str, LeaderboardEntry]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing leaderboard."""
        if self.leaderboard_path.exists():
            with open(self.leaderboard_path) as f:
                data = json.load(f)
                for domain, entries in data.get("domains", {}).items():
                    self._leaderboard[domain] = {}
                    for model, entry_data in entries.items():
                        self._leaderboard[domain][model] = LeaderboardEntry.from_dict(entry_data)

    def _save(self) -> None:
        """Save leaderboard to disk."""
        data = {
            "updated": datetime.now().isoformat(),
            "domains": {
                domain: {
                    model: entry.to_dict()
                    for model, entry in entries.items()
                }
                for domain, entries in self._leaderboard.items()
            },
        }
        with open(self.leaderboard_path, "w") as f:
            json.dump(data, f, indent=2)

    def _append_history(self, entry: LeaderboardEntry) -> None:
        """Append entry to history log."""
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def update(self, result: "BenchmarkResult") -> bool:
        """Update leaderboard with new result.

        Returns True if this is a new best for the model/domain.
        """
        from .base import BenchmarkResult

        entry = LeaderboardEntry.from_result(result)

        if entry.domain not in self._leaderboard:
            self._leaderboard[entry.domain] = {}

        # Check if this is an improvement
        is_new_best = False
        existing = self._leaderboard[entry.domain].get(entry.model)

        if existing is None or entry.pass_rate > existing.pass_rate:
            is_new_best = True
            self._leaderboard[entry.domain][entry.model] = entry

        # Always log to history
        self._append_history(entry)

        # Save if new best
        if is_new_best:
            self._save()

        return is_new_best

    def get_leaders(self, domain: str, limit: int = 10) -> list[LeaderboardEntry]:
        """Get top models for a domain."""
        if domain not in self._leaderboard:
            return []

        entries = list(self._leaderboard[domain].values())
        entries.sort(key=lambda e: e.pass_rate, reverse=True)
        return entries[:limit]

    def get_all_leaders(self, limit_per_domain: int = 5) -> dict[str, list[LeaderboardEntry]]:
        """Get leaders for all domains."""
        return {
            domain: self.get_leaders(domain, limit_per_domain)
            for domain in self._leaderboard
        }

    def compare(
        self,
        baseline_model: str,
        candidate_model: str,
        domain: str,
        metric: str = "pass_rate",
    ) -> ComparisonResult | None:
        """Compare two models on a specific metric."""
        if domain not in self._leaderboard:
            return None

        baseline = self._leaderboard[domain].get(baseline_model)
        candidate = self._leaderboard[domain].get(candidate_model)

        if baseline is None or candidate is None:
            return None

        # Get metric values
        if metric == "pass_rate":
            baseline_value = baseline.pass_rate
            candidate_value = candidate.pass_rate
        else:
            baseline_value = baseline.metrics.get(metric, 0)
            candidate_value = candidate.metrics.get(metric, 0)

        # Calculate improvement
        if baseline_value == 0:
            improvement = 1.0 if candidate_value > 0 else 0.0
        else:
            improvement = (candidate_value - baseline_value) / baseline_value

        # Simple effect size (Cohen's d approximation)
        effect_size = abs(candidate_value - baseline_value)

        # Significance threshold (arbitrary, would need proper stats)
        is_significant = abs(improvement) > 0.05  # 5% improvement threshold

        return ComparisonResult(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            domain=domain,
            metric=metric,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            improvement=improvement,
            is_significant=is_significant,
            effect_size=effect_size,
        )

    def get_history(
        self,
        model: str | None = None,
        domain: str | None = None,
        days: int = 30,
    ) -> list[LeaderboardEntry]:
        """Get historical entries, optionally filtered."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        entries = []

        if not self.history_path.exists():
            return entries

        with open(self.history_path) as f:
            for line in f:
                if line.strip():
                    entry = LeaderboardEntry.from_dict(json.loads(line))

                    # Filter by time
                    try:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if entry_time < cutoff:
                            continue
                    except ValueError:
                        pass

                    # Filter by model/domain
                    if model and entry.model != model:
                        continue
                    if domain and entry.domain != domain:
                        continue

                    entries.append(entry)

        return entries

    def trend(
        self,
        model: str,
        domain: str,
        metric: str = "pass_rate",
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get trend data for a model/domain/metric."""
        history = self.get_history(model=model, domain=domain, days=days)

        trend_data = []
        for entry in history:
            if metric == "pass_rate":
                value = entry.pass_rate
            else:
                value = entry.metrics.get(metric, 0)

            trend_data.append({
                "timestamp": entry.timestamp,
                "value": value,
            })

        # Sort by timestamp
        trend_data.sort(key=lambda x: x["timestamp"])
        return trend_data

    def generate_report(self) -> str:
        """Generate a markdown leaderboard report."""
        lines = [
            "# AFS Model Leaderboard",
            "",
            f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        for domain in sorted(self._leaderboard.keys()):
            lines.extend([
                f"## {domain.title()}",
                "",
                "| Rank | Model | Pass Rate | Items | Last Updated |",
                "|------|-------|-----------|-------|--------------|",
            ])

            leaders = self.get_leaders(domain, limit=10)
            for i, entry in enumerate(leaders, 1):
                # Parse timestamp for display
                try:
                    dt = datetime.fromisoformat(entry.timestamp)
                    updated = dt.strftime("%Y-%m-%d")
                except ValueError:
                    updated = entry.timestamp[:10]

                lines.append(
                    f"| {i} | {entry.model} | {entry.pass_rate:.1%} | "
                    f"{entry.total_items} | {updated} |"
                )

            lines.append("")

        return "\n".join(lines)
