"""Continuous monitoring daemon for AFS health.

Runs health checks on a schedule and alerts on score drops.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

from afs.logging_config import get_logger

from .enhanced_checks import EnhancedHealthChecker, HealthCheckLevel

logger = get_logger(__name__)


class HealthMonitoringDaemon:
    """Continuous health monitoring with alerting."""

    def __init__(
        self,
        context_root: Path | None = None,
        check_interval_s: int = 60,
        alert_threshold: float = 0.1,
        auto_heal: bool = False,
    ):
        """Initialize monitoring daemon.

        Args:
            context_root: Health logs directory
            check_interval_s: Interval between checks (seconds)
            alert_threshold: Alert on score drop >N (0.0-1.0)
            auto_heal: Enable automatic healing
        """
        self.context_root = context_root or Path.home() / ".context"
        self.health_dir = self.context_root / "health"
        self.health_dir.mkdir(parents=True, exist_ok=True)

        self.check_interval_s = check_interval_s
        self.alert_threshold = alert_threshold
        self.auto_heal = auto_heal
        self.checker = EnhancedHealthChecker(context_root=context_root)

        self.running = False
        self.last_score: float | None = None
        self.last_check: datetime | None = None
        self.check_history: list[tuple[datetime, float]] = []

    async def start(
        self,
        check_level: HealthCheckLevel = HealthCheckLevel.STANDARD,
        duration_s: int | None = None,
    ) -> None:
        """Start continuous monitoring.

        Args:
            check_level: Depth of health checks
            duration_s: Run for N seconds (None = indefinite)
        """
        self.running = True
        logger.info(
            f"Health monitoring daemon starting (interval={self.check_interval_s}s, level={check_level.value})",
            extra={"interval": self.check_interval_s, "level": check_level.value},
        )

        start_time = datetime.now()
        check_count = 0

        try:
            while self.running:
                # Check duration limit
                if duration_s and (datetime.now() - start_time).total_seconds() > duration_s:
                    logger.info(f"Health monitoring duration limit reached ({duration_s}s)")
                    break

                # Run health check
                check_start = time.time()
                result = self.checker.check(
                    level=check_level,
                    auto_heal=self.auto_heal,
                    save_report=True,
                )
                check_count += 1

                # Track score
                previous_score = self.last_score
                self.last_score = result.overall_score
                self.last_check = result.timestamp
                self.check_history.append((result.timestamp, result.overall_score))

                # Keep only last 24 hours of history (60-second intervals)
                cutoff = datetime.now() - timedelta(hours=24)
                self.check_history = [
                    (ts, score)
                    for ts, score in self.check_history
                    if ts > cutoff
                ]

                # Alert on significant drops
                if (
                    previous_score is not None
                    and previous_score - self.last_score > self.alert_threshold
                ):
                    await self._alert_score_drop(previous_score, self.last_score)

                logger.info(
                    f"Health check #{check_count}: {result.overall_status.value.upper()} "
                    f"({result.overall_score:.2f}) [{(time.time() - check_start):.2f}s]",
                    extra={
                        "check_number": check_count,
                        "score": result.overall_score,
                        "status": result.overall_status.value,
                        "duration_s": time.time() - check_start,
                    },
                )

                # Wait for next interval
                await asyncio.sleep(self.check_interval_s)

        except KeyboardInterrupt:
            logger.info("Health monitoring daemon stopped by user")
        except Exception as e:
            logger.error(f"Health monitoring daemon error: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info(
                f"Health monitoring daemon stopped (ran {check_count} checks over "
                f"{(datetime.now() - start_time).total_seconds():.1f}s)"
            )

    def stop(self) -> None:
        """Stop the monitoring daemon."""
        self.running = False

    async def _alert_score_drop(self, previous: float, current: float) -> None:
        """Handle score drop alert."""
        drop = previous - current
        logger.warning(
            f"Health score drop detected: {previous:.2f} -> {current:.2f} (drop: {drop:.2f})",
            extra={
                "previous_score": previous,
                "current_score": current,
                "drop": drop,
                "threshold": self.alert_threshold,
            },
        )

        # Would integrate with notification system here
        try:
            from afs.notifications.base import EventType, NotificationEvent, NotificationLevel

            alert_event = NotificationEvent(
                event_type=EventType.ERROR_OCCURRED,
                title="Health Score Drop Detected",
                message=f"System health dropped from {previous:.2f} to {current:.2f}",
                level=NotificationLevel.WARNING,
                metrics={"previous_score": previous, "current_score": current, "drop": drop},
            )
            # Would send notification here
            logger.info(f"Alert would be sent: {alert_event.title}")
        except Exception as e:
            logger.debug(f"Could not send notification: {e}")

    def get_trend(self, hours: int = 24) -> dict[str, Any]:
        """Get health trend for N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        trend_data = [
            (ts.isoformat(), score)
            for ts, score in self.check_history
            if ts > cutoff
        ]

        if not trend_data:
            return {"error": "No trend data available"}

        scores = [score for _, score in trend_data]
        return {
            "hours": hours,
            "data_points": len(scores),
            "current": scores[-1] if scores else None,
            "average": sum(scores) / len(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "degrading",
            "history": trend_data,
        }


def run_daemon_cli(
    interval: int = 60,
    level: str = "standard",
    duration: int | None = None,
    auto_heal: bool = False,
) -> None:
    """Run health monitoring daemon from CLI."""
    daemon = HealthMonitoringDaemon(
        check_interval_s=interval,
        alert_threshold=0.1,
        auto_heal=auto_heal,
    )

    try:
        check_level = HealthCheckLevel(level)
    except ValueError:
        print(f"Invalid check level: {level}")
        print(f"Valid options: {', '.join([lvl.value for lvl in HealthCheckLevel])}")
        return

    logger.info(
        f"Starting health monitoring daemon: interval={interval}s, level={level}, auto_heal={auto_heal}"
    )

    try:
        asyncio.run(daemon.start(check_level=check_level, duration_s=duration))
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by user")
