"""Base notification classes and event types."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from afs.logging_config import get_logger

logger = get_logger(__name__)


class NotificationLevel(str, Enum):
    """Notification severity levels."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(str, Enum):
    """Types of training events that trigger notifications."""

    # Training lifecycle
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    TRAINING_PAUSED = "training_paused"
    TRAINING_RESUMED = "training_resumed"

    # Checkpointing
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADING = "checkpoint_loading"

    # Evaluation
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"

    # Milestones
    EPOCH_COMPLETED = "epoch_completed"
    BATCH_PROCESSED = "batch_processed"
    LOSS_IMPROVED = "loss_improved"
    LOSS_DEGRADED = "loss_degraded"

    # Cost and resource alerts
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    GPU_MEMORY_WARNING = "gpu_memory_warning"
    GPU_UTILIZATION_LOW = "gpu_utilization_low"
    DISK_SPACE_WARNING = "disk_space_warning"

    # Errors
    ERROR_OCCURRED = "error_occurred"
    OUT_OF_MEMORY = "out_of_memory"
    NAN_DETECTED = "nan_detected"


@dataclass
class NotificationEvent:
    """Structured notification event."""

    event_type: EventType
    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional context
    model_name: Optional[str] = None
    run_id: Optional[str] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None
    cost: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["level"] = self.level.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.level.upper()}] {self.title}: {self.message}"


class NotificationHandler(ABC):
    """Base class for notification handlers."""

    @abstractmethod
    def send(self, event: NotificationEvent) -> bool:
        """Send notification for event.

        Args:
            event: Notification event

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if handler is properly configured."""
        pass


class NotificationManager:
    """Manages notifications across multiple channels."""

    def __init__(self):
        """Initialize notification manager."""
        self.handlers: dict[str, NotificationHandler] = {}
        self.enabled_channels: set[str] = set()
        self.logger = logger

    def register_handler(self, name: str, handler: NotificationHandler) -> None:
        """Register a notification handler.

        Args:
            name: Handler name (e.g., 'desktop', 'email')
            handler: Notification handler instance
        """
        self.handlers[name] = handler
        if handler.is_configured():
            self.enabled_channels.add(name)
            self.logger.info(f"Registered notification handler: {name}")
        else:
            self.logger.warning(f"Handler {name} not properly configured, skipping")

    def notify(
        self,
        title: str,
        message: str,
        event_type: EventType | str = EventType.ERROR_OCCURRED,
        level: NotificationLevel = NotificationLevel.INFO,
        **kwargs
    ) -> bool:
        """Send notification to all enabled channels.

        Args:
            title: Notification title
            message: Notification message
            event_type: Type of event
            level: Notification level
            **kwargs: Additional event context (model_name, run_id, metrics, etc.)

        Returns:
            True if sent to at least one channel
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            try:
                event_type = EventType[event_type.upper()]
            except KeyError:
                event_type = EventType.ERROR_OCCURRED

        event = NotificationEvent(
            event_type=event_type,
            title=title,
            message=message,
            level=level,
            **kwargs
        )

        return self.send_event(event)

    def send_event(self, event: NotificationEvent) -> bool:
        """Send a pre-constructed event.

        Args:
            event: NotificationEvent instance

        Returns:
            True if sent to at least one channel
        """
        sent_count = 0
        handlers = {
            channel: self.handlers.get(channel)
            for channel in self.enabled_channels
            if self.handlers.get(channel)
        }
        if not handlers:
            handlers = self._fallback_handlers()
        if not handlers:
            self.logger.warning("No notification channels enabled")
            return False

        for channel, handler in handlers.items():
            if handler:
                try:
                    if handler.send(event):
                        sent_count += 1
                except Exception as e:
                    self.logger.error(
                        f"Failed to send notification via {channel}: {e}",
                        exc_info=True
                    )

        return sent_count > 0

    def _fallback_handlers(self) -> dict[str, NotificationHandler]:
        """Return handlers assigned directly to common channel attributes."""
        fallback: dict[str, NotificationHandler] = {}
        for name in ("desktop", "email", "slack", "discord"):
            handler = getattr(self, name, None)
            if handler and hasattr(handler, "send"):
                fallback[name] = handler
        return fallback

    def notify_training_started(
        self,
        model_name: str,
        run_id: str,
        epochs: int,
        batch_size: int
    ) -> bool:
        """Notify that training has started."""
        return self.notify(
            title=f"Training started: {model_name}",
            message=f"Run {run_id}\nEpochs: {epochs}, Batch size: {batch_size}",
            event_type=EventType.TRAINING_STARTED,
            level=NotificationLevel.INFO,
            model_name=model_name,
            run_id=run_id,
            metrics={"epochs": epochs, "batch_size": batch_size}
        )

    def notify_training_completed(
        self,
        model_name: str,
        run_id: str,
        duration: float,
        final_loss: float,
        eval_metrics: Optional[dict] = None
    ) -> bool:
        """Notify that training has completed."""
        message_lines = [
            f"Model: {model_name}",
            f"Run: {run_id}",
            f"Duration: {duration:.2f}s",
            f"Final loss: {final_loss:.4f}"
        ]
        if eval_metrics:
            for key, value in eval_metrics.items():
                message_lines.append(f"{key}: {value}")

        return self.notify(
            title=f"Training completed: {model_name}",
            message="\n".join(message_lines),
            event_type=EventType.TRAINING_COMPLETED,
            level=NotificationLevel.SUCCESS,
            model_name=model_name,
            run_id=run_id,
            metrics={
                "duration": duration,
                "final_loss": final_loss,
                **(eval_metrics or {})
            }
        )

    def notify_training_failed(
        self,
        model_name: str,
        run_id: str,
        error: str
    ) -> bool:
        """Notify that training has failed."""
        return self.notify(
            title=f"Training failed: {model_name}",
            message=f"Run {run_id}\nError: {error}",
            event_type=EventType.TRAINING_FAILED,
            level=NotificationLevel.ERROR,
            model_name=model_name,
            run_id=run_id,
            error_details=error
        )

    def notify_cost_alert(
        self,
        current_cost: float,
        threshold: float,
        model_name: Optional[str] = None
    ) -> bool:
        """Notify cost threshold exceeded."""
        return self.notify(
            title="Cost threshold alert",
            message=f"Current cost: ${current_cost:.2f}\nThreshold: ${threshold:.2f}",
            event_type=EventType.COST_THRESHOLD_EXCEEDED,
            level=NotificationLevel.WARNING,
            model_name=model_name,
            cost=current_cost,
            metrics={"threshold": threshold}
        )

    def notify_evaluation_completed(
        self,
        model_name: str,
        run_id: str,
        metrics: dict[str, float]
    ) -> bool:
        """Notify evaluation completed."""
        message_lines = [
            f"Model: {model_name}",
            f"Run: {run_id}",
            "Results:"
        ]
        for key, value in metrics.items():
            message_lines.append(f"  {key}: {value:.4f}")

        return self.notify(
            title=f"Evaluation completed: {model_name}",
            message="\n".join(message_lines),
            event_type=EventType.EVALUATION_COMPLETED,
            level=NotificationLevel.SUCCESS,
            model_name=model_name,
            run_id=run_id,
            metrics=metrics
        )

    def notify_error(
        self,
        error_message: str,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> bool:
        """Notify generic error."""
        return self.notify(
            title="Training error",
            message=error_message,
            event_type=EventType.ERROR_OCCURRED,
            level=NotificationLevel.ERROR,
            model_name=model_name,
            run_id=run_id,
            error_details=error_message
        )

    def get_status(self) -> dict[str, Any]:
        """Get notification system status."""
        return {
            "registered_handlers": list(self.handlers.keys()),
            "enabled_channels": list(self.enabled_channels),
            "channel_status": {
                name: {
                    "configured": handler.is_configured(),
                    "handler_type": handler.__class__.__name__
                }
                for name, handler in self.handlers.items()
            }
        }
