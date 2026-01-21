"""Production-grade logging configuration for AFS.

Provides structured logging with:
- JSON output for log aggregation
- Contextual information (model, dataset, run_id)
- Performance metrics
- Error tracking with stack traces
- Log rotation
- Multiple outputs (file, stdout, remote)

Usage:
    from afs.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Training started", extra={
        "model": "majora-v1",
        "dataset_size": 187,
        "epochs": 3
    })
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    @staticmethod
    def _utc_isoformat() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self._utc_isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data)


class PerformanceLogger:
    """Track performance metrics alongside logs."""

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}

    def record_metric(self, name: str, value: float, unit: str = ""):
        """Record a performance metric."""
        self.metrics[name] = {
            "value": value,
            "unit": unit,
            "timestamp": self._utc_now().isoformat(),
        }
        self.logger.info(f"Metric: {name}", extra={
            "metric_name": name,
            "metric_value": value,
            "metric_unit": unit
        })

    def get_metrics(self) -> dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()


def setup_logging(
    name: str = "afs",
    level: int = logging.INFO,
    log_dir: Path | None = None,
    enable_json: bool = True,
    enable_console: bool = True,
    enable_rotation: bool = True
) -> logging.Logger:
    """Set up production logging configuration.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files (default: ~/.context/logs/)
        enable_json: Use JSON formatter
        enable_console: Log to console
        enable_rotation: Enable log rotation

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Setup log directory
    if log_dir is None:
        log_dir = Path("~/.context/logs/afs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Formatter
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Use simpler format for console
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if enable_rotation:
        log_file = log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # JSON file handler for structured logs
    if enable_json:
        json_log_file = log_dir / f"{name}.json.log"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration."""
    return logging.getLogger(f"afs.{name}")


# Global logger instance
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get the default AFS logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging("afs")
    return _default_logger


class LogContext:
    """Context manager for adding contextual information to logs.

    Usage:
        with LogContext(run_id="training_123", model="majora-v1"):
            logger.info("Training started")
            # All logs in this block will include run_id and model
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = None

    def __enter__(self):
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra = self.context
            return record

        logging.setLogRecordFactory(record_factory)
        self.old_factory = old_factory

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Export main APIs
__all__ = [
    "setup_logging",
    "get_logger",
    "get_default_logger",
    "LogContext",
    "PerformanceLogger",
    "JSONFormatter"
]
