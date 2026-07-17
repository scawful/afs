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
        "model": "sample-model-v1",
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

from .context_layout import LAYOUT_VERSION, detect_layout_version, resolve_runtime_root
from .path_safety import assert_no_linklike_components


def _default_context_root() -> Path:
    return Path.home() / ".context"


def _default_log_dir(*, create: bool = False) -> tuple[Path, Path | None]:
    context_root = _default_context_root().expanduser().resolve()
    logs_root = resolve_runtime_root(
        context_root,
        "logs",
        legacy_relative="logs/afs",
        create=create,
    )
    if detect_layout_version(context_root) != LAYOUT_VERSION:
        return logs_root, None
    log_dir = assert_no_linklike_components(
        logs_root / "afs",
        boundary=logs_root,
    )
    if create:
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_dir = assert_no_linklike_components(
            log_dir,
            boundary=logs_root,
            allow_missing=False,
        )
    return log_dir, context_root

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
        log_dir: Explicit log directory. The default is layout-aware:
            ``~/.context/.afs/logs/afs`` for v2, ``~/.context/logs/afs`` for v1.
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
    managed_context_root: Path | None = None
    if log_dir is None:
        log_dir, managed_context_root = _default_log_dir(create=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    def managed_log_path(filename: str) -> Path:
        path = log_dir / filename
        if managed_context_root is None:
            return path
        return assert_no_linklike_components(
            path,
            boundary=log_dir,
        )

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
        log_file = managed_log_path(f"{name}.log")
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
        json_log_file = managed_log_path(f"{name}.json.log")
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
        with LogContext(run_id="training_123", model="sample-model-v1"):
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
