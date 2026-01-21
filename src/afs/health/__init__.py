"""Health check and monitoring system for AFS.

Provides comprehensive diagnostics and auto-healing capabilities for:
- Model health (load time, inference latency, output quality)
- System health (CPU, memory, disk, VRAM)
- Service health (LMStudio API, MCP servers, dependencies)
- Data health (training data integrity, cache validity)
- Integration health (external APIs, notification channels)

Usage:
    from afs.health import EnhancedHealthChecker

    checker = EnhancedHealthChecker()
    result = checker.check(level='comprehensive', auto_heal=True)
    print(result.summary())
"""

from __future__ import annotations

from .enhanced_checks import (
    EnhancedHealthChecker,
    HealthCheckLevel,
    HealthCheckResult,
    HealthScore,
    HealthStatus,
)

__all__ = [
    "EnhancedHealthChecker",
    "HealthCheckLevel",
    "HealthCheckResult",
    "HealthScore",
    "HealthStatus",
]
