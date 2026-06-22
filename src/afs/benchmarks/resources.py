"""Resource usage monitoring for model inference.

Tracks:
- Memory footprint (RAM)
- VRAM/GPU memory usage
- CPU utilization
- Power consumption (if available on platform)
- Disk I/O
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    psutil = None

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - platform dependent
    resource = None


def _fallback_memory_snapshot() -> tuple[float, float, float]:
    rss_mb = 0.0
    if resource:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if os.name == "posix" and os.uname().sysname.lower() == "darwin":
            rss_mb = rss / (1024 * 1024)
        else:
            rss_mb = rss / 1024
    vms_mb = rss_mb
    percent = 0.0
    if rss_mb > 0 and hasattr(os, "sysconf"):
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total_bytes = None
        if total_bytes:
            percent = (rss_mb * 1024 * 1024) / total_bytes * 100
    return rss_mb, vms_mb, percent


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percent of total memory

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rss_mb": self.rss_mb,
            "vms_mb": self.vms_mb,
            "percent": self.percent,
        }


@dataclass
class MemoryMetrics:
    """Aggregated memory metrics."""

    peak_rss_mb: float
    average_rss_mb: float
    peak_vms_mb: float
    average_vms_mb: float
    peak_percent: float
    average_percent: float
    samples: int
    timeline: list[MemorySnapshot] = field(default_factory=list)

    def to_dict(self, include_timeline: bool = False) -> dict[str, Any]:
        data = {
            "peak_rss_mb": self.peak_rss_mb,
            "average_rss_mb": self.average_rss_mb,
            "peak_vms_mb": self.peak_vms_mb,
            "average_vms_mb": self.average_vms_mb,
            "peak_percent": self.peak_percent,
            "average_percent": self.average_percent,
            "samples": self.samples,
        }
        if include_timeline:
            data["timeline"] = [s.to_dict() for s in self.timeline]
        return data


@dataclass
class VRAMMetrics:
    """VRAM/GPU memory metrics."""

    peak_allocated_mb: float
    average_allocated_mb: float
    peak_reserved_mb: float
    average_reserved_mb: float
    device: str
    unified_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_allocated_mb": self.peak_allocated_mb,
            "average_allocated_mb": self.average_allocated_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "average_reserved_mb": self.average_reserved_mb,
            "device": self.device,
            "unified_memory": self.unified_memory,
        }


@dataclass
class CPUMetrics:
    """CPU utilization metrics."""

    peak_percent: float
    average_percent: float
    peak_per_core: list[float]
    average_per_core: list[float]
    context_switches: int
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_percent": self.peak_percent,
            "average_percent": self.average_percent,
            "peak_per_core": self.peak_per_core,
            "average_per_core": self.average_per_core,
            "context_switches": self.context_switches,
            "samples": self.samples,
        }


@dataclass
class PowerMetrics:
    """Power consumption metrics (if available)."""

    average_watts: float | None
    peak_watts: float | None
    total_joules: float | None
    available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_watts": self.average_watts,
            "peak_watts": self.peak_watts,
            "total_joules": self.total_joules,
            "available": self.available,
        }


@dataclass
class ResourceBenchmarkResult:
    """Complete resource monitoring results."""

    model_name: str
    model_path: str
    memory: MemoryMetrics
    vram: VRAMMetrics | None
    cpu: CPUMetrics
    power: PowerMetrics
    monitoring_duration_seconds: float
    timestamp: str

    def to_dict(self, include_timeline: bool = False) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "memory": self.memory.to_dict(include_timeline),
            "vram": self.vram.to_dict() if self.vram else None,
            "cpu": self.cpu.to_dict(),
            "power": self.power.to_dict(),
            "monitoring_duration_seconds": self.monitoring_duration_seconds,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Resource Benchmark: {self.model_name}",
            f"Path: {self.model_path}",
            "",
            "Memory:",
            f"  Peak RSS: {self.memory.peak_rss_mb:.1f} MB",
            f"  Average RSS: {self.memory.average_rss_mb:.1f} MB",
            f"  Peak Usage: {self.memory.peak_percent:.1f}%",
            "",
            "CPU:",
            f"  Peak: {self.cpu.peak_percent:.1f}%",
            f"  Average: {self.cpu.average_percent:.1f}%",
            f"  Context Switches: {self.cpu.context_switches:,}",
            "",
        ]

        if self.vram:
            lines.extend([
                "VRAM:",
                f"  Device: {self.vram.device}",
                f"  Peak Allocated: {self.vram.peak_allocated_mb:.1f} MB",
                f"  Average Allocated: {self.vram.average_allocated_mb:.1f} MB",
                "",
            ])

        if self.power.available:
            lines.extend([
                "Power:",
                f"  Average: {self.power.average_watts:.1f}W",
                f"  Peak: {self.power.peak_watts:.1f}W",
                f"  Total Energy: {self.power.total_joules:.1f}J",
                "",
            ])

        lines.append(f"Monitoring Duration: {self.monitoring_duration_seconds:.2f}s")

        return "\n".join(lines)


class MemoryMonitor:
    """Monitor memory usage of a process."""

    def __init__(self, pid: int | None = None, sample_interval: float = 0.1):
        """Initialize memory monitor.

        Args:
            pid: Process ID to monitor (default: current process)
            sample_interval: Sampling interval in seconds
        """
        if psutil is None:
            self.pid = pid or os.getpid()
            self.process = None
        else:
            self.pid = pid or psutil.Process().pid
            self.process = psutil.Process(self.pid)
        self.sample_interval = sample_interval
        self._snapshots: list[MemorySnapshot] = []
        self._monitoring = False
        self._thread = None

    def start(self):
        """Start monitoring in background thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._snapshots = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MemoryMetrics:
        """Stop monitoring and return metrics."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self._snapshots:
            return MemoryMetrics(0, 0, 0, 0, 0, 0, 0)

        # Aggregate metrics
        rss_values = [s.rss_mb for s in self._snapshots]
        vms_values = [s.vms_mb for s in self._snapshots]
        percent_values = [s.percent for s in self._snapshots]

        return MemoryMetrics(
            peak_rss_mb=max(rss_values),
            average_rss_mb=sum(rss_values) / len(rss_values),
            peak_vms_mb=max(vms_values),
            average_vms_mb=sum(vms_values) / len(vms_values),
            peak_percent=max(percent_values),
            average_percent=sum(percent_values) / len(percent_values),
            samples=len(self._snapshots),
            timeline=self._snapshots,
        )

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            if psutil is None or self.process is None:
                rss_mb, vms_mb, percent = _fallback_memory_snapshot()
                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=rss_mb,
                    vms_mb=vms_mb,
                    percent=percent,
                )
                self._snapshots.append(snapshot)
            else:
                try:
                    mem_info = self.process.memory_info()
                    snapshot = MemorySnapshot(
                        timestamp=time.time(),
                        rss_mb=mem_info.rss / (1024 * 1024),
                        vms_mb=mem_info.vms / (1024 * 1024),
                        percent=self.process.memory_percent(),
                    )
                    self._snapshots.append(snapshot)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

            time.sleep(self.sample_interval)


class VRAMMonitor:
    """Monitor VRAM/GPU memory usage."""

    def __init__(self):
        """Initialize VRAM monitor."""
        self.device = "unknown"
        self.unified_memory = False
        self._samples: list[dict[str, float]] = []
        self._monitoring = False
        self._thread = None

        if find_spec("mlx.core") is not None:
            self.device = "Apple Silicon GPU"
            self.unified_memory = True

    def start(self):
        """Start monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> VRAMMetrics | None:
        """Stop monitoring and return metrics."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self._samples:
            return None

        allocated = [s["allocated_mb"] for s in self._samples]
        reserved = [s["reserved_mb"] for s in self._samples]

        return VRAMMetrics(
            peak_allocated_mb=max(allocated),
            average_allocated_mb=sum(allocated) / len(allocated),
            peak_reserved_mb=max(reserved),
            average_reserved_mb=sum(reserved) / len(reserved),
            device=self.device,
            unified_memory=self.unified_memory,
        )

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                if find_spec("mlx.core") is not None:
                    # MLX doesn't expose direct memory stats, so estimate
                    # based on system memory for unified memory systems
                    if psutil is None:
                        break
                    mem = psutil.virtual_memory()
                    sample = {
                        "allocated_mb": mem.used / (1024 * 1024),
                        "reserved_mb": mem.total / (1024 * 1024),
                    }
                    self._samples.append(sample)

            except Exception:
                break

            time.sleep(0.1)


class CPUMonitor:
    """Monitor CPU utilization."""

    def __init__(self, pid: int | None = None, sample_interval: float = 0.1):
        """Initialize CPU monitor.

        Args:
            pid: Process ID to monitor (default: current process)
            sample_interval: Sampling interval in seconds
        """
        self.pid = pid or os.getpid()
        self.sample_interval = sample_interval
        self.process = psutil.Process(self.pid) if psutil else None
        self._samples: list[dict[str, Any]] = []
        self._monitoring = False
        self._thread = None
        self._initial_ctx_switches = (
            self.process.num_ctx_switches() if self.process else None
        )
        self._last_cpu_time: float | None = None
        self._last_wall_time: float | None = None

    def start(self):
        """Start monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> CPUMetrics:
        """Stop monitoring and return metrics."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self._samples:
            return CPUMetrics(0, 0, [], [], 0, 0)

        # Aggregate
        overall = [s["percent"] for s in self._samples]
        per_core_samples = [s["per_core"] for s in self._samples if s["per_core"]]

        # Average per core
        num_cores = len(per_core_samples[0]) if per_core_samples else 0
        avg_per_core = []
        peak_per_core = []

        if per_core_samples:
            for i in range(num_cores):
                core_values = [s[i] for s in per_core_samples]
                avg_per_core.append(sum(core_values) / len(core_values))
                peak_per_core.append(max(core_values))

        # Context switches
        ctx_switches = 0
        if self.process and self._initial_ctx_switches:
            final_ctx = self.process.num_ctx_switches()
            ctx_switches = (
                final_ctx.voluntary - self._initial_ctx_switches.voluntary
                + final_ctx.involuntary - self._initial_ctx_switches.involuntary
            )

        return CPUMetrics(
            peak_percent=max(overall),
            average_percent=sum(overall) / len(overall),
            peak_per_core=peak_per_core,
            average_per_core=avg_per_core,
            context_switches=ctx_switches,
            samples=len(self._samples),
        )

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            if psutil is None or self.process is None:
                wall_time = time.time()
                cpu_time = time.process_time()
                if self._last_wall_time is None:
                    self._last_wall_time = wall_time
                    self._last_cpu_time = cpu_time
                else:
                    delta_wall = wall_time - self._last_wall_time
                    delta_cpu = cpu_time - (self._last_cpu_time or 0.0)
                    percent = (delta_cpu / delta_wall) * 100 if delta_wall > 0 else 0.0
                    self._samples.append(
                        {
                            "timestamp": wall_time,
                            "percent": max(0.0, percent),
                            "per_core": [],
                        }
                    )
                    self._last_wall_time = wall_time
                    self._last_cpu_time = cpu_time
            else:
                try:
                    sample = {
                        "timestamp": time.time(),
                        "percent": self.process.cpu_percent(),
                        "per_core": psutil.cpu_percent(percpu=True),
                    }
                    self._samples.append(sample)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

            time.sleep(self.sample_interval)


class PowerMonitor:
    """Monitor power consumption (platform-specific)."""

    def __init__(self):
        """Initialize power monitor."""
        self.available = False
        self._samples: list[float] = []
        self._start_time = None
        self._monitoring = False

        # Check if powermetrics is available (macOS)
        import shutil

        if shutil.which("powermetrics"):
            self.available = True

    def start(self):
        """Start monitoring."""
        if not self.available or self._monitoring:
            return

        self._monitoring = True
        self._samples = []
        self._start_time = time.time()

    def stop(self) -> PowerMetrics:
        """Stop monitoring and return metrics."""
        self._monitoring = False

        if not self.available or not self._samples:
            return PowerMetrics(None, None, None, False)

        duration = time.time() - self._start_time if self._start_time else 1.0

        return PowerMetrics(
            average_watts=sum(self._samples) / len(self._samples),
            peak_watts=max(self._samples),
            total_joules=sum(self._samples) * duration,
            available=True,
        )


class ResourceBenchmark:
    """Comprehensive resource usage benchmarking."""

    def __init__(
        self,
        model_path: Path | str,
        model_name: str | None = None,
        sample_interval: float = 0.1,
    ):
        """Initialize resource benchmark.

        Args:
            model_path: Path to model checkpoint
            model_name: Display name for model
            sample_interval: Sampling interval in seconds
        """
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.name
        self.sample_interval = sample_interval

    def run(
        self,
        workload_fn: callable,
        monitor_vram: bool = True,
        monitor_power: bool = True,
    ) -> ResourceBenchmarkResult:
        """Run resource benchmark during workload execution.

        Args:
            workload_fn: Function to execute while monitoring (e.g., model inference)
            monitor_vram: Whether to monitor VRAM usage
            monitor_power: Whether to monitor power consumption

        Returns:
            ResourceBenchmarkResult with all metrics
        """
        from datetime import datetime

        # Initialize monitors
        memory_monitor = MemoryMonitor(sample_interval=self.sample_interval)
        cpu_monitor = CPUMonitor(sample_interval=self.sample_interval)
        vram_monitor = VRAMMonitor() if monitor_vram else None
        power_monitor = PowerMonitor() if monitor_power else None

        # Start monitoring
        start_time = time.perf_counter()
        memory_monitor.start()
        cpu_monitor.start()
        if vram_monitor:
            vram_monitor.start()
        if power_monitor:
            power_monitor.start()

        # Run workload
        try:
            workload_fn()
        finally:
            # Stop monitoring
            memory_metrics = memory_monitor.stop()
            cpu_metrics = cpu_monitor.stop()
            vram_metrics = vram_monitor.stop() if vram_monitor else None
            power_metrics = power_monitor.stop() if power_monitor else PowerMetrics(None, None, None, False)

        end_time = time.perf_counter()

        return ResourceBenchmarkResult(
            model_name=self.model_name,
            model_path=str(self.model_path),
            memory=memory_metrics,
            vram=vram_metrics,
            cpu=cpu_metrics,
            power=power_metrics,
            monitoring_duration_seconds=end_time - start_time,
            timestamp=datetime.now().isoformat(),
        )
