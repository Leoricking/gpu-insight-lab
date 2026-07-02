"""
GPU Insight Lab - Real-time nvidia-smi Monitor
Polls GPU metrics during benchmarks. Returns time series data.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUSample:
    timestamp: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    mem_utilization_pct: Optional[float] = None
    gpu_clock_mhz: Optional[float] = None
    mem_clock_mhz: Optional[float] = None
    vram_used_mb: Optional[float] = None


@dataclass
class MonitorResult:
    available: bool = False
    samples: List[GPUSample] = field(default_factory=list)
    poll_interval_s: float = 1.0
    error: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics over all samples."""
        if not self.samples:
            return {}

        def _avg(attr: str) -> Optional[float]:
            vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def _max_v(attr: str) -> Optional[float]:
            vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
            return max(vals) if vals else None

        return {
            "sample_count": len(self.samples),
            "avg_temperature_c": _avg("temperature_c"),
            "max_temperature_c": _max_v("temperature_c"),
            "avg_power_draw_w": _avg("power_draw_w"),
            "max_power_draw_w": _max_v("power_draw_w"),
            "avg_gpu_utilization_pct": _avg("gpu_utilization_pct"),
            "max_gpu_utilization_pct": _max_v("gpu_utilization_pct"),
            "avg_mem_utilization_pct": _avg("mem_utilization_pct"),
        }


def _parse_sample(line: str) -> Optional[GPUSample]:
    """Parse one line of nvidia-smi --query-gpu output."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None

    def _f(s: str) -> Optional[float]:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    return GPUSample(
        timestamp=time.time(),
        temperature_c=_f(parts[0]),
        power_draw_w=_f(parts[1]),
        gpu_utilization_pct=_f(parts[2]),
        mem_utilization_pct=_f(parts[3]),
        gpu_clock_mhz=_f(parts[4]),
        mem_clock_mhz=_f(parts[5]),
        vram_used_mb=_f(parts[6]),
    )


class GPUMonitor:
    """Background monitor that polls nvidia-smi periodically."""

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self._interval = poll_interval_s
        self._samples: List[GPUSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    def _query_once(self) -> Optional[GPUSample]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,utilization.gpu,"
                    "utilization.memory,clocks.gr,clocks.mem,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
            )
            if result.returncode == 0 and result.stdout.strip():
                return _parse_sample(result.stdout.strip().splitlines()[0])
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            self._error = str(exc)
        return None

    def _run(self) -> None:
        while self._running:
            sample = self._query_once()
            if sample:
                self._samples.append(sample)
            time.sleep(self._interval)

    def start(self) -> bool:
        """Start background monitoring. Returns False if nvidia-smi unavailable."""
        # Test availability
        sample = self._query_once()
        if sample is None:
            return False
        self._samples = [sample]
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-monitor")
        self._thread.start()
        return True

    def stop(self) -> MonitorResult:
        """Stop monitoring and return collected data."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        return MonitorResult(
            available=True,
            samples=list(self._samples),
            poll_interval_s=self._interval,
            error=self._error,
        )


def monitor_during(fn: Any, poll_interval_s: float = 1.0) -> tuple[Any, MonitorResult]:
    """
    Run fn() and monitor GPU during execution.
    Returns (fn_result, MonitorResult).
    """
    monitor = GPUMonitor(poll_interval_s=poll_interval_s)
    available = monitor.start()
    try:
        result = fn()
    finally:
        monitor_result = monitor.stop()
        if not available:
            monitor_result.available = False
            monitor_result.error = monitor._error or "nvidia-smi unavailable"
    return result, monitor_result
