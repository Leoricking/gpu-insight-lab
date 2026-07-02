"""
GPU Insight Lab - Custom Command Workload
Run user-supplied command and measure GPU impact via nvidia-smi monitoring.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CustomCommandResult:
    status: str = "ok"
    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    wall_time_ms: float = 0.0
    gpu_monitor_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def run_command(
    command: str,
    timeout: int = 300,
    monitor_gpu: bool = True,
    cwd: Optional[str] = None,
) -> CustomCommandResult:
    """
    Run a user-supplied command in a subprocess, optionally monitoring GPU.
    Never uses shell=True.
    """
    result = CustomCommandResult(command=command)

    import shlex  # noqa: PLC0415
    try:
        args = shlex.split(command)
    except ValueError as exc:
        result.status = "error"
        result.error = f"Failed to parse command: {exc}"
        return result

    if not args:
        result.status = "error"
        result.error = "Empty command"
        return result

    from profilers.nvidia_smi_monitor import GPUMonitor  # noqa: PLC0415
    monitor = GPUMonitor(poll_interval_s=1.0) if monitor_gpu else None
    if monitor:
        monitor.start()

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            cwd=cwd,
        )
        t1 = time.perf_counter()
        result.returncode = proc.returncode
        result.stdout = proc.stdout[:5000]
        result.stderr = proc.stderr[:2000]
        result.wall_time_ms = (t1 - t0) * 1000

        if proc.returncode != 0:
            result.status = "error"
            result.error = f"Command exited with code {proc.returncode}"

    except subprocess.TimeoutExpired:
        t1 = time.perf_counter()
        result.wall_time_ms = (t1 - t0) * 1000
        result.status = "error"
        result.error = f"Command timed out after {timeout}s"
        result.returncode = -1
    except FileNotFoundError as exc:
        result.status = "error"
        result.error = f"Command not found: {exc}"
        result.returncode = -1
    except Exception as exc:  # noqa: BLE001
        result.status = "error"
        result.error = str(exc)
        result.returncode = -1
    finally:
        if monitor:
            monitor_data = monitor.stop()
            result.gpu_monitor_summary = monitor_data.summary()

    return result
