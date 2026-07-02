"""
GPU Insight Lab - System Information Collector
Collects OS, CPU, RAM, Python, uptime. Uses psutil + platform.
Gracefully degrades if psutil unavailable.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    os_name: str = ""
    os_version: str = ""
    os_release: str = ""
    hostname: str = ""
    architecture: str = ""
    cpu_model: str = ""
    cpu_physical_cores: int = 0
    cpu_logical_cores: int = 0
    cpu_max_freq_mhz: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    python_version: str = ""
    python_implementation: str = ""
    uptime_seconds: float = 0.0
    error: Optional[str] = None


def collect() -> SystemInfo:
    """Collect system information. Never raises; returns partial data on error."""
    info = SystemInfo()

    # Platform basics (always available)
    try:
        info.os_name = platform.system()
        info.os_version = platform.version()
        info.os_release = platform.release()
        info.hostname = platform.node()
        info.architecture = platform.machine()
        info.python_version = platform.python_version()
        info.python_implementation = platform.python_implementation()
    except Exception as exc:  # noqa: BLE001
        logger.warning("platform collection failed: %s", exc)
        info.error = str(exc)

    # psutil-based metrics
    try:
        import psutil  # type: ignore

        # CPU
        try:
            info.cpu_physical_cores = psutil.cpu_count(logical=False) or 0
            info.cpu_logical_cores = psutil.cpu_count(logical=True) or 0
            freq = psutil.cpu_freq()
            if freq:
                info.cpu_max_freq_mhz = freq.max
        except Exception as exc:  # noqa: BLE001
            logger.debug("psutil cpu info failed: %s", exc)

        # CPU model
        try:
            if platform.system() == "Windows":
                import winreg  # noqa: PLC0415
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                info.cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                winreg.CloseKey(key)
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("model name"):
                            info.cpu_model = line.split(":")[1].strip()
                            break
            elif platform.system() == "Darwin":
                import subprocess  # noqa: PLC0415
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                info.cpu_model = result.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("cpu model detection failed: %s", exc)
            info.cpu_model = platform.processor() or "Unknown"

        # RAM
        try:
            mem = psutil.virtual_memory()
            info.ram_total_gb = round(mem.total / (1024**3), 2)
            info.ram_available_gb = round(mem.available / (1024**3), 2)
            info.ram_used_gb = round(mem.used / (1024**3), 2)
            info.ram_percent = mem.percent
        except Exception as exc:  # noqa: BLE001
            logger.debug("psutil memory info failed: %s", exc)

        # Uptime
        try:
            boot_time = psutil.boot_time()
            info.uptime_seconds = round(time.time() - boot_time, 1)
        except Exception as exc:  # noqa: BLE001
            logger.debug("psutil uptime failed: %s", exc)

    except ImportError:
        logger.warning("psutil not available; RAM/CPU info limited")
        info.cpu_model = platform.processor() or "Unknown"

    return info
