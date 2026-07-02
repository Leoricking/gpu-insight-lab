"""
GPU Insight Lab - AMD GPU Collector
Detects ROCm and AMD GPU info if available.
Status: NOT_VALIDATED - no AMD GPU was available during development.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


VALIDATION_STATUS = "NOT_VALIDATED"


@dataclass
class AMDGPUInfo:
    available: bool = False
    validation_status: str = VALIDATION_STATUS
    rocm_available: bool = False
    rocm_version: Optional[str] = None
    gpu_count: int = 0
    gpu_name: Optional[str] = None
    driver_version: Optional[str] = None
    vram_total_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    hipcc_available: bool = False
    rocminfo_available: bool = False
    error: Optional[str] = None
    note: str = (
        "AMD/ROCm support is NOT_VALIDATED in GPU Insight Lab v0.1.0. "
        "No AMD GPU was present during development. "
        "HIP vector_add stub is provided in native/hip/ for reference."
    )


def _try_rocm_smi() -> tuple[bool, Optional[str], Optional[str]]:
    """Try rocm-smi. Return (available, gpu_name, version)."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            gpu_name = None
            for line in lines:
                if "GPU" in line and ":" in line:
                    gpu_name = line.split(":", 1)[-1].strip()
                    break
            return True, gpu_name, None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("rocm-smi not available: %s", exc)
    return False, None, None


def _try_rocminfo() -> tuple[bool, Optional[str]]:
    """Try rocminfo. Return (available, version)."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
        )
        if result.returncode == 0:
            return True, None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("rocminfo not available: %s", exc)
    return False, None


def _try_hipcc_version() -> bool:
    """Check if hipcc is available."""
    try:
        result = subprocess.run(
            ["hipcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_rocm_version() -> Optional[str]:
    """Try to find ROCm version from filesystem."""
    from pathlib import Path  # noqa: PLC0415
    candidates = [
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/version.txt"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except OSError:
                pass
    return None


def collect() -> AMDGPUInfo:
    """Collect AMD GPU info. Always returns NOT_VALIDATED in v0.1.0. Never raises."""
    info = AMDGPUInfo()

    try:
        info.rocm_version = _detect_rocm_version()
        if info.rocm_version:
            info.rocm_available = True

        smi_ok, gpu_name, _ = _try_rocm_smi()
        if smi_ok:
            info.available = True
            info.gpu_name = gpu_name

        rocminfo_ok, _ = _try_rocminfo()
        info.rocminfo_available = rocminfo_ok
        if rocminfo_ok:
            info.rocm_available = True

        info.hipcc_available = _try_hipcc_version()

    except Exception as exc:  # noqa: BLE001
        logger.debug("AMD collector error: %s", exc)
        info.error = str(exc)

    # Always stamp NOT_VALIDATED
    info.validation_status = VALIDATION_STATUS
    return info
