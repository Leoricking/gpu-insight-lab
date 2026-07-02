"""
GPU Insight Lab - NVIDIA GPU Collector
Tries pynvml first; falls back to nvidia-smi subprocess.
Never crashes when NVIDIA hardware is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NvidiaGPUInfo:
    available: bool = False
    collection_method: str = "none"
    gpu_count: int = 0
    gpu_name: Optional[str] = None
    gpu_uuid: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    gpu_clock_mhz: Optional[float] = None
    mem_clock_mhz: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    mem_utilization_pct: Optional[float] = None
    vram_total_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_free_mb: Optional[float] = None
    persistence_mode: Optional[str] = None
    performance_state: Optional[str] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    pcie_gen_max: Optional[int] = None
    pcie_width_max: Optional[int] = None
    compute_capability: Optional[str] = None
    error: Optional[str] = None


def _collect_via_pynvml() -> NvidiaGPUInfo:
    """Attempt collection using pynvml."""
    import pynvml  # type: ignore  # noqa: PLC0415

    pynvml.nvmlInit()
    info = NvidiaGPUInfo(available=True, collection_method="pynvml")
    info.gpu_count = pynvml.nvmlDeviceGetCount()
    if info.gpu_count == 0:
        pynvml.nvmlShutdown()
        info.available = False
        return info

    # Use device 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    try:
        info.gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(info.gpu_name, bytes):
            info.gpu_name = info.gpu_name.decode("utf-8")
    except pynvml.NVMLError:
        pass

    try:
        info.gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(info.gpu_uuid, bytes):
            info.gpu_uuid = info.gpu_uuid.decode("utf-8")
    except pynvml.NVMLError:
        pass

    try:
        info.driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(info.driver_version, bytes):
            info.driver_version = info.driver_version.decode("utf-8")
    except pynvml.NVMLError:
        pass

    try:
        cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion()
        major = cuda_ver // 1000
        minor = (cuda_ver % 1000) // 10
        info.cuda_driver_version = f"{major}.{minor}"
    except pynvml.NVMLError:
        pass

    try:
        info.temperature_c = float(pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        ))
    except pynvml.NVMLError:
        pass

    try:
        info.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except pynvml.NVMLError:
        pass

    try:
        info.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    except pynvml.NVMLError:
        pass

    try:
        clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        info.gpu_clock_mhz = float(clocks)
    except pynvml.NVMLError:
        pass

    try:
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        info.mem_clock_mhz = float(mem_clock)
    except pynvml.NVMLError:
        pass

    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        info.gpu_utilization_pct = float(util.gpu)
        info.mem_utilization_pct = float(util.memory)
    except pynvml.NVMLError:
        pass

    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info.vram_total_mb = mem_info.total / (1024 * 1024)
        info.vram_used_mb = mem_info.used / (1024 * 1024)
        info.vram_free_mb = mem_info.free / (1024 * 1024)
    except pynvml.NVMLError:
        pass

    try:
        pm = pynvml.nvmlDeviceGetPersistenceMode(handle)
        info.persistence_mode = "Enabled" if pm == 1 else "Disabled"
    except pynvml.NVMLError:
        pass

    try:
        ps = pynvml.nvmlDeviceGetPerformanceState(handle)
        info.performance_state = f"P{ps}"
    except pynvml.NVMLError:
        pass

    try:
        info.pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        info.pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        info.pcie_gen_max = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
        info.pcie_width_max = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
    except pynvml.NVMLError:
        pass

    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        info.compute_capability = f"{major}.{minor}"
    except pynvml.NVMLError:
        pass

    pynvml.nvmlShutdown()
    return info


def _collect_via_nvidia_smi() -> NvidiaGPUInfo:
    """Fallback: parse nvidia-smi --query-gpu output."""
    query_fields = ",".join([
        "name", "uuid", "driver_version", "temperature.gpu",
        "power.draw", "power.limit",
        "clocks.gr", "clocks.mem",
        "utilization.gpu", "utilization.memory",
        "memory.total", "memory.used", "memory.free",
        "persistence_mode", "pstate",
        "pcie.link.gen.current", "pcie.link.width.current",
        "pcie.link.gen.max", "pcie.link.width.max",
        "compute_cap",
    ])
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi not available: %s", exc)
        return NvidiaGPUInfo(available=False, error=str(exc))

    if result.returncode != 0:
        err = result.stderr.strip()
        logger.debug("nvidia-smi failed: %s", err)
        return NvidiaGPUInfo(available=False, error=err)

    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return NvidiaGPUInfo(available=False, error="no output from nvidia-smi")

    parts = [p.strip() for p in lines[0].split(",")]

    def _f(val: str) -> Optional[float]:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _i(val: str) -> Optional[int]:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    info = NvidiaGPUInfo(available=True, collection_method="nvidia-smi", gpu_count=len(lines))
    if len(parts) >= 21:
        info.gpu_name = parts[0] if parts[0] != "[Not Supported]" else None
        info.gpu_uuid = parts[1] if parts[1] != "[Not Supported]" else None
        info.driver_version = parts[2] if parts[2] != "[Not Supported]" else None
        info.temperature_c = _f(parts[3])
        info.power_draw_w = _f(parts[4])
        info.power_limit_w = _f(parts[5])
        info.gpu_clock_mhz = _f(parts[6])
        info.mem_clock_mhz = _f(parts[7])
        info.gpu_utilization_pct = _f(parts[8])
        info.mem_utilization_pct = _f(parts[9])
        info.vram_total_mb = _f(parts[10])
        info.vram_used_mb = _f(parts[11])
        info.vram_free_mb = _f(parts[12])
        info.persistence_mode = parts[13] if parts[13] not in ("[Not Supported]", "") else None
        info.performance_state = parts[14] if parts[14] not in ("[Not Supported]", "") else None
        info.pcie_gen = _i(parts[15])
        info.pcie_width = _i(parts[16])
        info.pcie_gen_max = _i(parts[17])
        info.pcie_width_max = _i(parts[18])
        cc = parts[19] if len(parts) > 19 else ""
        info.compute_capability = cc if cc and cc != "[Not Supported]" else None

    return info


def collect() -> NvidiaGPUInfo:
    """
    Collect NVIDIA GPU info. Tries pynvml, falls back to nvidia-smi.
    Returns NvidiaGPUInfo with available=False if no NVIDIA GPU present.
    Never raises.
    """
    try:
        return _collect_via_pynvml()
    except ImportError:
        logger.debug("pynvml not installed; trying nvidia-smi")
    except Exception as exc:  # noqa: BLE001
        logger.debug("pynvml collection failed: %s; trying nvidia-smi", exc)

    try:
        return _collect_via_nvidia_smi()
    except Exception as exc:  # noqa: BLE001
        logger.warning("All NVIDIA collection methods failed: %s", exc)
        return NvidiaGPUInfo(available=False, error=str(exc))
