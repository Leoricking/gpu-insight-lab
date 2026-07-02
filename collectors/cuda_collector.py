"""
GPU Insight Lab - CUDA Toolchain Collector
Detects nvcc version, CUDA_HOME, CUDA_PATH, runtime availability,
and native benchmark executable path.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CudaInfo:
    nvcc_available: bool = False
    nvcc_version: Optional[str] = None
    nvcc_path: Optional[str] = None
    cuda_home: Optional[str] = None
    cuda_path: Optional[str] = None
    cuda_runtime_available: bool = False
    cuda_runtime_version: Optional[str] = None
    native_benchmark_available: bool = False
    native_benchmark_path: Optional[str] = None
    error: Optional[str] = None


def _find_nvcc_version() -> tuple[bool, Optional[str], Optional[str]]:
    """Return (found, version_string, path_string)."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
        )
        if result.returncode == 0:
            # Parse e.g. "release 12.3, V12.3.107"
            out = result.stdout
            import re  # noqa: PLC0415
            m = re.search(r"release\s+([\d.]+)", out)
            version = m.group(1) if m else out.strip().splitlines()[-1]

            # Find path
            try:
                which_result = subprocess.run(
                    ["where", "nvcc"] if sys.platform == "win32" else ["which", "nvcc"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                nvcc_path = which_result.stdout.strip().splitlines()[0] if which_result.returncode == 0 else None
            except Exception:  # noqa: BLE001
                nvcc_path = None

            return True, version, nvcc_path
        return False, None, None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvcc not found: %s", exc)
        return False, None, None


def _detect_cuda_runtime_version() -> tuple[bool, Optional[str]]:
    """Try to detect CUDA runtime version via ctypes."""
    try:
        import ctypes  # noqa: PLC0415
        if sys.platform == "win32":
            lib_names = ["nvcuda.dll", "cudart64_12.dll", "cudart64_11.dll", "cudart64_110.dll"]
        elif sys.platform == "darwin":
            lib_names = ["libcuda.dylib", "libcudart.dylib"]
        else:
            lib_names = ["libcuda.so.1", "libcudart.so"]

        for lib in lib_names:
            try:
                cuda_lib = ctypes.CDLL(lib)
                # Try to get version
                version = ctypes.c_int(0)
                if hasattr(cuda_lib, "cudaRuntimeGetVersion"):
                    ret = cuda_lib.cudaRuntimeGetVersion(ctypes.byref(version))
                    if ret == 0:
                        v = version.value
                        major = v // 1000
                        minor = (v % 1000) // 10
                        return True, f"{major}.{minor}"
                return True, None  # Library found but version not accessible
            except OSError:
                continue
        return False, None
    except Exception as exc:  # noqa: BLE001
        logger.debug("CUDA runtime detection failed: %s", exc)
        return False, None


def _find_native_benchmark() -> tuple[bool, Optional[str]]:
    """Search for gpu_insight_benchmark executable."""
    from app.branding import NATIVE_EXECUTABLE  # noqa: PLC0415
    exe_name = NATIVE_EXECUTABLE + (".exe" if sys.platform == "win32" else "")

    search_paths = [
        Path(__file__).parent.parent / "bin" / exe_name,
        Path(__file__).parent.parent / "build" / exe_name,
        Path(__file__).parent.parent / "build" / "Release" / exe_name,
        Path(__file__).parent.parent / "build" / "Debug" / exe_name,
    ]

    for p in search_paths:
        if p.exists():
            return True, str(p)

    # Check PATH
    try:
        which_cmd = ["where", exe_name] if sys.platform == "win32" else ["which", exe_name]
        result = subprocess.run(which_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            path = result.stdout.strip().splitlines()[0]
            return True, path
    except Exception:  # noqa: BLE001
        pass

    return False, None


def collect() -> CudaInfo:
    """Collect CUDA toolchain info. Never raises."""
    info = CudaInfo()

    # CUDA env vars
    info.cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    info.cuda_path = os.environ.get("CUDA_PATH")

    # nvcc
    try:
        info.nvcc_available, info.nvcc_version, info.nvcc_path = _find_nvcc_version()
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvcc detection error: %s", exc)

    # CUDA runtime
    try:
        info.cuda_runtime_available, info.cuda_runtime_version = _detect_cuda_runtime_version()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CUDA runtime detection error: %s", exc)

    # Native benchmark
    try:
        info.native_benchmark_available, info.native_benchmark_path = _find_native_benchmark()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Native benchmark search error: %s", exc)

    return info
