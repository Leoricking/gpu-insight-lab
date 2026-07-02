"""
GPU Insight Lab - Nsight Systems Profiler
Launches nsys if available; returns status=unavailable if not.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NsysResult:
    status: str = "unavailable"   # "ok", "unavailable", "error"
    report_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    nsys_version: Optional[str] = None
    error: Optional[str] = None


def find_nsys() -> Optional[Path]:
    """Search for nsys executable."""
    candidates = ["nsys"]
    if sys.platform == "win32":
        program_files = Path(r"C:/Program Files/NVIDIA Corporation/Nsight Systems 2023.3.1/target-windows-x64")
        if program_files.exists():
            candidates.insert(0, str(program_files / "nsys.exe"))

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=8,
                encoding="utf-8",
            )
            if result.returncode == 0:
                return Path(candidate) if candidate == "nsys" else Path(candidate)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def get_version(nsys_path: Optional[Path] = None) -> Optional[str]:
    """Return nsys version string."""
    cmd = str(nsys_path) if nsys_path else "nsys"
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
        )
        if result.returncode == 0:
            output = (result.stdout + result.stderr).strip()
            return output.splitlines()[0][:100] if output else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def profile_benchmark(
    benchmark_exe: Path,
    test_name: str,
    output_dir: Path,
    repeat: int = 5,
    nsys_path: Optional[Path] = None,
    timeout: int = 120,
) -> NsysResult:
    """
    Run nsys to profile a benchmark test.
    Returns NsysResult with status=unavailable if nsys is not installed.
    """
    nsys_cmd = str(nsys_path) if nsys_path else "nsys"

    # Check availability
    try:
        version_result = subprocess.run(
            [nsys_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
        )
        if version_result.returncode != 0:
            return NsysResult(
                status="unavailable",
                error="nsys returned non-zero for --version",
            )
        nsys_version = (version_result.stdout + version_result.stderr).strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return NsysResult(
            status="unavailable",
            error=f"nsys not found or timed out: {exc}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    rep_path = output_dir / f"nsys_{test_name}_{ts}.nsys-rep"

    cmd = [
        nsys_cmd, "profile",
        "--output", str(rep_path).replace(".nsys-rep", ""),
        "--force-overwrite", "true",
        "--trace", "cuda,nvtx",
        str(benchmark_exe),
        "--test", test_name,
        "--repeat", str(repeat),
    ]
    cmd_str = " ".join(cmd)
    logger.info("Running nsys: %s", cmd_str)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return NsysResult(
            status="error",
            command=cmd_str,
            error=f"nsys timed out after {timeout}s",
            nsys_version=nsys_version,
        )
    except Exception as exc:  # noqa: BLE001
        return NsysResult(
            status="error",
            command=cmd_str,
            error=str(exc),
            nsys_version=nsys_version,
        )

    if result.returncode != 0:
        return NsysResult(
            status="error",
            report_path=str(rep_path) if rep_path.exists() else None,
            stdout=result.stdout[:2000],
            stderr=result.stderr[:2000],
            command=cmd_str,
            nsys_version=nsys_version,
            error=f"nsys exited with code {result.returncode}",
        )

    return NsysResult(
        status="ok",
        report_path=str(rep_path) if rep_path.exists() else None,
        stdout=result.stdout[:2000],
        stderr=result.stderr[:2000],
        command=cmd_str,
        nsys_version=nsys_version,
    )
