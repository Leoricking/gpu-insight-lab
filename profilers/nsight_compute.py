"""
GPU Insight Lab - Nsight Compute Profiler
Launches ncu if available. Returns status=unavailable if not.
Warns about duration: ncu profiling significantly increases kernel execution time.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NcuResult:
    status: str = "unavailable"  # "ok", "unavailable", "error"
    report_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    ncu_version: Optional[str] = None
    warning: str = ""
    error: Optional[str] = None


# Conservative metric set - avoid ones that cause very long profiling
_DEFAULT_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
])


def find_ncu() -> Optional[Path]:
    """Search for ncu executable."""
    try:
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
        )
        if result.returncode == 0:
            return Path("ncu")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def profile_kernel(
    benchmark_exe: Path,
    test_name: str,
    output_dir: Path,
    ncu_path: Optional[Path] = None,
    timeout: int = 120,
) -> NcuResult:
    """
    Run ncu to profile a single kernel test.
    Returns NcuResult with status=unavailable if ncu is not installed.

    WARNING: ncu profiling can make kernels 10-100x slower due to replay mode.
    """
    ncu_cmd = str(ncu_path) if ncu_path else "ncu"

    # Check availability
    try:
        version_result = subprocess.run(
            [ncu_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
        )
        if version_result.returncode != 0:
            return NcuResult(
                status="unavailable",
                error="ncu returned non-zero for --version",
            )
        ncu_version = (version_result.stdout + version_result.stderr).strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return NcuResult(
            status="unavailable",
            error=f"ncu not found or timed out: {exc}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    rep_path = output_dir / f"ncu_{test_name}_{ts}.ncu-rep"

    cmd = [
        ncu_cmd,
        "--output", str(rep_path),
        "--force-overwrite",
        "--metrics", _DEFAULT_METRICS,
        "--launch-count", "1",   # Only profile 1 launch to limit duration
        str(benchmark_exe),
        "--test", test_name,
        "--repeat", "1",          # Minimal repeat to reduce profiling time
    ]
    cmd_str = " ".join(cmd)
    logger.info("Running ncu: %s", cmd_str)

    warning = (
        "ncu profiling can significantly increase kernel execution time due to replay mode. "
        "The metrics collected here use a conservative set to minimize duration."
    )
    logger.warning(warning)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return NcuResult(
            status="error",
            command=cmd_str,
            error=f"ncu timed out after {timeout}s",
            ncu_version=ncu_version,
            warning=warning,
        )
    except Exception as exc:  # noqa: BLE001
        return NcuResult(
            status="error",
            command=cmd_str,
            error=str(exc),
            ncu_version=ncu_version,
            warning=warning,
        )

    if result.returncode != 0:
        return NcuResult(
            status="error",
            report_path=str(rep_path) if rep_path.exists() else None,
            stdout=result.stdout[:2000],
            stderr=result.stderr[:2000],
            command=cmd_str,
            ncu_version=ncu_version,
            warning=warning,
            error=f"ncu exited with code {result.returncode}",
        )

    return NcuResult(
        status="ok",
        report_path=str(rep_path) if rep_path.exists() else None,
        stdout=result.stdout[:2000],
        stderr=result.stderr[:2000],
        command=cmd_str,
        ncu_version=ncu_version,
        warning=warning,
    )
