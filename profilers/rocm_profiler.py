"""
GPU Insight Lab - ROCm Profiler
Stub profiler for AMD ROCm. Status: NOT_VALIDATED in v0.1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

VALIDATION_STATUS = "NOT_VALIDATED"


@dataclass
class ROCmProfileResult:
    status: str = "unavailable"
    validation_status: str = VALIDATION_STATUS
    error: Optional[str] = None
    note: str = (
        "ROCm profiling is NOT_VALIDATED in GPU Insight Lab v0.1.0. "
        "No AMD GPU was present during development. "
        "rocminfo and rocm-smi calls are included in tool_collector.py but untested."
    )


def profile(
    benchmark_exe: str = "",
    test_name: str = "",
    output_dir: str = "",
    timeout: int = 60,
) -> ROCmProfileResult:
    """
    Placeholder for ROCm profiling.
    Returns NOT_VALIDATED status without crashing.
    """
    return ROCmProfileResult(
        status="unavailable",
        validation_status=VALIDATION_STATUS,
        error=None,
    )
