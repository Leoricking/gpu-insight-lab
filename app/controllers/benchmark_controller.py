"""GPU Insight Lab - Benchmark Controller (thin layer between GUI/CLI and runner)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from benchmarks.runner import run_quick_test, run_full_test, run_single_test
from benchmarks.schemas import BenchmarkSession, BenchmarkResult


def quick_test(
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkSession:
    return run_quick_test(progress_callback=progress_callback)


def full_test(
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkSession:
    return run_full_test(progress_callback=progress_callback)


def single_test(
    test_name: str,
    repeat: int = 10,
    block_size: Optional[int] = None,
    data_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkResult:
    return run_single_test(
        test_name,
        repeat=repeat,
        block_size=block_size,
        data_size=data_size,
        progress_callback=progress_callback,
    )
