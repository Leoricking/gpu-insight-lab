"""
GPU Insight Lab - CPU Baseline Benchmarks
Pure Python/NumPy implementations used as reference for GPU speedup calculations.
Each runs 3 warmup + 10 measured iterations and computes statistics.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import List

from benchmarks.schemas import BenchmarkResult

logger = logging.getLogger(__name__)

_WARMUP_RUNS = 3
_MEASURED_RUNS = 10


def _compute_stats(measurements: List[float]) -> dict:
    """Compute mean, median, min, max, std_dev from a list of floats."""
    if not measurements:
        return {}
    n = len(measurements)
    mean = sum(measurements) / n
    sorted_m = sorted(measurements)
    median = sorted_m[n // 2] if n % 2 == 1 else (sorted_m[n // 2 - 1] + sorted_m[n // 2]) / 2
    min_val = sorted_m[0]
    max_val = sorted_m[-1]
    variance = sum((x - mean) ** 2 for x in measurements) / n
    std_dev = variance ** 0.5
    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min_val": round(min_val, 4),
        "max_val": round(max_val, 4),
        "standard_deviation": round(std_dev, 4),
    }


def vector_add(n: int = 1_000_000) -> BenchmarkResult:
    """
    CPU baseline: element-wise vector addition using NumPy.
    Returns BenchmarkResult with timing statistics.
    """
    result = BenchmarkResult(
        test_name="cpu_vector_add",
        data_type="float32",
        input_size=n,
        warmup_runs=_WARMUP_RUNS,
        measured_runs=_MEASURED_RUNS,
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
    )

    try:
        import numpy as np  # type: ignore  # noqa: PLC0415

        a = np.random.rand(n).astype(np.float32)
        b = np.random.rand(n).astype(np.float32)

        # Warmup
        for _ in range(_WARMUP_RUNS):
            _ = a + b

        # Measured runs
        measurements: List[float] = []
        for _ in range(_MEASURED_RUNS):
            t0 = time.perf_counter()
            c = a + b
            t1 = time.perf_counter()
            measurements.append((t1 - t0) * 1000.0)  # ms

        result.raw_measurements = measurements
        stats = _compute_stats(measurements)
        result.mean = stats.get("mean")
        result.median = stats.get("median")
        result.min_val = stats.get("min_val")
        result.max_val = stats.get("max_val")
        result.standard_deviation = stats.get("standard_deviation")
        result.cpu_time_ms = result.mean
        result.end_to_end_time_ms = result.mean
        result.correctness_pass = True

        # Throughput: elements/sec
        if result.mean and result.mean > 0:
            result.throughput = (n / (result.mean / 1000.0)) / 1e9  # Gelem/s
            # Bandwidth: read 2 arrays, write 1 array
            bytes_transferred = 3 * n * 4  # float32 = 4 bytes
            result.bandwidth_gbps = (bytes_transferred / (result.mean / 1000.0)) / 1e9

        result.notes = f"NumPy CPU baseline, n={n:,}"

    except ImportError:
        logger.warning("NumPy not available for CPU baseline vector_add")
        result.error = "NumPy not installed"
    except Exception as exc:  # noqa: BLE001
        logger.error("CPU vector_add failed: %s", exc)
        result.error = str(exc)

    return result


def matrix_multiply(m: int = 512, k: int = 512, n: int = 512) -> BenchmarkResult:
    """
    CPU baseline: matrix multiplication using NumPy.
    Computes C = A @ B where A is (m, k) and B is (k, n).
    """
    result = BenchmarkResult(
        test_name="cpu_matrix_multiply",
        data_type="float32",
        input_size=m * k + k * n,
        warmup_runs=_WARMUP_RUNS,
        measured_runs=_MEASURED_RUNS,
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        notes=f"GEMM m={m} k={k} n={n}",
    )

    try:
        import numpy as np  # type: ignore  # noqa: PLC0415

        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)

        # Warmup
        for _ in range(_WARMUP_RUNS):
            _ = a @ b

        # Measured
        measurements: List[float] = []
        for _ in range(_MEASURED_RUNS):
            t0 = time.perf_counter()
            c = a @ b
            t1 = time.perf_counter()
            measurements.append((t1 - t0) * 1000.0)

        result.raw_measurements = measurements
        stats = _compute_stats(measurements)
        result.mean = stats.get("mean")
        result.median = stats.get("median")
        result.min_val = stats.get("min_val")
        result.max_val = stats.get("max_val")
        result.standard_deviation = stats.get("standard_deviation")
        result.cpu_time_ms = result.mean
        result.end_to_end_time_ms = result.mean
        result.correctness_pass = True

        # FLOPS = 2 * m * k * n (multiply-add per element)
        flops = 2.0 * m * k * n
        if result.mean and result.mean > 0:
            result.throughput = (flops / (result.mean / 1000.0)) / 1e9  # GFLOPS

    except ImportError:
        logger.warning("NumPy not available for CPU baseline matrix_multiply")
        result.error = "NumPy not installed"
    except Exception as exc:  # noqa: BLE001
        logger.error("CPU matrix_multiply failed: %s", exc)
        result.error = str(exc)

    return result


def image_grayscale(width: int = 1920, height: int = 1080, batch: int = 10) -> BenchmarkResult:
    """
    CPU baseline: RGB-to-grayscale conversion using NumPy weighted sum.
    Weights: Y = 0.2989*R + 0.5870*G + 0.1140*B
    """
    total_pixels = width * height * batch
    result = BenchmarkResult(
        test_name="cpu_image_grayscale",
        data_type="uint8",
        input_size=total_pixels,
        warmup_runs=_WARMUP_RUNS,
        measured_runs=_MEASURED_RUNS,
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        notes=f"RGB grayscale {width}x{height} batch={batch}",
    )

    try:
        import numpy as np  # type: ignore  # noqa: PLC0415

        # Create synthetic batch of RGB images
        images = np.random.randint(0, 256, size=(batch, height, width, 3), dtype=np.uint8)
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

        # Warmup
        for _ in range(_WARMUP_RUNS):
            _ = (images.astype(np.float32) @ weights).astype(np.uint8)

        # Measured
        measurements: List[float] = []
        for _ in range(_MEASURED_RUNS):
            t0 = time.perf_counter()
            gray = (images.astype(np.float32) @ weights).astype(np.uint8)
            t1 = time.perf_counter()
            measurements.append((t1 - t0) * 1000.0)

        result.raw_measurements = measurements
        stats = _compute_stats(measurements)
        result.mean = stats.get("mean")
        result.median = stats.get("median")
        result.min_val = stats.get("min_val")
        result.max_val = stats.get("max_val")
        result.standard_deviation = stats.get("standard_deviation")
        result.cpu_time_ms = result.mean
        result.end_to_end_time_ms = result.mean
        result.correctness_pass = True

        # Throughput: images/second
        if result.mean and result.mean > 0:
            result.throughput = batch / (result.mean / 1000.0)  # images/sec

        # Bandwidth: read 3 channels, write 1
        bytes_per_batch = batch * height * width * (3 + 1)
        if result.mean and result.mean > 0:
            result.bandwidth_gbps = (bytes_per_batch / (result.mean / 1000.0)) / 1e9

    except ImportError:
        logger.warning("NumPy not available for CPU baseline image_grayscale")
        result.error = "NumPy not installed"
    except Exception as exc:  # noqa: BLE001
        logger.error("CPU image_grayscale failed: %s", exc)
        result.error = str(exc)

    return result
