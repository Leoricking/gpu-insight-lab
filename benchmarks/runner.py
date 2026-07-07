"""
GPU Insight Lab - Benchmark Session Orchestrator
Collects system info, runs collectors, runs benchmarks, produces BenchmarkSession.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from benchmarks.schemas import BenchmarkResult, BenchmarkSession
from benchmarks import cpu_baselines, native_runner
from benchmarks.workload_profiles import get_full_profiles, get_quick_profiles, get_profile

logger = logging.getLogger(__name__)


def _collect_system_snapshot() -> Dict[str, Any]:
    """Collect all system info, return as plain dict. Never raises."""
    snapshot: Dict[str, Any] = {}

    from collectors import system_collector, nvidia_collector, cuda_collector
    from collectors import pcie_collector, tool_collector, amd_collector

    try:
        sys_info = system_collector.collect()
        snapshot["system"] = dataclasses.asdict(sys_info)
    except Exception as exc:  # noqa: BLE001
        snapshot["system"] = {"error": str(exc)}

    try:
        nv_info = nvidia_collector.collect()
        snapshot["nvidia"] = dataclasses.asdict(nv_info)
    except Exception as exc:  # noqa: BLE001
        snapshot["nvidia"] = {"error": str(exc)}

    try:
        cuda_info = cuda_collector.collect()
        snapshot["cuda"] = dataclasses.asdict(cuda_info)
    except Exception as exc:  # noqa: BLE001
        snapshot["cuda"] = {"error": str(exc)}

    try:
        pcie_info = pcie_collector.collect()
        snapshot["pcie"] = dataclasses.asdict(pcie_info)
    except Exception as exc:  # noqa: BLE001
        snapshot["pcie"] = {"error": str(exc)}

    try:
        tools = tool_collector.collect()
        snapshot["tools"] = {k: dataclasses.asdict(v) for k, v in tools.items()}
    except Exception as exc:  # noqa: BLE001
        snapshot["tools"] = {"error": str(exc)}

    try:
        amd_info = amd_collector.collect()
        snapshot["amd"] = dataclasses.asdict(amd_info)
    except Exception as exc:  # noqa: BLE001
        snapshot["amd"] = {"error": str(exc)}

    return snapshot


def _native_result_to_benchmark_result(raw: Dict[str, Any]) -> BenchmarkResult:
    """Convert native JSON dict to BenchmarkResult dataclass."""
    r = BenchmarkResult()
    field_names = {f.name for f in dataclasses.fields(r)}
    for k, v in raw.items():
        if k in field_names:
            setattr(r, k, v)
    # Ensure test_name is populated
    if not r.test_name and "name" in raw:
        r.test_name = raw["name"]
    return r


def run_quick_test(
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkSession:
    """
    Run quick benchmark suite:
    - System collection
    - CPU baselines (vector_add)
    - Native quick tests (if executable available)
    """
    session = BenchmarkSession(
        session_id=str(uuid.uuid4()),
        session_name="Quick Test",
        started_at=time.time(),
        status="running",
    )

    def _progress(pct: int, msg: str) -> None:
        logger.info("[%d%%] %s", pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

    try:
        _progress(5, "Collecting system information...")
        snapshot = _collect_system_snapshot()
        session.system_info = snapshot.get("system", {})
        session.nvidia_info = snapshot.get("nvidia", {})
        session.cuda_info = snapshot.get("cuda", {})
        session.pcie_info = snapshot.get("pcie", {})
        session.tool_status = snapshot.get("tools", {})
        session.amd_info = snapshot.get("amd", {})

        _progress(20, "Running CPU baseline (vector add)...")
        cpu_result = cpu_baselines.vector_add(n=1_000_000)
        session.results.append(cpu_result)

        # Native benchmarks
        if native_runner.is_available():
            _progress(40, "Running native device info...")
            dev_info = native_runner.run_device_info()
            if dev_info and "error" not in dev_info:
                r = _native_result_to_benchmark_result(dev_info)
                r.test_name = r.test_name or "device_info"
                session.results.append(r)

            _progress(55, "Running native quick benchmarks...")
            native_results = native_runner.run_quick()
            for raw in native_results:
                r = _native_result_to_benchmark_result(raw)
                session.results.append(r)
            _progress(85, f"Native benchmarks complete ({len(native_results)} tests)")
        else:
            _progress(85, "Native executable not found; skipping GPU kernels")

        _progress(92, "Running diagnosis...")
        from diagnosis.engine import run_diagnosis  # noqa: PLC0415
        session.diagnosis_results = run_diagnosis(session)

        _progress(97, "Computing GPU Insight Score...")
        from diagnosis.scoring import compute_score  # noqa: PLC0415
        score_result = compute_score(session)
        session.health_score = score_result.score
        session.score_confidence = score_result.confidence
        session.score_details = dataclasses.asdict(score_result)

        session.completed_at = time.time()
        session.status = "completed"
        _progress(100, "Quick test complete")

    except Exception as exc:  # noqa: BLE001
        logger.error("Quick test failed: %s", exc)
        session.status = "failed"
        session.error = str(exc)
        session.completed_at = time.time()

    return session


def run_full_test(
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkSession:
    """Run full benchmark suite including all profiles."""
    session = BenchmarkSession(
        session_id=str(uuid.uuid4()),
        session_name="Full Test",
        started_at=time.time(),
        status="running",
    )

    def _progress(pct: int, msg: str) -> None:
        logger.info("[%d%%] %s", pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

    try:
        _progress(2, "Collecting system information...")
        snapshot = _collect_system_snapshot()
        session.system_info = snapshot.get("system", {})
        session.nvidia_info = snapshot.get("nvidia", {})
        session.cuda_info = snapshot.get("cuda", {})
        session.pcie_info = snapshot.get("pcie", {})
        session.tool_status = snapshot.get("tools", {})
        session.amd_info = snapshot.get("amd", {})

        _progress(10, "Running CPU baselines...")
        session.results.append(cpu_baselines.vector_add())
        session.results.append(cpu_baselines.matrix_multiply())
        session.results.append(cpu_baselines.image_grayscale())

        if native_runner.is_available():
            _progress(25, "Running native full benchmark suite...")
            native_results = native_runner.run_full()
            total = len(native_results)
            for i, raw in enumerate(native_results):
                r = _native_result_to_benchmark_result(raw)
                session.results.append(r)
                pct = 25 + int((i + 1) / max(total, 1) * 55)
                _progress(pct, f"Native: {r.test_name or 'unknown'}")
        else:
            _progress(80, "Native executable not found; CPU-only mode")

        _progress(85, "Running diagnosis engine...")
        from diagnosis.engine import run_diagnosis  # noqa: PLC0415
        session.diagnosis_results = run_diagnosis(session)

        _progress(95, "Computing GPU Insight Score...")
        from diagnosis.scoring import compute_score  # noqa: PLC0415
        import dataclasses as dc  # noqa: PLC0415
        score_result = compute_score(session)
        session.health_score = score_result.score
        session.score_confidence = score_result.confidence
        session.score_details = dc.asdict(score_result)

        session.completed_at = time.time()
        session.status = "completed"
        _progress(100, "Full test complete")

    except Exception as exc:  # noqa: BLE001
        logger.error("Full test failed: %s", exc)
        session.status = "failed"
        session.error = str(exc)
        session.completed_at = time.time()

    return session


def run_single_test(
    test_name: str,
    repeat: int = 10,
    block_size: Optional[int] = None,
    data_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BenchmarkResult:
    """Run a single named test. Returns BenchmarkResult."""

    def _progress(pct: int, msg: str) -> None:
        logger.info("[%d%%] %s", pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

    _progress(10, f"Running test: {test_name}")

    # CPU baselines
    if test_name == "cpu_vector_add":
        n = data_size or 1_000_000
        return cpu_baselines.vector_add(n=n)
    elif test_name == "cpu_matrix_multiply":
        s = data_size or 512
        return cpu_baselines.matrix_multiply(m=s, k=s, n=s)
    elif test_name == "cpu_image_grayscale":
        return cpu_baselines.image_grayscale()

    # Native tests
    if native_runner.is_available():
        profile = get_profile(test_name)
        timeout = profile.timeout_seconds if profile else 120
        raw = native_runner.run_test(
            test_name,
            repeat=repeat,
            block_size=block_size,
            data_size=data_size,
            timeout=timeout,
        )
        if raw:
            r = _native_result_to_benchmark_result(raw)
            _progress(100, f"Test {test_name} complete")
            return r

    # Fallback: native not available — return SKIPPED, never raise
    result = BenchmarkResult(
        test_name=test_name,
        notes="SKIPPED — native executable not found or CUDA unavailable",
        error=f"Test '{test_name}' not available: native executable not found",
    )
    return result
