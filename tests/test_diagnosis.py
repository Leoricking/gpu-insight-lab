"""
Tests for diagnosis package.
Tests each rule with synthetic data and score calculation.
"""

import unittest
from typing import Any, Dict, List, Optional


def _make_session(
    nvidia_info: Optional[Dict] = None,
    cuda_info: Optional[Dict] = None,
    pcie_info: Optional[Dict] = None,
    tool_status: Optional[Dict] = None,
    results: Optional[List] = None,
) -> Any:
    """Create a minimal fake session for diagnosis tests."""
    class FakeSession:
        pass

    s = FakeSession()
    s.nvidia_info = nvidia_info or {}
    s.cuda_info = cuda_info or {}
    s.pcie_info = pcie_info or {}
    s.tool_status = tool_status or {}
    s.system_info = {}
    s.amd_info = {}
    s.results = results or []
    s.diagnosis_results = []
    return s


def _make_result(**kwargs: Any) -> Any:
    """Create a BenchmarkResult-like object."""
    from benchmarks.schemas import BenchmarkResult
    r = BenchmarkResult()
    for k, v in kwargs.items():
        setattr(r, k, v)
    return r


class TestRulePinnedVsPageable(unittest.TestCase):
    def test_insufficient_data_returns_info(self) -> None:
        from diagnosis.rules import rule_pinned_vs_pageable
        session = _make_session()
        result = rule_pinned_vs_pageable(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "INSUFFICIENT_DATA")

    def test_pinned_faster_returns_healthy(self) -> None:
        from diagnosis.rules import rule_pinned_vs_pageable
        r = _make_result(
            test_name="memory_bandwidth",
            notes="pinned_h2d=20.0 pageable_h2d=12.0",
        )
        session = _make_session(results=[r])
        result = rule_pinned_vs_pageable(session)
        self.assertIsNotNone(result)
        # Should note the improvement
        self.assertIn(result.category, ("HEALTHY", "INSUFFICIENT_DATA"))


class TestRuleTransferOverhead(unittest.TestCase):
    def test_high_overhead_returns_warning(self) -> None:
        from diagnosis.rules import rule_transfer_overhead
        r = _make_result(
            test_name="vector_add",
            transfer_time_ms=50.0,
            end_to_end_time_ms=100.0,
        )
        session = _make_session(results=[r])
        result = rule_transfer_overhead(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "PCIE_TRANSFER_BOUND")
        self.assertEqual(result.severity, "WARNING")

    def test_low_overhead_returns_healthy(self) -> None:
        from diagnosis.rules import rule_transfer_overhead
        r = _make_result(
            test_name="vector_add",
            transfer_time_ms=5.0,
            end_to_end_time_ms=100.0,
        )
        session = _make_session(results=[r])
        result = rule_transfer_overhead(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "HEALTHY")

    def test_no_data_returns_insufficient(self) -> None:
        from diagnosis.rules import rule_transfer_overhead
        session = _make_session()
        result = rule_transfer_overhead(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "INSUFFICIENT_DATA")


class TestRuleLaunchOverhead(unittest.TestCase):
    def test_gpu_slower_than_cpu_small_n(self) -> None:
        from diagnosis.rules import rule_launch_overhead
        r = _make_result(
            test_name="vector_add_small",
            input_size=100,
            gpu_time_ms=5.0,
            cpu_time_ms=0.1,
        )
        session = _make_session(results=[r])
        result = rule_launch_overhead(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "KERNEL_LAUNCH_OVERHEAD")

    def test_no_small_n_results_returns_none(self) -> None:
        from diagnosis.rules import rule_launch_overhead
        r = _make_result(test_name="gemm", input_size=1_000_000)
        session = _make_session(results=[r])
        result = rule_launch_overhead(session)
        self.assertIsNone(result)


class TestRuleThermalThrottle(unittest.TestCase):
    def test_high_temp_warning(self) -> None:
        from diagnosis.rules import rule_thermal_throttle
        session = _make_session(nvidia_info={"temperature_c": 90.0, "performance_state": "P4"})
        result = rule_thermal_throttle(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "THERMAL_THROTTLING")

    def test_normal_temp_healthy(self) -> None:
        from diagnosis.rules import rule_thermal_throttle
        session = _make_session(nvidia_info={"temperature_c": 65.0, "performance_state": "P0"})
        result = rule_thermal_throttle(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "HEALTHY")

    def test_no_temp_data_insufficient(self) -> None:
        from diagnosis.rules import rule_thermal_throttle
        session = _make_session(nvidia_info={})
        result = rule_thermal_throttle(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "INSUFFICIENT_DATA")


class TestRuleVRAMPressure(unittest.TestCase):
    def test_high_vram_warning(self) -> None:
        from diagnosis.rules import rule_vram_pressure
        session = _make_session(nvidia_info={"vram_total_mb": 8192.0, "vram_used_mb": 7800.0})
        result = rule_vram_pressure(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "VRAM_CAPACITY_BOUND")

    def test_low_vram_healthy(self) -> None:
        from diagnosis.rules import rule_vram_pressure
        session = _make_session(nvidia_info={"vram_total_mb": 8192.0, "vram_used_mb": 2048.0})
        result = rule_vram_pressure(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "HEALTHY")


class TestRuleDriverMismatch(unittest.TestCase):
    def test_version_mismatch_error(self) -> None:
        from diagnosis.rules import rule_driver_mismatch
        session = _make_session(
            nvidia_info={"cuda_driver_version": "12.0"},
            cuda_info={"cuda_runtime_version": "11.8"},
        )
        result = rule_driver_mismatch(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "DRIVER_RUNTIME_MISMATCH")
        self.assertEqual(result.severity, "ERROR")

    def test_version_match_healthy(self) -> None:
        from diagnosis.rules import rule_driver_mismatch
        session = _make_session(
            nvidia_info={"cuda_driver_version": "12.3"},
            cuda_info={"cuda_runtime_version": "12.3"},
        )
        result = rule_driver_mismatch(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "HEALTHY")


class TestRuleProfilerUnavailable(unittest.TestCase):
    def test_missing_profilers_info(self) -> None:
        from diagnosis.rules import rule_profiler_unavailable
        session = _make_session(tool_status={
            "nsys": {"exists": False},
            "ncu": {"exists": False},
        })
        result = rule_profiler_unavailable(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.severity, "INFO")
        self.assertEqual(result.category, "TOOLCHAIN_INCOMPLETE")

    def test_profilers_available_healthy(self) -> None:
        from diagnosis.rules import rule_profiler_unavailable
        session = _make_session(tool_status={
            "nsys": {"exists": True, "version": "2023.3"},
            "ncu": {"exists": True, "version": "2023.3"},
        })
        result = rule_profiler_unavailable(session)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "HEALTHY")


class TestAllRulesHaveEvidence(unittest.TestCase):
    def test_all_results_have_evidence(self) -> None:
        """Every DiagnosisResult must have a non-empty evidence string."""
        from diagnosis.rules import (  # noqa: PLC0415
            rule_pinned_vs_pageable,
            rule_transfer_overhead,
            rule_launch_overhead,
            rule_thermal_throttle,
            rule_vram_pressure,
            rule_driver_mismatch,
            rule_profiler_unavailable,
            rule_toolchain_completeness,
        )
        session = _make_session()
        for rule_fn in [
            rule_pinned_vs_pageable,
            rule_transfer_overhead,
            rule_thermal_throttle,
            rule_vram_pressure,
            rule_driver_mismatch,
            rule_profiler_unavailable,
            rule_toolchain_completeness,
        ]:
            result = rule_fn(session)
            if result is not None:
                self.assertTrue(
                    result.evidence,
                    f"Rule {rule_fn.__name__} returned empty evidence",
                )


class TestScoreCalculation(unittest.TestCase):
    def test_score_range(self) -> None:
        from diagnosis.scoring import compute_score
        session = _make_session()
        score = compute_score(session)
        self.assertGreaterEqual(score.score, 0.0)
        self.assertLessEqual(score.score, 100.0)

    def test_score_confidence_range(self) -> None:
        from diagnosis.scoring import compute_score
        session = _make_session()
        score = compute_score(session)
        self.assertGreaterEqual(score.confidence, 0.0)
        self.assertLessEqual(score.confidence, 1.0)

    def test_score_with_gpu_data(self) -> None:
        from diagnosis.scoring import compute_score
        session = _make_session(
            nvidia_info={
                "available": True,
                "gpu_name": "Test GPU",
                "temperature_c": 60.0,
                "performance_state": "P0",
                "vram_total_mb": 8192.0,
                "vram_used_mb": 1024.0,
            },
            cuda_info={"nvcc_available": True, "nvcc_version": "12.3", "native_benchmark_available": True},
            tool_status={
                "nvcc": {"exists": True},
                "cmake": {"exists": True},
                "nvidia-smi": {"exists": True},
            },
        )
        score = compute_score(session)
        # With GPU data, score should be higher than with no data
        empty_session = _make_session()
        empty_score = compute_score(empty_session)
        self.assertGreater(score.score, empty_score.score)

    def test_missing_data_listed(self) -> None:
        from diagnosis.scoring import compute_score
        session = _make_session()
        score = compute_score(session)
        self.assertIsInstance(score.missing_data, list)

    def test_deductions_structure(self) -> None:
        from diagnosis.scoring import compute_score
        session = _make_session()
        score = compute_score(session)
        for d in score.deductions:
            self.assertIn("category", d)
            self.assertIn("reason", d)


if __name__ == "__main__":
    unittest.main()
