"""
Tests for statistics computations (CPU baselines stats and native schemas).
"""

import unittest
import math


class TestCPUBaselineStatistics(unittest.TestCase):
    """Test the _compute_stats function used in cpu_baselines."""

    def _compute_stats(self, data: list) -> dict:
        from benchmarks.cpu_baselines import _compute_stats
        return _compute_stats(data)

    def test_mean(self) -> None:
        stats = self._compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(stats["mean"], 3.0, places=4)

    def test_median_odd(self) -> None:
        stats = self._compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(stats["median"], 3.0, places=4)

    def test_median_even(self) -> None:
        stats = self._compute_stats([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats["median"], 2.5, places=4)

    def test_min_val(self) -> None:
        stats = self._compute_stats([5.0, 3.0, 1.0, 4.0, 2.0])
        self.assertAlmostEqual(stats["min_val"], 1.0, places=4)

    def test_max_val(self) -> None:
        stats = self._compute_stats([5.0, 3.0, 1.0, 4.0, 2.0])
        self.assertAlmostEqual(stats["max_val"], 5.0, places=4)

    def test_std_dev_known(self) -> None:
        # Population std dev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = self._compute_stats(data)
        self.assertAlmostEqual(stats["standard_deviation"], 2.0, places=2)

    def test_std_dev_single_value(self) -> None:
        stats = self._compute_stats([5.0])
        self.assertAlmostEqual(stats["standard_deviation"], 0.0, places=4)

    def test_empty_returns_empty_dict(self) -> None:
        stats = self._compute_stats([])
        self.assertEqual(stats, {})

    def test_all_same_values(self) -> None:
        stats = self._compute_stats([3.14] * 10)
        self.assertAlmostEqual(stats["mean"], 3.14, places=4)
        self.assertAlmostEqual(stats["standard_deviation"], 0.0, places=4)

    def test_large_dataset(self) -> None:
        data = list(range(1, 101))  # 1..100
        stats = self._compute_stats([float(x) for x in data])
        self.assertAlmostEqual(stats["mean"], 50.5, places=2)
        self.assertAlmostEqual(stats["min_val"], 1.0, places=4)
        self.assertAlmostEqual(stats["max_val"], 100.0, places=4)


class TestBenchmarkResultStats(unittest.TestCase):
    """Test that cpu_baselines properly populates statistics in BenchmarkResult."""

    def test_vector_add_has_statistics(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not installed")
        from benchmarks.cpu_baselines import vector_add
        result = vector_add(n=10000)
        self.assertIsNotNone(result.mean)
        self.assertIsNotNone(result.median)
        self.assertIsNotNone(result.min_val)
        self.assertIsNotNone(result.max_val)
        self.assertIsNotNone(result.standard_deviation)
        self.assertEqual(len(result.raw_measurements), 10)

    def test_vector_add_mean_positive(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not installed")
        from benchmarks.cpu_baselines import vector_add
        result = vector_add(n=10000)
        if result.error is None:
            self.assertGreater(result.mean, 0.0)

    def test_matrix_multiply_throughput(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not installed")
        from benchmarks.cpu_baselines import matrix_multiply
        result = matrix_multiply(m=64, k=64, n=64)
        if result.error is None:
            self.assertIsNotNone(result.throughput)
            self.assertGreater(result.throughput, 0.0)

    def test_correctness_pass_set(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not installed")
        from benchmarks.cpu_baselines import vector_add
        result = vector_add(n=1000)
        if result.error is None:
            self.assertTrue(result.correctness_pass)

    def test_statistics_consistency(self) -> None:
        """min <= median <= max, mean in [min, max]."""
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not installed")
        from benchmarks.cpu_baselines import vector_add
        result = vector_add(n=50000)
        if result.error is None and result.mean is not None:
            self.assertLessEqual(result.min_val, result.median)
            self.assertLessEqual(result.median, result.max_val)
            self.assertGreaterEqual(result.mean, result.min_val)
            self.assertLessEqual(result.mean, result.max_val)


if __name__ == "__main__":
    unittest.main()
