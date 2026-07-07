"""
Tests that native source files exist and CMakeLists.txt target name is correct.
No GPU required.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
CUDA_DIR = ROOT / "native" / "cuda"
HIP_DIR = ROOT / "native" / "hip"
NATIVE_CMAKE = ROOT / "native" / "CMakeLists.txt"


class TestCUDAFileManifest(unittest.TestCase):
    """All expected CUDA source files must exist."""

    _REQUIRED = [
        "vector_add.cu",
        "reduction.cu",
        "transpose.cu",
        "gemm_naive.cu",
        "gemm_tiled.cu",
        "memory_bandwidth.cu",
        "stream_pipeline.cu",
        "image_grayscale.cu",
        "device_info.cu",
        "benchmark_main.cu",
        "prefix_sum.cu",
        "convolution_2d.cu",
    ]

    def _check(self, filename: str) -> None:
        path = CUDA_DIR / filename
        self.assertTrue(path.exists(), f"Missing CUDA source: {path}")

    def test_vector_add(self) -> None:
        self._check("vector_add.cu")

    def test_reduction(self) -> None:
        self._check("reduction.cu")

    def test_transpose(self) -> None:
        self._check("transpose.cu")

    def test_gemm_naive(self) -> None:
        self._check("gemm_naive.cu")

    def test_gemm_tiled(self) -> None:
        self._check("gemm_tiled.cu")

    def test_memory_bandwidth(self) -> None:
        self._check("memory_bandwidth.cu")

    def test_stream_pipeline(self) -> None:
        self._check("stream_pipeline.cu")

    def test_image_grayscale(self) -> None:
        self._check("image_grayscale.cu")

    def test_device_info(self) -> None:
        self._check("device_info.cu")

    def test_benchmark_main(self) -> None:
        self._check("benchmark_main.cu")

    def test_prefix_sum(self) -> None:
        self._check("prefix_sum.cu")

    def test_convolution_2d(self) -> None:
        self._check("convolution_2d.cu")


class TestHIPFileManifest(unittest.TestCase):
    """All expected HIP source files must exist."""

    def test_vector_add_hip(self) -> None:
        self.assertTrue((HIP_DIR / "vector_add_hip.cpp").exists())

    def test_reduction_hip(self) -> None:
        self.assertTrue((HIP_DIR / "reduction_hip.cpp").exists())

    def test_gemm_naive_hip(self) -> None:
        self.assertTrue((HIP_DIR / "gemm_naive_hip.cpp").exists())


class TestCMakeTarget(unittest.TestCase):
    """CMakeLists.txt must declare target gpu_insight_benchmark."""

    def test_target_name(self) -> None:
        self.assertTrue(NATIVE_CMAKE.exists(), "native/CMakeLists.txt not found")
        content = NATIVE_CMAKE.read_text(encoding="utf-8")
        self.assertIn("gpu_insight_benchmark", content,
                      "CMakeLists.txt must define target gpu_insight_benchmark")

    def test_target_is_add_executable(self) -> None:
        content = NATIVE_CMAKE.read_text(encoding="utf-8")
        # Should have add_executable(gpu_insight_benchmark ...)
        self.assertRegex(content, r"add_executable\s*\(\s*gpu_insight_benchmark")


if __name__ == "__main__":
    unittest.main()
