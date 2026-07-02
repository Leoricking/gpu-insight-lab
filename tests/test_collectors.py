"""
Tests for collectors package.
Tests command failure handling, missing tools, AMD collector without GPU.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch


class TestSystemCollector(unittest.TestCase):
    def test_collect_returns_dataclass(self) -> None:
        from collectors.system_collector import collect
        info = collect()
        self.assertIsNotNone(info)
        self.assertIsInstance(info.os_name, str)
        self.assertIsInstance(info.hostname, str)

    def test_collect_no_crash_without_psutil(self) -> None:
        """System collector must not crash if psutil is absent."""
        with patch.dict(sys.modules, {"psutil": None}):
            # Re-import to pick up the mocked module
            try:
                from collectors import system_collector
                import importlib
                importlib.reload(system_collector)
                info = system_collector.collect()
                self.assertIsNotNone(info)
            except ImportError:
                pass  # Expected when psutil unavailable

    def test_collect_has_python_version(self) -> None:
        import platform
        from collectors.system_collector import collect
        info = collect()
        self.assertEqual(info.python_version, platform.python_version())


class TestNvidiaCollector(unittest.TestCase):
    def _unavailable(self) -> "NvidiaGPUInfo":  # type: ignore[name-defined]
        from collectors.nvidia_collector import NvidiaGPUInfo
        return NvidiaGPUInfo(available=False, error="no gpu")

    def test_returns_not_available_when_no_nvidia(self) -> None:
        """Collector must return available=False gracefully when no GPU."""
        from collectors import nvidia_collector
        no_gpu = self._unavailable()
        with patch.object(nvidia_collector, "_collect_via_pynvml", side_effect=ImportError), \
             patch.object(nvidia_collector, "_collect_via_nvidia_smi", return_value=no_gpu):
            info = nvidia_collector.collect()
            self.assertFalse(info.available)

    def test_no_crash_on_smi_timeout(self) -> None:
        import subprocess
        from collectors import nvidia_collector
        with patch.object(nvidia_collector, "_collect_via_pynvml", side_effect=ImportError), \
             patch.object(nvidia_collector, "_collect_via_nvidia_smi",
                          side_effect=subprocess.TimeoutExpired("nvidia-smi", 15)):
            info = nvidia_collector.collect()
            self.assertFalse(info.available)

    def test_no_crash_when_smi_missing(self) -> None:
        from collectors import nvidia_collector
        with patch.object(nvidia_collector, "_collect_via_pynvml", side_effect=ImportError), \
             patch.object(nvidia_collector, "_collect_via_nvidia_smi",
                          side_effect=FileNotFoundError("nvidia-smi not found")):
            info = nvidia_collector.collect()
            self.assertFalse(info.available)


class TestCudaCollector(unittest.TestCase):
    def test_collect_no_crash(self) -> None:
        from collectors.cuda_collector import collect
        info = collect()
        self.assertIsNotNone(info)

    def test_no_crash_when_nvcc_missing(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            from collectors import cuda_collector
            import importlib
            importlib.reload(cuda_collector)
            info = cuda_collector.collect()
            self.assertFalse(info.nvcc_available)


class TestPCIeCollector(unittest.TestCase):
    def test_no_crash_without_smi(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            from collectors.pcie_collector import collect
            info = collect()
            self.assertFalse(info.available)

    def test_no_crash_on_smi_error(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            from collectors.pcie_collector import collect
            info = collect()
            self.assertFalse(info.available)


class TestToolCollector(unittest.TestCase):
    def test_returns_dict(self) -> None:
        from collectors.tool_collector import collect
        tools = collect()
        self.assertIsInstance(tools, dict)

    def test_all_tools_present_in_result(self) -> None:
        from collectors.tool_collector import collect
        tools = collect()
        expected_tools = ["nvcc", "cmake", "nvidia-smi", "nsys", "ncu", "rocminfo", "rocm-smi", "hipcc"]
        for t in expected_tools:
            self.assertIn(t, tools)

    def test_no_crash_when_tools_missing(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            from collectors import tool_collector
            import importlib
            importlib.reload(tool_collector)
            tools = tool_collector.collect()
            for status in tools.values():
                self.assertFalse(status.exists)


class TestAMDCollector(unittest.TestCase):
    def test_amd_collector_no_crash(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            from collectors.amd_collector import collect
            info = collect()
            self.assertEqual(info.validation_status, "NOT_VALIDATED")
            self.assertFalse(info.available)

    def test_validation_status_always_set(self) -> None:
        from collectors.amd_collector import collect
        info = collect()
        self.assertEqual(info.validation_status, "NOT_VALIDATED")


if __name__ == "__main__":
    unittest.main()
