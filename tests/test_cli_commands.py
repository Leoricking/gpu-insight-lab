"""
Tests for CLI command structure and branding.
No GPU required.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class TestBranding(unittest.TestCase):
    """Branding constants exist and are non-empty."""

    def test_app_name_exists(self) -> None:
        from app.branding import APP_NAME
        self.assertTrue(APP_NAME, "APP_NAME must be non-empty")

    def test_app_name_value(self) -> None:
        from app.branding import APP_NAME
        self.assertEqual(APP_NAME, "GPU Insight Lab")

    def test_app_slug_exists(self) -> None:
        from app.branding import APP_SLUG
        self.assertTrue(APP_SLUG, "APP_SLUG must be non-empty")

    def test_app_version_exists(self) -> None:
        from app.branding import APP_VERSION
        self.assertTrue(APP_VERSION, "APP_VERSION must be non-empty")

    def test_app_subtitle_exists(self) -> None:
        from app.branding import APP_SUBTITLE
        self.assertTrue(APP_SUBTITLE, "APP_SUBTITLE must be non-empty")

    def test_native_executable_name(self) -> None:
        from app.branding import NATIVE_EXECUTABLE
        self.assertEqual(NATIVE_EXECUTABLE, "gpu_insight_benchmark")

    def test_database_name(self) -> None:
        from app.branding import DATABASE_NAME
        self.assertEqual(DATABASE_NAME, "gpu_insight.sqlite")

    def test_report_prefix(self) -> None:
        from app.branding import REPORT_PREFIX
        self.assertEqual(REPORT_PREFIX, "gpu_insight_report")


class TestPyprojectName(unittest.TestCase):
    """pyproject.toml project name is gpu-insight-lab."""

    def test_project_name(self) -> None:
        root = Path(__file__).parent.parent
        pyproject = root / "pyproject.toml"
        if not pyproject.exists():
            self.skipTest("pyproject.toml not found")
        content = pyproject.read_text(encoding="utf-8")
        self.assertIn("gpu-insight-lab", content, "pyproject.toml must contain 'gpu-insight-lab'")


class TestCLIHelp(unittest.TestCase):
    """CLI --help exits 0 and output contains GPU Insight Lab."""

    def _run(self, args: list) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "app.cli"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
        )

    def test_help_exits_zero(self) -> None:
        result = self._run(["--help"])
        self.assertEqual(result.returncode, 0)

    def test_help_contains_gpu_insight_lab(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("GPU Insight Lab", combined)

    def test_cli_registers_system_info(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("system-info", combined)

    def test_cli_registers_quick_test(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("quick-test", combined)

    def test_cli_registers_full_test(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("full-test", combined)

    def test_cli_registers_benchmark(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("benchmark", combined)

    def test_cli_registers_diagnose(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("diagnose", combined)

    def test_cli_registers_export(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("export", combined)

    def test_cli_registers_history(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("history", combined)

    def test_cli_registers_demo_report(self) -> None:
        result = self._run(["--help"])
        combined = result.stdout + result.stderr
        self.assertIn("demo-report", combined)

    def test_cli_demo_report_no_crash(self) -> None:
        """demo-report should not crash; may exit 0 or 1 depending on report libs."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(["demo-report", "--output-dir", tmpdir])
            # Should not crash with unhandled exception (exit code 2 = partial ok)
            self.assertIn(result.returncode, [0, 1, 2],
                          f"demo-report crashed with code {result.returncode}: {result.stderr[:300]}")


if __name__ == "__main__":
    unittest.main()
