"""
Smoke tests - import all modules, CLI smoke test, no GPU required.
"""

import subprocess
import sys
import unittest


class TestAllImports(unittest.TestCase):
    """Verify all packages can be imported without error."""

    def _import(self, module: str) -> None:
        import importlib
        importlib.import_module(module)

    def test_app_branding(self) -> None:
        self._import("app.branding")

    def test_app_version(self) -> None:
        self._import("app.version")

    def test_app_config(self) -> None:
        self._import("app.config")

    def test_app_features(self) -> None:
        self._import("app.features")

    def test_collectors_system(self) -> None:
        self._import("collectors.system_collector")

    def test_collectors_nvidia(self) -> None:
        self._import("collectors.nvidia_collector")

    def test_collectors_cuda(self) -> None:
        self._import("collectors.cuda_collector")

    def test_collectors_pcie(self) -> None:
        self._import("collectors.pcie_collector")

    def test_collectors_tool(self) -> None:
        self._import("collectors.tool_collector")

    def test_collectors_amd(self) -> None:
        self._import("collectors.amd_collector")

    def test_benchmarks_schemas(self) -> None:
        self._import("benchmarks.schemas")

    def test_benchmarks_native_runner(self) -> None:
        self._import("benchmarks.native_runner")

    def test_benchmarks_runner(self) -> None:
        self._import("benchmarks.runner")

    def test_benchmarks_cpu_baselines(self) -> None:
        self._import("benchmarks.cpu_baselines")

    def test_benchmarks_workload_profiles(self) -> None:
        self._import("benchmarks.workload_profiles")

    def test_diagnosis_engine(self) -> None:
        self._import("diagnosis.engine")

    def test_diagnosis_rules(self) -> None:
        self._import("diagnosis.rules")

    def test_diagnosis_scoring(self) -> None:
        self._import("diagnosis.scoring")

    def test_diagnosis_evidence(self) -> None:
        self._import("diagnosis.evidence")

    def test_diagnosis_recommendations(self) -> None:
        self._import("diagnosis.recommendations")

    def test_storage_database(self) -> None:
        self._import("storage.database")

    def test_storage_migrations(self) -> None:
        self._import("storage.migrations")

    def test_storage_models(self) -> None:
        self._import("storage.models")

    def test_reports_json(self) -> None:
        self._import("reports.json_report")

    def test_reports_csv(self) -> None:
        self._import("reports.csv_report")

    def test_reports_markdown(self) -> None:
        self._import("reports.markdown_report")

    def test_reports_html(self) -> None:
        self._import("reports.html_report")

    def test_reports_excel(self) -> None:
        self._import("reports.excel_report")

    def test_profilers_nsight_systems(self) -> None:
        self._import("profilers.nsight_systems")

    def test_profilers_nsight_compute(self) -> None:
        self._import("profilers.nsight_compute")

    def test_profilers_monitor(self) -> None:
        self._import("profilers.nvidia_smi_monitor")

    def test_profilers_rocm(self) -> None:
        self._import("profilers.rocm_profiler")

    def test_workloads_image_batch(self) -> None:
        self._import("workloads.image_batch")

    def test_workloads_media_preprocess(self) -> None:
        self._import("workloads.media_preprocess")

    def test_workloads_custom_command(self) -> None:
        self._import("workloads.custom_command")

    def test_workloads_llm(self) -> None:
        self._import("workloads.llm_benchmark")

    def test_app_cli(self) -> None:
        self._import("app.cli")

    def test_app_main(self) -> None:
        self._import("app.main")


class TestGUIImport(unittest.TestCase):
    def test_gui_main_window_import(self) -> None:
        try:
            from app.gui.main_window import MainWindow
            print("GUI import OK")
        except ImportError as exc:
            if "PySide6" in str(exc):
                self.skipTest("PySide6 not installed")
            raise


class TestCLISmoke(unittest.TestCase):
    """CLI smoke tests - no GPU required."""

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "app.cli"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
        )

    def test_cli_help(self) -> None:
        result = self._run_cli(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("GPU Insight Lab", result.stdout + result.stderr)

    def test_cli_version(self) -> None:
        result = self._run_cli(["--version"])
        self.assertIn("0.1.0", result.stdout + result.stderr)

    def test_cli_system_info_json(self) -> None:
        """system-info --json must exit 0 and return valid JSON."""
        import json as _json
        result = self._run_cli(["system-info", "--json"])
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        try:
            data = _json.loads(result.stdout)
            self.assertIn("system", data)
        except _json.JSONDecodeError as exc:
            self.fail(f"system-info --json did not return valid JSON: {exc}\nstdout: {result.stdout[:500]}")

    def test_cli_history_json(self) -> None:
        result = self._run_cli(["history", "--json"])
        # May return empty list, but should exit 0
        self.assertEqual(result.returncode, 0)

    def test_cli_no_command_shows_help(self) -> None:
        result = self._run_cli([])
        self.assertEqual(result.returncode, 0)

    def test_cli_quick_test_no_crash(self) -> None:
        """quick-test should not crash even without GPU."""
        result = self._run_cli(["quick-test", "--json", "--no-save"])
        # Exit code may be 0 or 2 (partial), but not 1 (crash)
        self.assertIn(result.returncode, [0, 2], f"stderr: {result.stderr}")


class TestBrandingConstants(unittest.TestCase):
    def test_app_name(self) -> None:
        from app.branding import APP_NAME
        self.assertEqual(APP_NAME, "GPU Insight Lab")

    def test_app_slug(self) -> None:
        from app.branding import APP_SLUG
        self.assertEqual(APP_SLUG, "gpu-insight-lab")

    def test_version(self) -> None:
        from app.branding import APP_VERSION
        self.assertEqual(APP_VERSION, "0.1.0")

    def test_no_old_brand_names(self) -> None:
        from app import branding
        content = open(branding.__file__, encoding="utf-8").read()
        self.assertNotIn("gpu_doctor", content.lower())
        self.assertNotIn("gpu workload doctor", content.lower())


if __name__ == "__main__":
    unittest.main()
