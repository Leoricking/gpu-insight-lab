"""
Tests for CODE_COVERAGE_AUDIT.md existence and content.
Also scans for old brand residue.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
AUDIT_FILE = ROOT / "docs" / "CODE_COVERAGE_AUDIT.md"

# File extensions to scan for old brand residue
_SCAN_PATTERNS = ["*.py", "*.md", "*.cu", "*.cpp", "*.txt", "*.cmake", "*.toml"]

# Old names that must not appear (except in migration notes / audit docs themselves)
_FORBIDDEN_STRINGS = [
    "GPU Workload Doctor",
    "GPU Doctor",
    "gpu_doctor_benchmark",
]
# Note: benchmark.sqlite is checked separately because it's allowed in migration code as a warning
_FORBIDDEN_DB = "benchmark.sqlite"


class TestCodeCoverageAudit(unittest.TestCase):
    """docs/CODE_COVERAGE_AUDIT.md must exist and contain required content."""

    def test_audit_file_exists(self) -> None:
        self.assertTrue(AUDIT_FILE.exists(),
                        f"docs/CODE_COVERAGE_AUDIT.md not found at {AUDIT_FILE}")

    def test_audit_contains_implemented(self) -> None:
        content = AUDIT_FILE.read_text(encoding="utf-8")
        self.assertIn("IMPLEMENTED", content,
                      "CODE_COVERAGE_AUDIT.md must contain 'IMPLEMENTED'")

    def test_audit_contains_roadmap(self) -> None:
        content = AUDIT_FILE.read_text(encoding="utf-8")
        self.assertIn("ROADMAP", content,
                      "CODE_COVERAGE_AUDIT.md must contain 'ROADMAP'")

    def test_audit_contains_partial(self) -> None:
        content = AUDIT_FILE.read_text(encoding="utf-8")
        self.assertIn("PARTIAL", content)

    def test_audit_has_feature_table(self) -> None:
        content = AUDIT_FILE.read_text(encoding="utf-8")
        # Should have a markdown table with the feature columns
        self.assertIn("| Claimed feature", content)


class TestNoBrandResidue(unittest.TestCase):
    """No source file should contain old brand names outside of migration notes."""

    def _collect_files(self):
        files = []
        for pattern in _SCAN_PATTERNS:
            files.extend(ROOT.rglob(pattern))
        # Exclude the audit file itself and migration notes (they're allowed to mention old names)
        exclusions = {
            AUDIT_FILE,
            ROOT / "storage" / "database.py",  # has legacy name in migration warning comment
            ROOT / "tests" / "test_code_coverage_audit.py",  # this file
        }
        return [f for f in files if f not in exclusions and ".git" not in str(f)]

    def test_no_gpu_workload_doctor(self) -> None:
        for fpath in self._collect_files():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for bad in ["GPU Workload Doctor", "GPU Doctor"]:
                self.assertNotIn(bad, content,
                                 f"Found forbidden string '{bad}' in {fpath}")

    def test_no_gpu_doctor_benchmark(self) -> None:
        for fpath in self._collect_files():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            self.assertNotIn("gpu_doctor_benchmark", content,
                             f"Found 'gpu_doctor_benchmark' in {fpath}")

    def test_no_benchmark_sqlite_in_code(self) -> None:
        """benchmark.sqlite should only appear in database.py migration warning, not elsewhere."""
        bad_files = []
        allowed = {
            ROOT / "storage" / "database.py",  # migration compatibility warning
            ROOT / "docs" / "CODE_COVERAGE_AUDIT.md",  # audit doc
            ROOT / "tests" / "test_code_coverage_audit.py",  # this file
        }
        for fpath in self._collect_files():
            if fpath in allowed:
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _FORBIDDEN_DB in content:
                bad_files.append(str(fpath))
        self.assertEqual(bad_files, [],
                         f"Found 'benchmark.sqlite' in unexpected files: {bad_files}")


if __name__ == "__main__":
    unittest.main()
