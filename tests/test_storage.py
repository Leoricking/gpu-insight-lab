"""
Tests for SQLite storage: migration, save/load, compare, delete.
"""

import tempfile
import unittest
from pathlib import Path


def _make_mock_session_dict() -> dict:
    return {
        "session_id": "test-001",
        "session_name": "Storage Test Session",
        "started_at": 1720000000.0,
        "completed_at": 1720000060.0,
        "status": "completed",
        "health_score": 65.0,
        "score_confidence": 0.75,
        "system_info": {"hostname": "testhost", "os_name": "Windows"},
        "nvidia_info": {"gpu_name": "Test GPU", "driver_version": "545.0"},
        "cuda_info": {"nvcc_version": "12.3"},
        "pcie_info": {},
        "tool_status": {},
        "amd_info": {},
        "results": [
            {
                "test_name": "vector_add",
                "mean": 1.0,
                "bandwidth_gbps": 10.0,
                "correctness_pass": True,
                "schema_version": "1.0",
            }
        ],
        "diagnosis_results": [
            {
                "rule_id": "thermal_throttle",
                "severity": "INFO",
                "category": "HEALTHY",
                "title": "Temperature Normal",
                "summary": "GPU temp is fine",
                "evidence": "temperature=65C",
                "confidence": 0.9,
            }
        ],
    }


class TestMigrations(unittest.TestCase):
    def test_migration_runs_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from storage.database import Database
            db = Database(Path(tmpdir) / "test.sqlite")
            db.close()

    def test_schema_version_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            import sqlite3
            from storage.database import Database
            from storage.migrations import get_current_version
            db = Database(Path(tmpdir) / "test.sqlite")
            version = get_current_version(db._conn)
            self.assertGreaterEqual(version, 1)
            db.close()

    def test_idempotent_migrations(self) -> None:
        """Running migrations twice should not error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from storage.database import Database
            db_path = Path(tmpdir) / "test.sqlite"
            db = Database(db_path)
            db.close()
            db2 = Database(db_path)
            db2.close()


class TestSaveLoadSession(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        from storage.database import Database
        self.db = Database(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_save_returns_id(self) -> None:
        session = _make_mock_session_dict()
        session_id = self.db.save_session(session)
        self.assertIsNotNone(session_id)
        self.assertIsInstance(session_id, int)

    def test_save_and_load(self) -> None:
        session = _make_mock_session_dict()
        session_id = self.db.save_session(session)
        loaded = self.db.get_session(session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.get("session_name"), "Storage Test Session")

    def test_results_preserved(self) -> None:
        session = _make_mock_session_dict()
        session_id = self.db.save_session(session)
        loaded = self.db.get_session(session_id)
        self.assertIsNotNone(loaded)
        results = loaded.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("test_name"), "vector_add")

    def test_diagnosis_preserved(self) -> None:
        session = _make_mock_session_dict()
        session_id = self.db.save_session(session)
        loaded = self.db.get_session(session_id)
        diag = loaded.get("diagnosis_results", [])
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag[0].get("rule_id"), "thermal_throttle")

    def test_get_sessions_returns_list(self) -> None:
        for _ in range(3):
            self.db.save_session(_make_mock_session_dict())
        sessions = self.db.get_sessions()
        self.assertGreaterEqual(len(sessions), 3)


class TestDeleteSession(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        from storage.database import Database
        self.db = Database(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_delete_removes_session(self) -> None:
        session_id = self.db.save_session(_make_mock_session_dict())
        self.assertIsNotNone(self.db.get_session(session_id))
        result = self.db.delete_session(session_id)
        self.assertTrue(result)
        self.assertIsNone(self.db.get_session(session_id))

    def test_delete_nonexistent_returns_true(self) -> None:
        """Deleting a nonexistent session should return True (no rows affected)."""
        result = self.db.delete_session(99999)
        self.assertTrue(result)


class TestCompareSessions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        from storage.database import Database
        self.db = Database(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_compare_returns_dict(self) -> None:
        a = _make_mock_session_dict()
        a["session_name"] = "Session A"
        b = _make_mock_session_dict()
        b["session_name"] = "Session B"
        a_id = self.db.save_session(a)
        b_id = self.db.save_session(b)
        result = self.db.compare_sessions(a_id, b_id)
        self.assertIsNotNone(result)
        self.assertIn("benchmarks", result)
        self.assertEqual(result["session_a_id"], a_id)
        self.assertEqual(result["session_b_id"], b_id)

    def test_compare_nonexistent_returns_none(self) -> None:
        result = self.db.compare_sessions(9998, 9999)
        self.assertIsNone(result)


class TestApplicationSettings(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        from storage.database import Database
        self.db = Database(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_get_missing_key_returns_default(self) -> None:
        val = self.db.get_setting("nonexistent_key", "default_val")
        self.assertEqual(val, "default_val")

    def test_set_and_get(self) -> None:
        self.db.set_setting("output_dir", "/tmp/test")
        val = self.db.get_setting("output_dir")
        self.assertEqual(val, "/tmp/test")

    def test_upsert_overwrites(self) -> None:
        self.db.set_setting("theme", "light")
        self.db.set_setting("theme", "dark")
        val = self.db.get_setting("theme")
        self.assertEqual(val, "dark")


if __name__ == "__main__":
    unittest.main()
