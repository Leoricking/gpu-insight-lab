"""
GPU Insight Lab - SQLite Database Manager
Manages persistence of benchmark sessions, results, and diagnosis data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from storage.migrations import run_migrations

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent / "gpu_insight.sqlite"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class Database:
    """SQLite-backed persistence layer for GPU Insight Lab."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or _DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        run_migrations(self._conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------
    # Systems
    # ------------------------------------------------------------------

    def _upsert_system(self, session_data: Dict[str, Any]) -> Optional[int]:
        """Insert or retrieve system record. Returns system_id."""
        sys_info = session_data.get("system_info", {}) or {}
        nv_info = session_data.get("nvidia_info", {}) or {}
        cuda_info = session_data.get("cuda_info", {}) or {}

        hostname = sys_info.get("hostname", "")
        os_str = f"{sys_info.get('os_name','')} {sys_info.get('os_version','')}".strip()
        cpu = sys_info.get("cpu_model", "")
        ram_gb = sys_info.get("ram_total_gb", 0.0) or 0.0
        gpu_name = nv_info.get("gpu_name", "")
        driver = nv_info.get("driver_version", "")
        cuda_ver = cuda_info.get("nvcc_version", "") or nv_info.get("cuda_driver_version", "")

        try:
            cursor = self._conn.execute(
                """INSERT INTO systems
                   (hostname, os, cpu, ram_gb, gpu_name, driver_version, cuda_version, collected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (hostname, os_str, cpu, ram_gb, gpu_name or "", driver or "", cuda_ver or "", _now_iso()),
            )
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as exc:
            logger.error("Failed to insert system record: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def save_session(self, session: Any) -> Optional[int]:
        """
        Persist a BenchmarkSession (or dict). Returns the DB session ID.
        """
        if hasattr(session, "to_dict"):
            data = session.to_dict()
        elif isinstance(session, dict):
            data = session
        else:
            logger.error("save_session: unsupported type %s", type(session))
            return None

        system_id = self._upsert_system(data)

        started_at = data.get("started_at")
        if isinstance(started_at, float):
            started_at = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
        else:
            started_at = started_at or _now_iso()

        completed_at = data.get("completed_at")
        if isinstance(completed_at, float):
            completed_at = datetime.fromtimestamp(completed_at, tz=timezone.utc).isoformat()

        try:
            cursor = self._conn.execute(
                """INSERT INTO benchmark_sessions
                   (system_id, session_name, started_at, completed_at, health_score, confidence, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    system_id,
                    data.get("session_name", ""),
                    started_at,
                    completed_at,
                    data.get("health_score"),
                    data.get("score_confidence"),
                    data.get("status", "completed"),
                ),
            )
            session_id = cursor.lastrowid
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to insert session: %s", exc)
            return None

        # Insert benchmark results
        for r in data.get("results", []):
            r_dict = r if isinstance(r, dict) else (r.to_dict() if hasattr(r, "to_dict") else {})
            test_name = r_dict.get("test_name", "")
            schema_ver = r_dict.get("schema_version", "1.0")
            try:
                self._conn.execute(
                    """INSERT INTO benchmark_results
                       (session_id, test_name, schema_version, result_json, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, test_name, schema_ver, json.dumps(r_dict), _now_iso()),
                )
            except sqlite3.Error as exc:
                logger.warning("Failed to insert benchmark result %s: %s", test_name, exc)

        # Insert diagnosis results
        for d in data.get("diagnosis_results", []):
            d_dict = d if isinstance(d, dict) else {}
            try:
                self._conn.execute(
                    """INSERT INTO diagnosis_results
                       (session_id, rule_id, severity, category, title, summary, evidence, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        d_dict.get("rule_id", ""),
                        d_dict.get("severity", "INFO"),
                        d_dict.get("category", ""),
                        d_dict.get("title", ""),
                        d_dict.get("summary", ""),
                        d_dict.get("evidence", ""),
                        d_dict.get("confidence", 0.0),
                        _now_iso(),
                    ),
                )
            except sqlite3.Error as exc:
                logger.warning("Failed to insert diagnosis result: %s", exc)

        self._conn.commit()
        return session_id

    def get_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return list of sessions with system info (joined)."""
        try:
            cursor = self._conn.execute(
                """SELECT s.id, s.session_name, s.started_at, s.completed_at,
                          s.health_score, s.confidence, s.status,
                          sys.hostname, sys.gpu_name, sys.driver_version, sys.cuda_version
                   FROM benchmark_sessions s
                   LEFT JOIN systems sys ON s.system_id = sys.id
                   ORDER BY s.started_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.error("get_sessions failed: %s", exc)
            return []

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Return full session data including results and diagnosis."""
        try:
            cursor = self._conn.execute(
                """SELECT s.*, sys.hostname, sys.os, sys.cpu, sys.ram_gb,
                          sys.gpu_name, sys.driver_version, sys.cuda_version
                   FROM benchmark_sessions s
                   LEFT JOIN systems sys ON s.system_id = sys.id
                   WHERE s.id = ?""",
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            data = dict(row)

            # Fetch benchmark results
            cursor = self._conn.execute(
                "SELECT test_name, schema_version, result_json FROM benchmark_results WHERE session_id = ?",
                (session_id,),
            )
            data["results"] = []
            for r in cursor.fetchall():
                try:
                    data["results"].append(json.loads(r["result_json"]))
                except json.JSONDecodeError:
                    data["results"].append({"test_name": r["test_name"], "error": "json decode error"})

            # Fetch diagnosis
            cursor = self._conn.execute(
                """SELECT rule_id, severity, category, title, summary, evidence, confidence
                   FROM diagnosis_results WHERE session_id = ?""",
                (session_id,),
            )
            data["diagnosis_results"] = [dict(r) for r in cursor.fetchall()]

            return data
        except sqlite3.Error as exc:
            logger.error("get_session(%d) failed: %s", session_id, exc)
            return None

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all related data."""
        try:
            self._conn.execute("DELETE FROM benchmark_results WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM diagnosis_results WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM reports WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM benchmark_sessions WHERE id = ?", (session_id,))
            self._conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("delete_session(%d) failed: %s", session_id, exc)
            return False

    def compare_sessions(
        self, session_a_id: int, session_b_id: int
    ) -> Optional[Dict[str, Any]]:
        """Return comparison dict for two sessions."""
        a = self.get_session(session_a_id)
        b = self.get_session(session_b_id)
        if a is None or b is None:
            return None

        def _map_tests(session: Dict) -> Dict[str, Dict]:
            m: Dict[str, Dict] = {}
            for r in session.get("results", []):
                name = r.get("test_name", "")
                if name:
                    m[name] = r
            return m

        a_tests = _map_tests(a)
        b_tests = _map_tests(b)
        all_tests = sorted(set(a_tests) | set(b_tests))

        comparison: Dict[str, Any] = {
            "session_a_id": session_a_id,
            "session_b_id": session_b_id,
            "session_a_name": a.get("session_name", ""),
            "session_b_name": b.get("session_name", ""),
            "session_a_gpu": a.get("gpu_name", ""),
            "session_b_gpu": b.get("gpu_name", ""),
            "session_a_score": a.get("health_score"),
            "session_b_score": b.get("health_score"),
            "benchmarks": [],
        }

        for test_name in all_tests:
            a_r = a_tests.get(test_name)
            b_r = b_tests.get(test_name)
            entry: Dict[str, Any] = {"test_name": test_name}

            if a_r:
                entry["a_mean_ms"] = a_r.get("mean")
                entry["a_bandwidth_gbps"] = a_r.get("bandwidth_gbps")
            if b_r:
                entry["b_mean_ms"] = b_r.get("mean")
                entry["b_bandwidth_gbps"] = b_r.get("bandwidth_gbps")

            # Delta
            a_m = a_r.get("mean") if a_r else None
            b_m = b_r.get("mean") if b_r else None
            if a_m and b_m and a_m > 0:
                entry["delta_pct"] = round((b_m - a_m) / a_m * 100, 1)

            comparison["benchmarks"].append(entry)

        return comparison

    def export_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Export full session as dict (for JSON export)."""
        return self.get_session(session_id)

    def save_report(self, session_id: int, fmt: str, file_path: str) -> None:
        """Record that a report was generated."""
        try:
            self._conn.execute(
                "INSERT INTO reports (session_id, format, file_path, created_at) VALUES (?, ?, ?, ?)",
                (session_id, fmt, file_path, _now_iso()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("save_report failed: %s", exc)

    def get_setting(self, key: str, default: str = "") -> str:
        """Retrieve an application setting."""
        try:
            cursor = self._conn.execute(
                "SELECT value FROM application_settings WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else default
        except sqlite3.Error:
            return default

    def set_setting(self, key: str, value: str) -> None:
        """Upsert an application setting."""
        try:
            self._conn.execute(
                """INSERT INTO application_settings (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, value, _now_iso()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("set_setting(%s) failed: %s", key, exc)


# Module-level singleton
_db: Optional[Database] = None


def get_database(db_path: Optional[Path] = None) -> Database:
    """Return the global Database instance."""
    global _db
    if _db is None:
        if db_path is None:
            from app.config import get_config  # noqa: PLC0415
            db_path = get_config().db_path()
        _db = Database(db_path)
    return _db


def reset_database() -> None:
    """Close and reset the global DB (for testing)."""
    global _db
    if _db is not None:
        _db.close()
        _db = None
