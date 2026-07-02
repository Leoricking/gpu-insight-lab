"""
GPU Insight Lab - Database Migrations
Versioned schema migration system. Initial schema = version 1.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

# Each migration is a function(conn) -> None
Migration = Callable[[sqlite3.Connection], None]


def _migration_v1(conn: sqlite3.Connection) -> None:
    """Initial schema creation."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS systems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hostname TEXT NOT NULL DEFAULT '',
            os TEXT NOT NULL DEFAULT '',
            cpu TEXT NOT NULL DEFAULT '',
            ram_gb REAL NOT NULL DEFAULT 0,
            gpu_name TEXT NOT NULL DEFAULT '',
            driver_version TEXT NOT NULL DEFAULT '',
            cuda_version TEXT NOT NULL DEFAULT '',
            collected_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS benchmark_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id INTEGER REFERENCES systems(id),
            session_name TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at TEXT,
            health_score REAL,
            confidence REAL,
            status TEXT NOT NULL DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER REFERENCES benchmark_sessions(id),
            test_name TEXT NOT NULL DEFAULT '',
            schema_version TEXT NOT NULL DEFAULT '1.0',
            result_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS diagnosis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER REFERENCES benchmark_sessions(id),
            rule_id TEXT NOT NULL DEFAULT '',
            severity TEXT NOT NULL DEFAULT 'INFO',
            category TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            evidence TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER REFERENCES benchmark_sessions(id),
            format TEXT NOT NULL DEFAULT '',
            file_path TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS application_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_started ON benchmark_sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_results_session ON benchmark_results(session_id);
        CREATE INDEX IF NOT EXISTS idx_diag_session ON diagnosis_results(session_id);
    """)


# Ordered list of migrations indexed by version number (1-based)
MIGRATIONS: List[Migration] = [
    _migration_v1,  # version 1
]


def get_current_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version from the DB (0 if uninitialized)."""
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            return 0
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error:
        return 0


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations in order."""
    current = get_current_version(conn)
    for idx, migration_fn in enumerate(MIGRATIONS):
        version = idx + 1
        if version <= current:
            continue
        logger.info("Applying database migration to version %d", version)
        try:
            migration_fn(conn)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (version,)
            )
            conn.commit()
            logger.info("Migration to version %d applied successfully", version)
        except sqlite3.Error as exc:
            logger.error("Migration to version %d failed: %s", version, exc)
            conn.rollback()
            raise
