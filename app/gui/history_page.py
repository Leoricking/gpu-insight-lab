"""
GPU Insight Lab - History and Comparison Page
Shows session history from SQLite. Allows selecting two sessions to compare.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Slot  # type: ignore
    from PySide6.QtGui import QFont  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False

if _PYSIDE_AVAILABLE:
    class HistoryPage(QWidget):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._sessions: List[Dict[str, Any]] = []
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            title_row = QHBoxLayout()
            title = QLabel("History and Comparison")
            title.setFont(QFont("Segoe UI", 16, QFont.Bold))
            title_row.addWidget(title)
            title_row.addStretch()
            self._refresh_btn = QPushButton("Refresh")
            self._refresh_btn.clicked.connect(self._load_sessions)
            title_row.addWidget(self._refresh_btn)
            layout.addLayout(title_row)

            # Sessions table
            tbl_group = QGroupBox("Benchmark Sessions")
            tbl_layout = QVBoxLayout(tbl_group)
            self._sessions_table = QTableWidget(0, 7)
            self._sessions_table.setHorizontalHeaderLabels([
                "ID", "Name", "Timestamp", "GPU", "Driver", "Score", "Status"
            ])
            self._sessions_table.setSelectionMode(QAbstractItemView.MultiSelection)
            self._sessions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self._sessions_table.horizontalHeader().setStretchLastSection(True)
            self._sessions_table.setAlternatingRowColors(True)
            tbl_layout.addWidget(self._sessions_table)
            layout.addWidget(tbl_group, 1)

            # Buttons
            btn_row = QHBoxLayout()
            self._compare_btn = QPushButton("Compare Selected (select 2)")
            self._compare_btn.clicked.connect(self._compare_sessions)
            btn_row.addWidget(self._compare_btn)
            self._delete_btn = QPushButton("Delete Selected")
            self._delete_btn.clicked.connect(self._delete_session)
            btn_row.addWidget(self._delete_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            # Comparison output
            compare_group = QGroupBox("Comparison Result")
            compare_layout = QVBoxLayout(compare_group)
            self._compare_output = QTextEdit()
            self._compare_output.setReadOnly(True)
            self._compare_output.setMaximumHeight(200)
            compare_layout.addWidget(self._compare_output)
            layout.addWidget(compare_group)

            self._status_label = QLabel("Click Refresh to load sessions.")
            self._status_label.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(self._status_label)

        @Slot()
        def _load_sessions(self) -> None:
            try:
                from storage.database import get_database  # noqa: PLC0415
                db = get_database()
                self._sessions = db.get_sessions()
                self._populate_table()
                self._status_label.setText(f"Loaded {len(self._sessions)} session(s)")
            except Exception as exc:  # noqa: BLE001
                self._status_label.setText(f"Error loading sessions: {exc}")

        def _populate_table(self) -> None:
            self._sessions_table.setRowCount(0)
            for row_i, s in enumerate(self._sessions):
                self._sessions_table.insertRow(row_i)
                vals = [
                    str(s.get("id", "")),
                    s.get("session_name", ""),
                    str(s.get("started_at", ""))[:19],
                    s.get("gpu_name", "N/A"),
                    s.get("driver_version", "N/A"),
                    f"{s.get('health_score', 0):.1f}" if s.get("health_score") is not None else "N/A",
                    s.get("status", ""),
                ]
                for ci, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self._sessions_table.setItem(row_i, ci, item)
            self._sessions_table.resizeColumnsToContents()

        @Slot()
        def _compare_sessions(self) -> None:
            selected_rows = list(set(idx.row() for idx in self._sessions_table.selectedIndexes()))
            if len(selected_rows) != 2:
                QMessageBox.warning(self, "Select 2 Sessions", "Please select exactly two sessions to compare.")
                return
            a_id = int(self._sessions_table.item(selected_rows[0], 0).text())
            b_id = int(self._sessions_table.item(selected_rows[1], 0).text())
            try:
                from storage.database import get_database  # noqa: PLC0415
                db = get_database()
                result = db.compare_sessions(a_id, b_id)
                if result is None:
                    self._compare_output.setText("Comparison failed: sessions not found")
                    return
                lines = [
                    f"Comparison: Session {a_id} vs Session {b_id}",
                    f"  A: {result.get('session_a_name')} — GPU: {result.get('session_a_gpu')} — Score: {result.get('session_a_score')}",
                    f"  B: {result.get('session_b_name')} — GPU: {result.get('session_b_gpu')} — Score: {result.get('session_b_score')}",
                    "",
                    "Benchmark Comparison:",
                ]
                for bm in result.get("benchmarks", []):
                    a_ms = bm.get("a_mean_ms")
                    b_ms = bm.get("b_mean_ms")
                    delta = bm.get("delta_pct")
                    line = f"  {bm.get('test_name', '')}: A={a_ms}ms B={b_ms}ms"
                    if delta is not None:
                        line += f" delta={delta:+.1f}%"
                    lines.append(line)
                self._compare_output.setText("\n".join(lines))
            except Exception as exc:  # noqa: BLE001
                self._compare_output.setText(f"Comparison error: {exc}")

        @Slot()
        def _delete_session(self) -> None:
            selected_rows = list(set(idx.row() for idx in self._sessions_table.selectedIndexes()))
            if not selected_rows:
                return
            confirm = QMessageBox.question(
                self, "Delete Session",
                f"Delete {len(selected_rows)} session(s)? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            try:
                from storage.database import get_database  # noqa: PLC0415
                db = get_database()
                for row_i in selected_rows:
                    session_id = int(self._sessions_table.item(row_i, 0).text())
                    db.delete_session(session_id)
                self._load_sessions()
            except Exception as exc:  # noqa: BLE001
                self._status_label.setText(f"Delete error: {exc}")

else:
    class HistoryPage:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass
