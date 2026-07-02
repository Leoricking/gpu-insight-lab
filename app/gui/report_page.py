"""
GPU Insight Lab - Report Studio Page
Select session, choose format, generate report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Slot  # type: ignore
    from PySide6.QtGui import QFont  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False

if _PYSIDE_AVAILABLE:
    class ReportPage(QWidget):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._worker: Any = None
            self._current_session: Optional[Dict[str, Any]] = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            title = QLabel("Report Studio")
            title.setFont(QFont("Segoe UI", 16, QFont.Bold))
            layout.addWidget(title)

            # Session selection
            sess_group = QGroupBox("Session")
            sess_layout = QHBoxLayout(sess_group)
            sess_layout.addWidget(QLabel("Session:"))
            self._session_combo = QComboBox()
            self._session_combo.setMinimumWidth(300)
            sess_layout.addWidget(self._session_combo)
            self._reload_sessions_btn = QPushButton("Load Sessions")
            self._reload_sessions_btn.clicked.connect(self._load_sessions)
            sess_layout.addWidget(self._reload_sessions_btn)
            self._use_current_btn = QPushButton("Use Current Session")
            self._use_current_btn.clicked.connect(self._use_current)
            sess_layout.addWidget(self._use_current_btn)
            sess_layout.addStretch()
            layout.addWidget(sess_group)

            # Format and output
            fmt_group = QGroupBox("Format and Output")
            fmt_layout = QHBoxLayout(fmt_group)
            fmt_layout.addWidget(QLabel("Format:"))
            self._format_combo = QComboBox()
            self._format_combo.addItems(["json", "csv", "markdown", "html", "excel"])
            fmt_layout.addWidget(self._format_combo)
            fmt_layout.addWidget(QLabel("Output Dir:"))
            self._output_edit = QLineEdit()
            self._output_edit.setPlaceholderText("Default: ~/gpu_insight_lab_output")
            fmt_layout.addWidget(self._output_edit)
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(self._browse_dir)
            fmt_layout.addWidget(browse_btn)
            fmt_layout.addStretch()
            layout.addWidget(fmt_group)

            # Generate button
            gen_row = QHBoxLayout()
            self._generate_btn = QPushButton("Generate Report")
            self._generate_btn.setStyleSheet(
                "QPushButton { background: #1a237e; color: white; font-weight: bold; "
                "padding: 8px 24px; border-radius: 4px; } "
                "QPushButton:hover { background: #283593; }"
            )
            self._generate_btn.clicked.connect(self._generate)
            gen_row.addWidget(self._generate_btn)
            gen_row.addStretch()
            layout.addLayout(gen_row)

            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            layout.addWidget(self._progress)

            self._status_label = QLabel("Ready")
            self._status_label.setStyleSheet("color: #444; font-style: italic;")
            layout.addWidget(self._status_label)

            # Output log
            log_group = QGroupBox("Output")
            log_layout = QVBoxLayout(log_group)
            self._output_log = QTextEdit()
            self._output_log.setReadOnly(True)
            log_layout.addWidget(self._output_log)
            layout.addWidget(log_group, 1)

        @Slot()
        def _browse_dir(self) -> None:
            d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if d:
                self._output_edit.setText(d)

        @Slot()
        def _load_sessions(self) -> None:
            try:
                from storage.database import get_database  # noqa: PLC0415
                db = get_database()
                sessions = db.get_sessions()
                self._session_combo.clear()
                for s in sessions:
                    label = f"[{s.get('id')}] {s.get('session_name','')} | {s.get('started_at','')[:19]} | {s.get('gpu_name','N/A')}"
                    self._session_combo.addItem(label, userData=s.get("id"))
                self._status_label.setText(f"Loaded {len(sessions)} session(s)")
            except Exception as exc:  # noqa: BLE001
                self._status_label.setText(f"Error: {exc}")

        @Slot()
        def _use_current(self) -> None:
            if self._current_session:
                self._status_label.setText("Using current in-memory session")
            else:
                self._status_label.setText("No current session — run a benchmark first")

        @Slot()
        def _generate(self) -> None:
            if self._worker and self._worker.isRunning():
                return

            # Determine session data
            session_data: Optional[Dict[str, Any]] = None
            if self._current_session:
                session_data = self._current_session
            else:
                session_id = self._session_combo.currentData()
                if session_id is not None:
                    try:
                        from storage.database import get_database  # noqa: PLC0415
                        session_data = get_database().get_session(session_id)
                    except Exception as exc:  # noqa: BLE001
                        self._status_label.setText(f"Error loading session: {exc}")
                        return

            if not session_data:
                self._status_label.setText("No session selected — run a benchmark or select from history")
                return

            fmt = self._format_combo.currentText()
            output_dir = self._output_edit.text().strip()

            self._generate_btn.setEnabled(False)
            self._progress.setValue(0)

            from app.gui.workers import ReportWorker  # noqa: PLC0415
            self._worker = ReportWorker(session_data, fmt=fmt, output_dir=output_dir)
            self._worker.progress.connect(self._progress.setValue)
            self._worker.status.connect(self._status_label.setText)
            self._worker.finished.connect(self._on_done)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        @Slot(dict)
        def _on_done(self, data: Dict[str, Any]) -> None:
            self._generate_btn.setEnabled(True)
            file_path = data.get("file_path", "")
            self._output_log.append(f"Report generated: {file_path}")

        @Slot(str)
        def _on_error(self, msg: str) -> None:
            self._generate_btn.setEnabled(True)
            self._output_log.append(f"ERROR: {msg}")

        def set_current_session(self, data: Dict[str, Any]) -> None:
            self._current_session = data

else:
    class ReportPage:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass
