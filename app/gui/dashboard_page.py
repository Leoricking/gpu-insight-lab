"""
GPU Insight Lab - Dashboard Page
Shows GPU Insight Score, GPU info, PCIe status, toolchain status, top issues.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Slot  # type: ignore
    from PySide6.QtGui import QFont  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False

if _PYSIDE_AVAILABLE:
    class ScoreWidget(QLabel):
        """Large circular-style score display."""
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.setAlignment(Qt.AlignCenter)
            self._update_display(None, None)
            self.setMinimumSize(120, 120)
            self.setMaximumSize(160, 160)
            self.setStyleSheet(
                "background: #1a237e; color: white; border-radius: 10px; "
                "font-size: 28px; font-weight: bold;"
            )

        def set_score(self, score: Optional[float], confidence: Optional[float]) -> None:
            self._update_display(score, confidence)

        def _update_display(self, score: Optional[float], confidence: Optional[float]) -> None:
            if score is not None:
                self.setText(f"{score:.1f}\n/ 100")
            else:
                self.setText("—\n/ 100")

    class DashboardPage(QWidget):
        """Main dashboard page."""
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._worker: Any = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            # Title row
            title_row = QHBoxLayout()
            title = QLabel("GPU Insight Dashboard")
            title.setFont(QFont("Segoe UI", 16, QFont.Bold))
            title_row.addWidget(title)
            title_row.addStretch()
            self._refresh_btn = QPushButton("Refresh")
            self._refresh_btn.clicked.connect(self._on_refresh)
            title_row.addWidget(self._refresh_btn)
            layout.addLayout(title_row)

            # Top row: score + GPU info
            top_row = QHBoxLayout()

            # Score card
            score_group = QGroupBox("GPU Insight Score")
            score_layout = QVBoxLayout(score_group)
            self._score_widget = ScoreWidget()
            score_layout.addWidget(self._score_widget, alignment=Qt.AlignCenter)
            self._confidence_label = QLabel("Confidence: —")
            self._confidence_label.setAlignment(Qt.AlignCenter)
            score_layout.addWidget(self._confidence_label)
            score_group.setMaximumWidth(200)
            top_row.addWidget(score_group)

            # GPU info card
            gpu_group = QGroupBox("GPU Status")
            gpu_layout = QGridLayout(gpu_group)
            self._gpu_labels: Dict[str, QLabel] = {}
            gpu_fields = [
                ("GPU Model", "gpu_name"),
                ("Driver Version", "driver_version"),
                ("CUDA Version", "cuda_driver_version"),
                ("VRAM Total", "vram_total_mb"),
                ("Temperature", "temperature_c"),
                ("GPU Utilization", "gpu_utilization_pct"),
                ("Performance State", "performance_state"),
                ("PCIe Link", "pcie_link"),
            ]
            for row_i, (label_text, key) in enumerate(gpu_fields):
                lbl = QLabel(f"{label_text}:")
                lbl.setStyleSheet("color: #666; font-weight: bold;")
                val = QLabel("—")
                gpu_layout.addWidget(lbl, row_i, 0)
                gpu_layout.addWidget(val, row_i, 1)
                self._gpu_labels[key] = val
            top_row.addWidget(gpu_group)
            layout.addLayout(top_row)

            # Toolchain status table
            tc_group = QGroupBox("Toolchain Status")
            tc_layout = QVBoxLayout(tc_group)
            self._tc_table = QTableWidget(0, 3)
            self._tc_table.setHorizontalHeaderLabels(["Tool", "Available", "Version"])
            self._tc_table.horizontalHeader().setStretchLastSection(True)
            self._tc_table.setMaximumHeight(180)
            tc_layout.addWidget(self._tc_table)
            layout.addWidget(tc_group)

            # Issues
            issues_group = QGroupBox("Top Issues & Recommendations")
            issues_layout = QVBoxLayout(issues_group)
            self._issues_label = QLabel("Run a benchmark to see diagnosis results.")
            self._issues_label.setWordWrap(True)
            issues_layout.addWidget(self._issues_label)
            layout.addWidget(issues_group)

            self._status_label = QLabel("Ready. Click Refresh to collect system info.")
            self._status_label.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(self._status_label)

        @Slot()
        def _on_refresh(self) -> None:
            from app.gui.workers import SystemInfoWorker  # noqa: PLC0415
            self._refresh_btn.setEnabled(False)
            self._status_label.setText("Collecting system info...")
            self._worker = SystemInfoWorker()
            self._worker.progress.connect(self._on_progress)
            self._worker.status.connect(self._status_label.setText)
            self._worker.finished.connect(self._on_data)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        @Slot(int)
        def _on_progress(self, value: int) -> None:
            self._status_label.setText(f"Collecting... {value}%")

        @Slot(dict)
        def _on_data(self, data: Dict[str, Any]) -> None:
            self._refresh_btn.setEnabled(True)
            self._status_label.setText("System info updated")
            self._update_display(data)

        @Slot(str)
        def _on_error(self, msg: str) -> None:
            self._refresh_btn.setEnabled(True)
            self._status_label.setText(f"Error: {msg}")

        def _update_display(self, data: Dict[str, Any]) -> None:
            nv = data.get("nvidia", {}) or {}
            pcie = data.get("pcie", {}) or {}
            tools = data.get("tools", {}) or {}

            def _v(d: dict, key: str, unit: str = "") -> str:
                val = d.get(key)
                if val is None:
                    return "N/A"
                if unit:
                    return f"{val}{unit}"
                return str(val)

            if nv.get("available"):
                self._gpu_labels["gpu_name"].setText(_v(nv, "gpu_name"))
                self._gpu_labels["driver_version"].setText(_v(nv, "driver_version"))
                self._gpu_labels["cuda_driver_version"].setText(_v(nv, "cuda_driver_version"))
                vram = nv.get("vram_total_mb")
                self._gpu_labels["vram_total_mb"].setText(
                    f"{vram/1024:.1f} GB" if vram else "N/A"
                )
                temp = nv.get("temperature_c")
                self._gpu_labels["temperature_c"].setText(
                    f"{temp:.0f} °C" if temp else "N/A"
                )
                util = nv.get("gpu_utilization_pct")
                self._gpu_labels["gpu_utilization_pct"].setText(
                    f"{util:.0f}%" if util is not None else "N/A"
                )
                self._gpu_labels["performance_state"].setText(_v(nv, "performance_state"))
            else:
                for lbl in self._gpu_labels.values():
                    lbl.setText("No NVIDIA GPU")

            if pcie.get("available"):
                gen = pcie.get("pcie_gen_current", "?")
                width = pcie.get("pcie_width_current", "?")
                self._gpu_labels["pcie_link"].setText(f"Gen{gen} x{width}")
            else:
                self._gpu_labels["pcie_link"].setText("N/A")

            # Toolchain table
            self._tc_table.setRowCount(0)
            for i, (tool_name, status) in enumerate(tools.items()):
                if not isinstance(status, dict):
                    continue
                self._tc_table.insertRow(i)
                self._tc_table.setItem(i, 0, QTableWidgetItem(tool_name))
                avail_item = QTableWidgetItem("Yes" if status.get("exists") else "No")
                if status.get("exists"):
                    avail_item.setForeground(Qt.darkGreen)
                else:
                    avail_item.setForeground(Qt.darkGray)
                self._tc_table.setItem(i, 1, avail_item)
                self._tc_table.setItem(i, 2, QTableWidgetItem(status.get("version") or "N/A"))
            self._tc_table.resizeColumnsToContents()

        def update_session(self, session_data: Dict[str, Any]) -> None:
            """Update from a completed benchmark session."""
            score = session_data.get("health_score")
            confidence = session_data.get("score_confidence")
            self._score_widget.set_score(score, confidence)
            if confidence is not None:
                self._confidence_label.setText(f"Confidence: {confidence*100:.0f}%")

            # Top issues from diagnosis
            diag = session_data.get("diagnosis_results", []) or []
            warnings = [d for d in diag if d.get("severity") in ("WARNING", "ERROR", "CRITICAL")]
            if warnings:
                issue_text = "\n".join(
                    f"[{d.get('severity')}] {d.get('title', '')}" for d in warnings[:5]
                )
            else:
                issue_text = "No warnings or errors detected."
            self._issues_label.setText(issue_text)

            # Update GPU info from session
            nv = session_data.get("nvidia_info", {}) or {}
            pcie = session_data.get("pcie_info", {}) or {}
            tools = session_data.get("tool_status", {}) or {}
            self._update_display({"nvidia": nv, "pcie": pcie, "tools": tools})

else:
    class DashboardPage:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass
