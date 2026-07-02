"""
GPU Insight Lab - Benchmark Page (Kernel Lab)
All long operations run in QThread workers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Slot  # type: ignore
    from PySide6.QtGui import QFont  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QFileDialog,
    )
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False

if _PYSIDE_AVAILABLE:
    class BenchmarkPage(QWidget):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._worker: Any = None
            self._last_session: Optional[Dict[str, Any]] = None
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            title = QLabel("Kernel Lab")
            title.setFont(QFont("Segoe UI", 16, QFont.Bold))
            layout.addWidget(title)

            # Control row
            ctrl_group = QGroupBox("Test Controls")
            ctrl_layout = QHBoxLayout(ctrl_group)

            btn_data = [
                ("Quick Test", "quick"),
                ("Full Test", "full"),
                ("Device Info", "device_info"),
                ("Memory Test", "memory_bandwidth"),
                ("Vector Add", "vector_add"),
                ("Reduction", "reduction"),
                ("Transpose", "transpose"),
                ("GEMM", "gemm_tiled"),
                ("Streams", "stream_pipeline"),
                ("Image Test", "image_grayscale"),
            ]
            self._test_buttons: Dict[str, QPushButton] = {}
            for label, test_name in btn_data:
                btn = QPushButton(label)
                btn.clicked.connect(lambda checked=False, n=test_name: self._run_test(n))
                ctrl_layout.addWidget(btn)
                self._test_buttons[test_name] = btn
            layout.addWidget(ctrl_group)

            # Options row
            opt_group = QGroupBox("Options")
            opt_layout = QHBoxLayout(opt_group)

            opt_layout.addWidget(QLabel("Repeat:"))
            self._repeat_spin = QSpinBox()
            self._repeat_spin.setRange(1, 100)
            self._repeat_spin.setValue(10)
            opt_layout.addWidget(self._repeat_spin)

            opt_layout.addWidget(QLabel("Data Size:"))
            self._size_spin = QSpinBox()
            self._size_spin.setRange(0, 268_435_456)
            self._size_spin.setValue(0)
            self._size_spin.setSpecialValueText("Default")
            opt_layout.addWidget(self._size_spin)

            opt_layout.addWidget(QLabel("Block Size:"))
            self._block_spin = QSpinBox()
            self._block_spin.setRange(0, 1024)
            self._block_spin.setValue(0)
            self._block_spin.setSpecialValueText("Default")
            opt_layout.addWidget(self._block_spin)

            opt_layout.addWidget(QLabel("Output:"))
            self._output_btn = QPushButton("Browse...")
            self._output_btn.clicked.connect(self._browse_output)
            self._output_dir = ""
            opt_layout.addWidget(self._output_btn)
            opt_layout.addStretch()

            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.clicked.connect(self._cancel_test)
            opt_layout.addWidget(self._cancel_btn)
            layout.addWidget(opt_group)

            # Progress
            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            layout.addWidget(self._progress)

            self._status_label = QLabel("Ready")
            self._status_label.setStyleSheet("color: #444; font-style: italic;")
            layout.addWidget(self._status_label)

            # Results table
            results_group = QGroupBox("Results")
            results_layout = QVBoxLayout(results_group)
            self._results_table = QTableWidget(0, 8)
            self._results_table.setHorizontalHeaderLabels([
                "Test", "Mean (ms)", "Std Dev", "BW (GB/s)", "Speedup", "Correct", "GPU", "Notes"
            ])
            self._results_table.horizontalHeader().setStretchLastSection(True)
            self._results_table.setAlternatingRowColors(True)
            results_layout.addWidget(self._results_table)
            layout.addWidget(results_group, 1)

        @Slot()
        def _browse_output(self) -> None:
            d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if d:
                self._output_dir = d
                self._output_btn.setText(f".../{d.split('/')[-1]}")

        def _run_test(self, test_name: str) -> None:
            if self._worker and self._worker.isRunning():
                return
            self._set_running(True)
            self._results_table.setRowCount(0)

            from app.gui.workers import BenchmarkWorker  # noqa: PLC0415
            self._worker = BenchmarkWorker(
                test_name=test_name,
                repeat=self._repeat_spin.value(),
                data_size=self._size_spin.value(),
                block_size=self._block_spin.value(),
                output_dir=self._output_dir,
            )
            self._worker.progress.connect(self._progress.setValue)
            self._worker.status.connect(self._status_label.setText)
            self._worker.finished.connect(self._on_finished)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        @Slot()
        def _cancel_test(self) -> None:
            if self._worker:
                self._worker.cancel()
            self._set_running(False)
            self._status_label.setText("Cancelled")

        @Slot(dict)
        def _on_finished(self, data: Dict[str, Any]) -> None:
            self._set_running(False)
            self._last_session = data
            self._populate_results(data)

        @Slot(str)
        def _on_error(self, msg: str) -> None:
            self._set_running(False)
            self._status_label.setText(f"Error: {msg}")

        def _set_running(self, running: bool) -> None:
            for btn in self._test_buttons.values():
                btn.setEnabled(not running)
            self._cancel_btn.setEnabled(running)
            if not running:
                self._progress.setValue(0)

        def _populate_results(self, data: Dict[str, Any]) -> None:
            results = data.get("results", []) or []
            if not results and "single_result" in data:
                results = [data["single_result"]]

            self._results_table.setRowCount(len(results))
            for ri, r in enumerate(results):
                r_d = r if isinstance(r, dict) else {}
                vals = [
                    r_d.get("test_name", ""),
                    f"{r_d.get('mean', ''):.3f}" if isinstance(r_d.get('mean'), float) else "N/A",
                    f"{r_d.get('standard_deviation', ''):.3f}" if isinstance(r_d.get('standard_deviation'), float) else "N/A",
                    f"{r_d.get('bandwidth_gbps', ''):.2f}" if isinstance(r_d.get('bandwidth_gbps'), float) else "N/A",
                    f"{r_d.get('speedup', ''):.1f}x" if isinstance(r_d.get('speedup'), float) else "N/A",
                    "Pass" if r_d.get('correctness_pass') is True else ("Fail" if r_d.get('correctness_pass') is False else "N/A"),
                    r_d.get("gpu_name", ""),
                    r_d.get("notes", ""),
                ]
                for ci, v in enumerate(vals):
                    self._results_table.setItem(ri, ci, QTableWidgetItem(str(v)))
            self._results_table.resizeColumnsToContents()

        def get_last_session(self) -> Optional[Dict[str, Any]]:
            return self._last_session

else:
    class BenchmarkPage:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass
