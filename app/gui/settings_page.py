"""
GPU Insight Lab - Settings Page
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Slot  # type: ignore
    from PySide6.QtGui import QFont  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False

if _PYSIDE_AVAILABLE:
    class SettingsPage(QWidget):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._setup_ui()
            self._load_settings()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            title = QLabel("Settings")
            title.setFont(QFont("Segoe UI", 16, QFont.Bold))
            layout.addWidget(title)

            # Paths group
            paths_group = QGroupBox("Paths")
            paths_form = QFormLayout(paths_group)

            self._output_dir_edit = QLineEdit()
            self._output_dir_btn = QPushButton("Browse...")
            self._output_dir_btn.clicked.connect(lambda: self._browse_dir(self._output_dir_edit))
            out_row = QHBoxLayout()
            out_row.addWidget(self._output_dir_edit)
            out_row.addWidget(self._output_dir_btn)
            paths_form.addRow("Output Directory:", out_row)

            self._native_exe_edit = QLineEdit()
            self._native_exe_btn = QPushButton("Browse...")
            self._native_exe_btn.clicked.connect(lambda: self._browse_file(self._native_exe_edit, "Executables (*.exe);;All files (*)"))
            exe_row = QHBoxLayout()
            exe_row.addWidget(self._native_exe_edit)
            exe_row.addWidget(self._native_exe_btn)
            paths_form.addRow("Native Benchmark Exe:", exe_row)

            self._nsys_edit = QLineEdit()
            self._nsys_btn = QPushButton("Browse...")
            self._nsys_btn.clicked.connect(lambda: self._browse_file(self._nsys_edit))
            nsys_row = QHBoxLayout()
            nsys_row.addWidget(self._nsys_edit)
            nsys_row.addWidget(self._nsys_btn)
            paths_form.addRow("Nsight Systems (nsys):", nsys_row)

            self._ncu_edit = QLineEdit()
            self._ncu_btn = QPushButton("Browse...")
            self._ncu_btn.clicked.connect(lambda: self._browse_file(self._ncu_edit))
            ncu_row = QHBoxLayout()
            ncu_row.addWidget(self._ncu_edit)
            ncu_row.addWidget(self._ncu_btn)
            paths_form.addRow("Nsight Compute (ncu):", ncu_row)
            layout.addWidget(paths_group)

            # Benchmark options
            bm_group = QGroupBox("Benchmark Defaults")
            bm_form = QFormLayout(bm_group)

            self._timeout_spin = QSpinBox()
            self._timeout_spin.setRange(10, 3600)
            self._timeout_spin.setValue(120)
            self._timeout_spin.setSuffix(" seconds")
            bm_form.addRow("Timeout:", self._timeout_spin)

            self._repeat_spin = QSpinBox()
            self._repeat_spin.setRange(1, 100)
            self._repeat_spin.setValue(10)
            bm_form.addRow("Default Repeat:", self._repeat_spin)
            layout.addWidget(bm_group)

            # Appearance
            appearance_group = QGroupBox("Appearance")
            appearance_form = QFormLayout(appearance_group)
            self._theme_combo = QComboBox()
            self._theme_combo.addItems(["light", "dark"])
            appearance_form.addRow("Theme (placeholder):", self._theme_combo)
            layout.addWidget(appearance_group)

            # Privacy
            privacy_group = QGroupBox("Privacy")
            privacy_form = QFormLayout(privacy_group)
            self._telemetry_check = QCheckBox("Enable telemetry (disabled by default)")
            self._telemetry_check.setChecked(False)
            privacy_form.addRow("", self._telemetry_check)
            layout.addWidget(privacy_group)

            # Buttons
            btn_row = QHBoxLayout()
            save_btn = QPushButton("Save Settings")
            save_btn.setStyleSheet(
                "QPushButton { background: #1a237e; color: white; padding: 6px 20px; "
                "border-radius: 4px; font-weight: bold; } "
                "QPushButton:hover { background: #283593; }"
            )
            save_btn.clicked.connect(self._save_settings)
            reset_btn = QPushButton("Reset to Defaults")
            reset_btn.clicked.connect(self._reset_defaults)
            btn_row.addWidget(save_btn)
            btn_row.addWidget(reset_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            self._status_label = QLabel("")
            self._status_label.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(self._status_label)
            layout.addStretch()

        def _browse_dir(self, target: QLineEdit) -> None:
            d = QFileDialog.getExistingDirectory(self, "Select Directory")
            if d:
                target.setText(d)

        def _browse_file(self, target: QLineEdit, filter_str: str = "All files (*)") -> None:
            f, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
            if f:
                target.setText(f)

        def _load_settings(self) -> None:
            from app.config import get_config  # noqa: PLC0415
            cfg = get_config()
            self._output_dir_edit.setText(cfg.output_dir)
            self._native_exe_edit.setText(cfg.native_executable_path)
            self._nsys_edit.setText(cfg.nsys_path)
            self._ncu_edit.setText(cfg.ncu_path)
            self._timeout_spin.setValue(cfg.timeout_seconds)
            self._repeat_spin.setValue(cfg.default_repeat)
            self._theme_combo.setCurrentText(cfg.theme)
            self._telemetry_check.setChecked(cfg.telemetry_enabled)

        @Slot()
        def _save_settings(self) -> None:
            from app.config import get_config, save_config  # noqa: PLC0415
            cfg = get_config()
            cfg.output_dir = self._output_dir_edit.text()
            cfg.native_executable_path = self._native_exe_edit.text()
            cfg.nsys_path = self._nsys_edit.text()
            cfg.ncu_path = self._ncu_edit.text()
            cfg.timeout_seconds = self._timeout_spin.value()
            cfg.default_repeat = self._repeat_spin.value()
            cfg.theme = self._theme_combo.currentText()
            cfg.telemetry_enabled = self._telemetry_check.isChecked()
            if save_config(cfg):
                self._status_label.setText("Settings saved")
            else:
                self._status_label.setText("Error saving settings")

        @Slot()
        def _reset_defaults(self) -> None:
            from app.config import AppConfig  # noqa: PLC0415
            defaults = AppConfig()
            self._output_dir_edit.setText(defaults.output_dir)
            self._native_exe_edit.setText(defaults.native_executable_path)
            self._nsys_edit.setText(defaults.nsys_path)
            self._ncu_edit.setText(defaults.ncu_path)
            self._timeout_spin.setValue(defaults.timeout_seconds)
            self._repeat_spin.setValue(defaults.default_repeat)
            self._theme_combo.setCurrentText(defaults.theme)
            self._telemetry_check.setChecked(defaults.telemetry_enabled)
            self._status_label.setText("Reset to defaults — click Save to apply")

else:
    class SettingsPage:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass
