"""
GPU Insight Lab - Main Window
PySide6 QMainWindow with sidebar navigation and page stack.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, QSize  # type: ignore
    from PySide6.QtGui import QFont, QIcon  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QApplication,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False
    logger.warning("PySide6 not installed; GUI will not be available")


if _PYSIDE_AVAILABLE:
    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._setup_window()
            self._setup_ui()
            self.statusBar().showMessage("Ready — GPU Insight Lab v0.1.0")

        def _setup_window(self) -> None:
            from app.branding import APP_NAME, APP_VERSION  # noqa: PLC0415
            self.setWindowTitle(f"{APP_NAME}")
            self.setMinimumSize(1280, 720)
            self.resize(1440, 900)

        def _setup_ui(self) -> None:
            # Central widget with horizontal splitter
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)

            # Sidebar
            sidebar = QWidget()
            sidebar.setFixedWidth(220)
            sidebar.setStyleSheet(
                "background: #1a237e; color: white;"
            )
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(0, 0, 0, 0)
            sidebar_layout.setSpacing(0)

            # App title in sidebar
            title_widget = QWidget()
            title_widget.setStyleSheet("background: #0d1358; padding: 12px;")
            title_layout = QVBoxLayout(title_widget)
            title_layout.setContentsMargins(12, 12, 12, 12)
            app_name = QLabel("GPU Insight Lab")
            app_name.setFont(QFont("Segoe UI", 12, QFont.Bold))
            app_name.setStyleSheet("color: white;")
            app_name.setWordWrap(True)
            title_layout.addWidget(app_name)
            sub = QLabel("v0.1.0")
            sub.setStyleSheet("color: #90caf9; font-size: 11px;")
            title_layout.addWidget(sub)
            sidebar_layout.addWidget(title_widget)

            # Navigation list
            self._nav_list = QListWidget()
            self._nav_list.setStyleSheet("""
                QListWidget {
                    background: transparent;
                    border: none;
                    color: white;
                    font-size: 13px;
                }
                QListWidget::item {
                    padding: 10px 16px;
                    border-bottom: 1px solid rgba(255,255,255,0.08);
                }
                QListWidget::item:selected {
                    background: rgba(255,255,255,0.15);
                    border-left: 3px solid #00bcd4;
                }
                QListWidget::item:hover {
                    background: rgba(255,255,255,0.08);
                }
            """)
            self._nav_list.currentRowChanged.connect(self._on_nav_changed)

            _NAV_ITEMS = [
                "Dashboard",
                "System Inspector",
                "PCIe Analyzer",
                "Memory Benchmark",
                "Kernel Lab",
                "Workload Profiler",
                "Diagnosis Engine",
                "Validation Center",
                "History & Comparison",
                "Report Studio",
                "Settings",
            ]
            for item_text in _NAV_ITEMS:
                self._nav_list.addItem(QListWidgetItem(item_text))
            sidebar_layout.addWidget(self._nav_list, 1)
            main_layout.addWidget(sidebar)

            # Page stack
            self._page_stack = QStackedWidget()
            self._page_stack.setStyleSheet("background: #f5f5f5;")
            main_layout.addWidget(self._page_stack, 1)

            # Create pages
            self._create_pages()
            self._nav_list.setCurrentRow(0)

        def _create_pages(self) -> None:
            from app.gui.dashboard_page import DashboardPage  # noqa: PLC0415
            from app.gui.benchmark_page import BenchmarkPage  # noqa: PLC0415
            from app.gui.history_page import HistoryPage  # noqa: PLC0415
            from app.gui.report_page import ReportPage  # noqa: PLC0415
            from app.gui.settings_page import SettingsPage  # noqa: PLC0415

            self._dashboard_page = DashboardPage()
            self._benchmark_page = BenchmarkPage()
            self._history_page = HistoryPage()
            self._report_page = ReportPage()
            self._settings_page = SettingsPage()

            # Placeholder pages for unimplemented sections
            def _placeholder(title: str) -> QWidget:
                w = QWidget()
                lyt = QVBoxLayout(w)
                lbl = QLabel(title)
                lbl.setFont(QFont("Segoe UI", 18, QFont.Bold))
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("color: #666;")
                lyt.addStretch()
                lyt.addWidget(lbl)
                note = QLabel("Coming in next release.")
                note.setAlignment(Qt.AlignCenter)
                note.setStyleSheet("color: #999;")
                lyt.addWidget(note)
                lyt.addStretch()
                return w

            # Map: nav index → page widget
            # 0=Dashboard, 1=System Inspector, 2=PCIe Analyzer, 3=Memory Benchmark,
            # 4=Kernel Lab, 5=Workload Profiler, 6=Diagnosis Engine, 7=Validation Center,
            # 8=History&Comparison, 9=Report Studio, 10=Settings
            self._pages = [
                self._dashboard_page,                   # 0
                _placeholder("System Inspector"),       # 1
                _placeholder("PCIe Analyzer"),          # 2
                _placeholder("Memory Benchmark"),       # 3
                self._benchmark_page,                   # 4 (Kernel Lab)
                _placeholder("Workload Profiler"),      # 5
                _placeholder("Diagnosis Engine"),       # 6
                _placeholder("Validation Center"),      # 7
                self._history_page,                     # 8
                self._report_page,                      # 9
                self._settings_page,                    # 10
            ]
            for page in self._pages:
                self._page_stack.addWidget(page)

            # Connect benchmark page result to dashboard and report page
            if hasattr(self._benchmark_page, "get_last_session"):
                pass  # Wired via finished signal in BenchmarkPage

        def _on_nav_changed(self, index: int) -> None:
            if 0 <= index < len(self._pages):
                self._page_stack.setCurrentIndex(index)
                self.statusBar().showMessage(f"Page: {self._nav_list.item(index).text()}")

        def closeEvent(self, event: Any) -> None:
            # Close DB connection cleanly
            try:
                from storage.database import reset_database  # noqa: PLC0415
                reset_database()
            except Exception:  # noqa: BLE001
                pass
            super().closeEvent(event)

else:
    class MainWindow:  # type: ignore
        def __init__(self) -> None:
            raise ImportError(
                "PySide6 is not installed. Install with: pip install PySide6"
            )
