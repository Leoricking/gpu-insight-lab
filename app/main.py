"""
GPU Insight Lab - GUI Entry Point
Launches the PySide6 desktop application.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication  # type: ignore
        from PySide6.QtCore import Qt  # type: ignore
    except ImportError:
        logger.error(
            "PySide6 is not installed. "
            "Install with: pip install PySide6>=6.6.0\n"
            "Or use the CLI: python -m app.cli"
        )
        return 1

    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("GPU Insight Lab")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("GPU Insight Lab")

    from app.gui.main_window import MainWindow  # noqa: PLC0415
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
