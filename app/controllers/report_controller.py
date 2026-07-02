"""GPU Insight Lab - Report Controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def generate_report(
    session: Any,
    fmt: str = "json",
    output_dir: Optional[Path] = None,
) -> Path:
    fmt = fmt.lower()
    if fmt == "json":
        from reports.json_report import generate  # noqa: PLC0415
    elif fmt == "csv":
        from reports.csv_report import generate  # noqa: PLC0415
    elif fmt in ("markdown", "md"):
        from reports.markdown_report import generate  # noqa: PLC0415
    elif fmt == "html":
        from reports.html_report import generate  # noqa: PLC0415
    elif fmt in ("excel", "xlsx"):
        from reports.excel_report import generate  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown report format: {fmt}")
    return generate(session, output_dir=output_dir)
