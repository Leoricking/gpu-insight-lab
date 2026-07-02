"""
GPU Insight Lab - JSON Report Generator
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.branding import APP_NAME, APP_VERSION, REPORT_PREFIX

logger = logging.getLogger(__name__)


def generate(
    session: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate a JSON report for the given session.
    Returns the path to the generated file.
    """
    if output_dir is None:
        from app.config import get_config  # noqa: PLC0415
        output_dir = get_config().output_path()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"gpu_insight_session_{ts}.json"
    file_path = output_dir / filename

    # Convert session to dict
    if hasattr(session, "to_dict"):
        data = session.to_dict()
    elif isinstance(session, dict):
        data = session
    else:
        data = {"error": "Unknown session type", "session": str(session)}

    # Add report metadata
    report = {
        "report_metadata": {
            "generated_by": f"{APP_NAME} v{APP_VERSION}",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "format": "json",
            "schema_version": "1.0",
        },
        "session": data,
    }

    try:
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("JSON report written to %s", file_path)
    except OSError as exc:
        logger.error("Failed to write JSON report: %s", exc)
        raise

    return file_path
