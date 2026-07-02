"""
GPU Insight Lab - CSV Report Generator
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from app.branding import APP_NAME, APP_VERSION, REPORT_PREFIX

logger = logging.getLogger(__name__)

_COLUMNS = [
    "test_name",
    "data_type",
    "input_size",
    "block_size",
    "warmup_runs",
    "measured_runs",
    "cpu_time_ms",
    "gpu_time_ms",
    "transfer_time_ms",
    "end_to_end_time_ms",
    "throughput",
    "bandwidth_gbps",
    "speedup",
    "correctness_pass",
    "max_error",
    "mean",
    "median",
    "min_val",
    "max_val",
    "standard_deviation",
    "gpu_name",
    "driver_version",
    "cuda_runtime_version",
    "notes",
    "error",
]


def generate(
    session: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate CSV report. Returns path to created file."""
    if output_dir is None:
        from app.config import get_config  # noqa: PLC0415
        output_dir = get_config().output_path()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{REPORT_PREFIX}_{ts}.csv"
    file_path = output_dir / filename

    # Get results list
    if hasattr(session, "results"):
        results = session.results
    elif isinstance(session, dict):
        results = session.get("results", [])
    else:
        results = []

    rows: List[dict] = []
    for r in results:
        if hasattr(r, "to_dict"):
            row = r.to_dict()
        elif isinstance(r, dict):
            row = r
        else:
            continue
        rows.append(row)

    try:
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in _COLUMNS})
        logger.info("CSV report written to %s", file_path)
    except OSError as exc:
        logger.error("Failed to write CSV report: %s", exc)
        raise

    return file_path
