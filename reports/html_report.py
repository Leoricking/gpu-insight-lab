"""
GPU Insight Lab - HTML Report Generator
Uses Jinja2 template to produce self-contained HTML reports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.branding import APP_NAME, APP_VERSION, REPORT_PREFIX

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILE = "report.html.j2"


def _get_score_details(data: Dict[str, Any]) -> tuple:
    """Extract score, confidence, positive findings, missing data."""
    score = data.get("health_score")
    confidence = data.get("score_confidence")
    score_details = data.get("score_details", {}) or {}
    positive = score_details.get("positive_findings", []) or []
    missing = score_details.get("missing_data", []) or []
    return score, confidence, positive, missing


def _extract_recommendations(diag_results: List[Dict]) -> List[str]:
    """Collect non-empty recommendations from diagnosis results."""
    recs: List[str] = []
    for d in diag_results:
        rec = d.get("recommendation", "")
        if rec and d.get("severity") in ("WARNING", "ERROR", "CRITICAL"):
            recs.append(rec)
    return recs[:5]


def _result_to_dict(r: Any) -> Dict[str, Any]:
    if isinstance(r, dict):
        return r
    if hasattr(r, "to_dict"):
        return r.to_dict()
    return {}


def generate(
    session: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate HTML report using Jinja2 template. Returns path to created file."""
    try:
        from jinja2 import Environment, FileSystemLoader  # type: ignore
    except ImportError as exc:
        logger.error("Jinja2 not installed; cannot generate HTML report: %s", exc)
        raise

    if output_dir is None:
        from app.config import get_config  # noqa: PLC0415
        output_dir = get_config().output_path()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{REPORT_PREFIX}_{ts}.html"
    file_path = output_dir / filename

    # Convert session
    if hasattr(session, "to_dict"):
        data = session.to_dict()
    elif isinstance(session, dict):
        data = session
    else:
        data = {}

    sys_info = data.get("system_info", {}) or {}
    nv_info = data.get("nvidia_info", {}) or {}
    pcie_info = data.get("pcie_info", {}) or {}
    tools = data.get("tool_status", {}) or {}
    results = [_result_to_dict(r) for r in (data.get("results", []) or [])]
    diag_results = data.get("diagnosis_results", []) or []

    score, confidence, positive, missing = _get_score_details(data)
    recommendations = _extract_recommendations(diag_results)

    # Convert tools dict values to simple dicts
    tools_clean: Dict[str, Any] = {}
    for k, v in tools.items():
        if isinstance(v, dict):
            tools_clean[k] = v
        else:
            tools_clean[k] = {"exists": False, "version": None, "path": None}

    # Convert result dicts: ensure None for missing fields
    results_clean = []
    for r in results:
        rc: Dict[str, Any] = {
            "test_name": r.get("test_name", ""),
            "mean": r.get("mean"),
            "median": r.get("median"),
            "standard_deviation": r.get("standard_deviation"),
            "bandwidth_gbps": r.get("bandwidth_gbps"),
            "speedup": r.get("speedup"),
            "correctness_pass": r.get("correctness_pass"),
            "notes": r.get("notes", ""),
            "error": r.get("error"),
        }
        results_clean.append(rc)

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    context: Dict[str, Any] = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "generated_at": generated_at,
        "session_id": data.get("session_id", ""),
        "session_name": data.get("session_name", ""),
        "started_at": data.get("started_at", ""),
        "completed_at": data.get("completed_at", ""),
        "status": data.get("status", ""),
        "sys_info": sys_info,
        "nv_info": nv_info,
        "pcie_info": pcie_info,
        "tools": tools_clean,
        "results": results_clean,
        "diag_results": diag_results,
        "score": score,
        "confidence": confidence,
        "positive_findings": positive,
        "missing_data": missing,
        "recommendations": recommendations,
    }

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template(_TEMPLATE_FILE)
    html_content = template.render(**context)

    try:
        file_path.write_text(html_content, encoding="utf-8")
        logger.info("HTML report written to %s", file_path)
    except OSError as exc:
        logger.error("Failed to write HTML report: %s", exc)
        raise

    return file_path
