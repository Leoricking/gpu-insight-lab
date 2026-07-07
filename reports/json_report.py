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

    # Environment summary
    cuda_info = data.get("cuda_info", {}) or {}
    nv_info = data.get("nvidia_info", {}) or {}
    amd_info = data.get("amd_info", {}) or {}
    tool_status = data.get("tool_status", {}) or {}
    score_details = data.get("score_details", {}) or {}

    nsight_status = "AVAILABLE" if (
        tool_status.get("nsys", {}).get("exists") or tool_status.get("ncu", {}).get("exists")
    ) else "NOT_FOUND"
    hip_status = "NOT_VALIDATED" if not amd_info.get("rocm_available") else "AVAILABLE"

    environment_summary = {
        "cuda_available": bool(cuda_info.get("cuda_runtime_available") or cuda_info.get("nvcc_available")),
        "cuda_version": cuda_info.get("cuda_runtime_version") or cuda_info.get("nvcc_version") or "N/A",
        "gpu_name": nv_info.get("gpu_name") or "Not detected",
        "nsight_status": nsight_status,
        "amd_hip_status": hip_status,
        "native_benchmark_available": bool(cuda_info.get("native_benchmark_available")),
    }

    # Interview demo summary
    interview_demo_summary = {
        "title": f"{APP_NAME} — Interview Demo Summary",
        "implemented_features": [
            "System Inspector (CPU/GPU/PCIe/CUDA telemetry)",
            "Memory Benchmark (H2D/D2H/D2D via native CUDA binary)",
            "Kernel Lab (vector_add, reduction, transpose, gemm_naive, gemm_tiled, stream_pipeline)",
            "Evidence-based Diagnosis Engine (9 rules, evidence strings required)",
            "GPU Insight Score (0-100 composite across 6 categories)",
            "Multi-format Reports (JSON, CSV, Markdown, HTML, Excel)",
            "SQLite Session History with comparison",
            "PySide6 GUI (QMainWindow + QThread workers)",
            "CLI Automation (argparse, 10 commands, --json flag)",
            "CUDA to HIP Portability Demo (vector_add_hip.cpp, NOT_VALIDATED)",
        ],
        "roadmap_features": [
            "ROADMAP: softmax, layer_norm, GELU (AI inference kernels)",
            "ROADMAP: Flash Attention kernel",
            "ROADMAP: INT8 quantization",
            "ROADMAP: PyTorch extension integration",
            "ROADMAP: TensorRT plugin",
            "ROADMAP: cuFFT / cuBLAS full benchmark suite",
            "ROADMAP: Streamlit dashboard",
            "ROADMAP: Parquet storage backend",
            "ROADMAP: Multi-machine import",
            "ROADMAP: Company report templates",
            "ROADMAP: Batch execution / session manifests",
            "ROADMAP: Pass/fail policy YAML",
            "NOT_VALIDATED: AMD HIP real GPU benchmarks (requires ROCm hardware)",
        ],
        "note": "Features marked ROADMAP are planned but not yet implemented. NOT_VALIDATED requires AMD GPU hardware.",
    }

    # Add report metadata
    report = {
        "report_metadata": {
            "generated_by": f"{APP_NAME} v{APP_VERSION}",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "format": "json",
            "schema_version": "1.0",
        },
        "session_id": data.get("session_id", ""),
        "timestamp": data.get("started_at", ""),
        "environment": environment_summary,
        "benchmarks": data.get("results", []),
        "diagnosis": data.get("diagnosis_results", []),
        "score": data.get("health_score"),
        "missing_data": score_details.get("missing_data", []),
        "recommendations": [
            d.get("recommendation", "")
            for d in (data.get("diagnosis_results", []) or [])
            if d.get("recommendation") and d.get("severity") in ("WARNING", "ERROR", "CRITICAL")
        ],
        "interview_demo_summary": interview_demo_summary,
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
