"""
GPU Insight Lab - Markdown Report Generator
Generates GitHub-compatible Markdown reports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.branding import APP_NAME, APP_VERSION, REPORT_PREFIX

logger = logging.getLogger(__name__)


def _safe(value: Any, fmt: str = "") -> str:
    if value is None:
        return "N/A"
    if fmt and isinstance(value, (int, float)):
        return format(value, fmt)
    return str(value)


def _session_data(session: Any) -> Dict[str, Any]:
    if hasattr(session, "to_dict"):
        return session.to_dict()
    if isinstance(session, dict):
        return session
    return {}


def generate(
    session: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate Markdown report. Returns path to created file."""
    if output_dir is None:
        from app.config import get_config  # noqa: PLC0415
        output_dir = get_config().output_path()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{REPORT_PREFIX}_{ts}.md"
    file_path = output_dir / filename

    data = _session_data(session)
    sys_info = data.get("system_info", {}) or {}
    nv_info = data.get("nvidia_info", {}) or {}
    cuda_info = data.get("cuda_info", {}) or {}
    pcie_info = data.get("pcie_info", {}) or {}
    tools = data.get("tool_status", {}) or {}
    results = data.get("results", []) or []
    diag_results = data.get("diagnosis_results", []) or []
    score = data.get("health_score")
    confidence = data.get("score_confidence")

    lines: List[str] = []
    lines.append("# GPU Insight Lab Session Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append(f"**Tool:** {APP_NAME} v{APP_VERSION}  ")
    lines.append("")

    # Score
    lines.append("## GPU Insight Score")
    lines.append("")
    if score is not None:
        lines.append(f"**Score: {score:.1f} / 100**  ")
        lines.append(f"Confidence: {(confidence or 0)*100:.0f}%  ")
    else:
        lines.append("Score: Not computed")
    lines.append("")

    # System summary
    lines.append("## System Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Hostname | {_safe(sys_info.get('hostname'))} |")
    lines.append(f"| OS | {_safe(sys_info.get('os_name'))} {_safe(sys_info.get('os_release'))} |")
    lines.append(f"| CPU | {_safe(sys_info.get('cpu_model'))} |")
    lines.append(f"| Logical Cores | {_safe(sys_info.get('cpu_logical_cores'))} |")
    lines.append(f"| RAM Total | {_safe(sys_info.get('ram_total_gb'), '.1f')} GB |")
    lines.append(f"| Python | {_safe(sys_info.get('python_version'))} |")
    lines.append("")

    # GPU info
    lines.append("## GPU Information")
    lines.append("")
    if nv_info.get("available"):
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| GPU | {_safe(nv_info.get('gpu_name'))} |")
        lines.append(f"| UUID | {_safe(nv_info.get('gpu_uuid'))} |")
        lines.append(f"| Driver Version | {_safe(nv_info.get('driver_version'))} |")
        lines.append(f"| CUDA Driver | {_safe(nv_info.get('cuda_driver_version'))} |")
        lines.append(f"| Compute Capability | {_safe(nv_info.get('compute_capability'))} |")
        lines.append(f"| VRAM Total | {_safe(nv_info.get('vram_total_mb'), '.0f')} MB |")
        lines.append(f"| VRAM Used | {_safe(nv_info.get('vram_used_mb'), '.0f')} MB |")
        lines.append(f"| Temperature | {_safe(nv_info.get('temperature_c'), '.1f')} °C |")
        lines.append(f"| GPU Utilization | {_safe(nv_info.get('gpu_utilization_pct'), '.0f')} % |")
        lines.append(f"| Power Draw | {_safe(nv_info.get('power_draw_w'), '.1f')} W |")
        lines.append(f"| Performance State | {_safe(nv_info.get('performance_state'))} |")
    else:
        lines.append("No NVIDIA GPU detected or data unavailable.")
    lines.append("")

    # PCIe
    if pcie_info.get("available"):
        lines.append("## PCIe Link Status")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Current Gen / Width | PCIe Gen{_safe(pcie_info.get('pcie_gen_current'))} x{_safe(pcie_info.get('pcie_width_current'))} |")
        lines.append(f"| Max Gen / Width | PCIe Gen{_safe(pcie_info.get('pcie_gen_max'))} x{_safe(pcie_info.get('pcie_width_max'))} |")
        lines.append(f"| Current BW | {_safe(pcie_info.get('bandwidth_gbps_current'), '.2f')} GB/s |")
        lines.append(f"| Theoretical Max BW | {_safe(pcie_info.get('bandwidth_gbps_theoretical'), '.2f')} GB/s |")
        is_bottn = pcie_info.get("is_bottlenecked")
        lines.append(f"| Bottlenecked | {'Yes' if is_bottn else 'No' if is_bottn is not None else 'N/A'} |")
        lines.append("")

    # Toolchain
    lines.append("## Toolchain Status")
    lines.append("")
    lines.append("| Tool | Available | Version |")
    lines.append("|------|-----------|---------|")
    for tool_name, status in tools.items():
        if isinstance(status, dict):
            exists = "Yes" if status.get("exists") else "No"
            version = status.get("version") or "N/A"
        else:
            exists = "Unknown"
            version = "N/A"
        lines.append(f"| {tool_name} | {exists} | {version} |")
    lines.append("")

    # Benchmark results
    lines.append("## Benchmark Results")
    lines.append("")
    if results:
        lines.append("| Test | Mean (ms) | Bandwidth (GB/s) | Speedup | Correct |")
        lines.append("|------|-----------|------------------|---------|---------|")
        for r in results:
            r_d = r if isinstance(r, dict) else (r.to_dict() if hasattr(r, "to_dict") else {})
            test_name = r_d.get("test_name", "")
            mean = r_d.get("mean")
            bw = r_d.get("bandwidth_gbps")
            speedup = r_d.get("speedup")
            correct = r_d.get("correctness_pass")
            correct_str = "Pass" if correct else ("Fail" if correct is False else "N/A")
            lines.append(
                f"| {test_name} "
                f"| {_safe(mean, '.3f')} "
                f"| {_safe(bw, '.2f')} "
                f"| {_safe(speedup, '.1f')} "
                f"| {correct_str} |"
            )
    else:
        lines.append("No benchmark results.")
    lines.append("")

    # Diagnosis
    lines.append("## Diagnosis Results")
    lines.append("")
    if diag_results:
        _SEVERITY_ICONS = {"CRITICAL": "🔴", "ERROR": "🔴", "WARNING": "🟡", "INFO": "🟢"}
        for d in diag_results:
            sev = d.get("severity", "INFO")
            icon = _SEVERITY_ICONS.get(sev, "")
            title = d.get("title", "")
            cat = d.get("category", "")
            summary = d.get("summary", "")
            evidence = d.get("evidence", "")
            rec = d.get("recommendation", "")
            conf = d.get("confidence", 0)
            lines.append(f"### {icon} [{sev}] {title}")
            lines.append("")
            lines.append(f"**Category:** {cat}  ")
            lines.append(f"**Confidence:** {conf*100:.0f}%  ")
            lines.append("")
            lines.append(f"{summary}")
            lines.append("")
            if evidence:
                lines.append(f"**Evidence:** {evidence}  ")
                lines.append("")
            if rec:
                lines.append(f"**Recommendation:** {rec}  ")
                lines.append("")
    else:
        lines.append("No diagnosis results.")
    lines.append("")

    # Missing data section
    score_details = data.get("score_details", {}) or {}
    missing_items = score_details.get("missing_data", []) or []
    if missing_items:
        lines.append("## Missing Data")
        lines.append("")
        for m in missing_items:
            lines.append(f"- {m}")
        lines.append("")

    # Interview Demo Summary
    lines.append("## Interview Demo Summary — Implemented vs. Roadmap Features")
    lines.append("")
    lines.append("### Implemented Features")
    lines.append("")
    for feat in [
        "System Inspector (CPU/GPU/PCIe/CUDA telemetry)",
        "Memory Benchmark (H2D/D2H/D2D via native CUDA binary)",
        "Kernel Lab (vector_add, reduction, transpose, gemm_naive, gemm_tiled, stream_pipeline)",
        "Evidence-based Diagnosis Engine (9 rules, non-empty evidence strings required)",
        "GPU Insight Score (0-100 composite across 6 categories)",
        "Multi-format Reports (JSON, CSV, Markdown, HTML, Excel)",
        "SQLite Session History with comparison",
        "PySide6 GUI (QMainWindow + QThread workers)",
        "CLI Automation (argparse, 10 commands, --json flag)",
        "CUDA to HIP Portability Demo (vector_add_hip.cpp — NOT_VALIDATED on AMD hardware)",
    ]:
        lines.append(f"- {feat}")
    lines.append("")
    lines.append("### Roadmap / NOT YET IMPLEMENTED")
    lines.append("")
    for feat in [
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
        "ROADMAP: Batch execution",
        "NOT_VALIDATED: AMD HIP real GPU benchmarks (requires ROCm hardware)",
    ]:
        lines.append(f"- {feat}")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "GPU Insight Lab provides evidence-based diagnostics based on measurable metrics. "
        "It is not a replacement for Nsight Systems or Nsight Compute for deep kernel profiling. "
        "Low GPU utilization does not automatically mean the GPU is the bottleneck. "
        "Low occupancy does not automatically indicate poor performance."
    )
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated by {APP_NAME} v{APP_VERSION}. "
        "NVIDIA, CUDA, Nsight, AMD, ROCm, HIP are trademarks of their respective owners. "
        "GPU Insight Lab is an independent project not affiliated with or endorsed by NVIDIA or AMD.*"
    )

    try:
        file_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report written to %s", file_path)
    except OSError as exc:
        logger.error("Failed to write Markdown report: %s", exc)
        raise

    return file_path
