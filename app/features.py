"""
GPU Insight Lab - Feature Registry
All features enabled in v0.1.0 (no DRM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Feature:
    feature_id: str
    name: str
    edition: str  # "free", "pro", "lab"
    enabled: bool
    description: str


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

_FEATURE_LIST: List[Feature] = [
    Feature("system_collector", "System Information Collector", "free", True,
            "Collect OS, CPU, RAM, and GPU hardware metadata"),
    Feature("nvidia_collector", "NVIDIA GPU Collector", "free", True,
            "Collect NVIDIA GPU telemetry via pynvml or nvidia-smi"),
    Feature("amd_collector", "AMD GPU Collector", "free", True,
            "Detect AMD GPU and ROCm stack (NOT_VALIDATED in v0.1.0)"),
    Feature("pcie_analyzer", "PCIe Link Analyzer", "free", True,
            "Report PCIe generation, width, and effective bandwidth"),
    Feature("cuda_collector", "CUDA Toolchain Detector", "free", True,
            "Detect nvcc, CUDA_HOME, runtime version"),
    Feature("tool_checker", "Toolchain Availability Checker", "free", True,
            "Check cmake, nvcc, cl.exe, nsys, ncu, rocminfo, hipcc"),
    Feature("native_benchmarks", "Native CUDA Benchmarks", "free", True,
            "Run compiled CUDA kernels: vector_add, reduction, transpose, GEMM, memory bandwidth"),
    Feature("memory_bandwidth", "Memory Bandwidth Test", "free", True,
            "Pageable vs pinned H2D/D2H transfer benchmarks"),
    Feature("stream_pipeline", "CUDA Stream Pipeline Benchmark", "free", True,
            "Sync vs async stream overlap test"),
    Feature("image_grayscale", "Image Grayscale Benchmark", "free", True,
            "CPU vs CUDA grayscale conversion"),
    Feature("diagnosis_engine", "Diagnosis Engine", "free", True,
            "Evidence-based rule engine for GPU performance analysis"),
    Feature("gpu_insight_score", "GPU Insight Score", "free", True,
            "Composite 0-100 score for GPU environment readiness and performance"),
    Feature("report_json", "JSON Report", "free", True,
            "Export session data as machine-readable JSON"),
    Feature("report_csv", "CSV Report", "free", True,
            "Export benchmark results as CSV"),
    Feature("report_markdown", "Markdown Report", "free", True,
            "Export GitHub-compatible Markdown report"),
    Feature("report_html", "HTML Report", "free", True,
            "Generate self-contained HTML report with Jinja2"),
    Feature("report_excel", "Excel Report", "pro", True,
            "Generate management-readable Excel workbook with multiple sheets"),
    Feature("session_history", "Session History", "free", True,
            "Store and query benchmark sessions in SQLite"),
    Feature("session_compare", "Session Comparison", "pro", True,
            "Compare two benchmark sessions side-by-side"),
    Feature("nsight_systems_profile", "Nsight Systems Profiling", "pro", True,
            "Launch nsys for CUDA timeline capture (requires nsys installed)"),
    Feature("nsight_compute_profile", "Nsight Compute Profiling", "pro", True,
            "Launch ncu for kernel-level metrics (requires ncu installed)"),
    Feature("nvidia_smi_monitor", "Real-time nvidia-smi Monitor", "free", True,
            "Poll GPU temperature, power, utilization during benchmarks"),
    Feature("image_batch_workload", "Image Batch Workload", "lab", True,
            "Batch image preprocessing benchmark with CPU/GPU comparison"),
    Feature("llm_benchmark", "LLM Benchmark Workload", "lab", True,
            "Placeholder for LLM inference throughput benchmarking"),
    Feature("custom_command_workload", "Custom Command Workload", "lab", True,
            "Run user-supplied command and measure GPU impact"),
    Feature("hip_support", "HIP/ROCm Support", "lab", True,
            "AMD HIP vector add and ROCm toolchain detection (NOT_VALIDATED)"),
    Feature("gui", "Graphical User Interface", "free", True,
            "PySide6 desktop GUI with dashboard, benchmark, history, and report pages"),
    Feature("cli", "Command Line Interface", "free", True,
            "argparse-based CLI for headless and scripted usage"),
]

# Build lookup dict
_FEATURE_MAP: Dict[str, Feature] = {f.feature_id: f for f in _FEATURE_LIST}


def get_feature(feature_id: str) -> Feature | None:
    """Return Feature by ID, or None if not found."""
    return _FEATURE_MAP.get(feature_id)


def is_enabled(feature_id: str) -> bool:
    """Return True if feature exists and is enabled."""
    feat = _FEATURE_MAP.get(feature_id)
    return feat is not None and feat.enabled


def list_features(edition: str | None = None) -> List[Feature]:
    """Return all features, optionally filtered by edition."""
    if edition is None:
        return list(_FEATURE_LIST)
    return [f for f in _FEATURE_LIST if f.edition == edition]


def list_enabled_features() -> List[Feature]:
    return [f for f in _FEATURE_LIST if f.enabled]
