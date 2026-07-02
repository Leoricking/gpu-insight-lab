"""
GPU Insight Lab - GPU Insight Score Calculator
Composite 0-100 score for GPU environment readiness and performance.
Never assumes normal when data is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    score: float = 0.0                           # 0-100
    confidence: float = 0.0                      # 0-1.0
    missing_data: List[str] = field(default_factory=list)
    deductions: List[Dict[str, Any]] = field(default_factory=list)
    positive_findings: List[str] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


# Score breakdown
_MAX_SCORES = {
    "environment_readiness": 20,
    "gpu_runtime_availability": 15,
    "pcie_memory_transfer": 20,
    "kernel_correctness": 20,
    "kernel_performance_consistency": 15,
    "thermal_power_stability": 10,
}


def _check_environment_readiness(session: Any) -> tuple[float, List[str], List[str], List[Dict]]:
    """
    Environment readiness: 20 pts
    - Python + psutil available: 4 pts
    - nvcc available: 5 pts
    - cmake available: 4 pts
    - nvidia-smi available: 4 pts
    - Native benchmark built: 3 pts
    """
    earned = 0.0
    missing: List[str] = []
    positive: List[str] = []
    deductions: List[Dict] = []
    MAX = _MAX_SCORES["environment_readiness"]

    tools = getattr(session, "tool_status", {}) or {}
    cuda_info = getattr(session, "cuda_info", {}) or {}

    def _exists(t: str) -> bool:
        e = tools.get(t)
        return bool(e.get("exists", False)) if isinstance(e, dict) else False

    # psutil (always available if we got here)
    earned += 4
    positive.append("Python environment operational")

    if _exists("nvcc"):
        earned += 5
        positive.append("nvcc available")
    else:
        missing.append("nvcc")
        deductions.append({"category": "environment_readiness", "pts": -5, "reason": "nvcc missing"})

    if _exists("cmake"):
        earned += 4
        positive.append("cmake available")
    else:
        missing.append("cmake")
        deductions.append({"category": "environment_readiness", "pts": -4, "reason": "cmake missing"})

    if _exists("nvidia-smi"):
        earned += 4
        positive.append("nvidia-smi available")
    else:
        missing.append("nvidia-smi")
        deductions.append({"category": "environment_readiness", "pts": -4, "reason": "nvidia-smi missing"})

    if cuda_info.get("native_benchmark_available"):
        earned += 3
        positive.append("Native benchmark executable present")
    else:
        missing.append("native_benchmark_executable")
        deductions.append({"category": "environment_readiness", "pts": -3, "reason": "native executable not built"})

    return min(earned, MAX), missing, positive, deductions


def _check_gpu_runtime(session: Any) -> tuple[float, List[str], List[str], List[Dict]]:
    """
    GPU runtime: 15 pts
    - NVIDIA GPU detected: 8 pts
    - CUDA runtime available: 4 pts
    - Driver/runtime version match: 3 pts
    """
    earned = 0.0
    missing: List[str] = []
    positive: List[str] = []
    deductions: List[Dict] = []
    MAX = _MAX_SCORES["gpu_runtime_availability"]

    nvidia_info = getattr(session, "nvidia_info", {}) or {}
    cuda_info = getattr(session, "cuda_info", {}) or {}

    if nvidia_info.get("available"):
        earned += 8
        gpu_name = nvidia_info.get("gpu_name", "Unknown GPU")
        positive.append(f"NVIDIA GPU detected: {gpu_name}")
    else:
        missing.append("nvidia_gpu")
        deductions.append({"category": "gpu_runtime", "pts": -8, "reason": "No NVIDIA GPU detected"})

    if cuda_info.get("cuda_runtime_available") or cuda_info.get("nvcc_available"):
        earned += 4
        positive.append("CUDA runtime available")
    else:
        missing.append("cuda_runtime")
        deductions.append({"category": "gpu_runtime", "pts": -4, "reason": "CUDA runtime not detected"})

    # Version match
    driver_ver = nvidia_info.get("cuda_driver_version")
    runtime_ver = cuda_info.get("cuda_runtime_version") or cuda_info.get("nvcc_version")
    if driver_ver and runtime_ver:
        def _mm(v: str) -> str:
            parts = str(v).split(".")
            return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(v)
        if _mm(driver_ver) == _mm(runtime_ver):
            earned += 3
            positive.append(f"CUDA driver/runtime versions match ({driver_ver})")
        else:
            deductions.append({
                "category": "gpu_runtime",
                "pts": -3,
                "reason": f"CUDA version mismatch: driver={driver_ver}, runtime={runtime_ver}",
            })
    else:
        missing.append("cuda_version_comparison")

    return min(earned, MAX), missing, positive, deductions


def _check_pcie_memory(session: Any) -> tuple[float, List[str], List[str], List[Dict]]:
    """
    PCIe/memory transfer: 20 pts
    - PCIe info collected: 5 pts
    - Running at max PCIe gen: 5 pts
    - Memory bandwidth test ran: 5 pts
    - Pinned bandwidth available or transfer overhead < 40%: 5 pts
    """
    earned = 0.0
    missing: List[str] = []
    positive: List[str] = []
    deductions: List[Dict] = []
    MAX = _MAX_SCORES["pcie_memory_transfer"]

    pcie_info = getattr(session, "pcie_info", {}) or {}
    results_map = {getattr(r, "test_name", ""): r for r in getattr(session, "results", [])}

    if pcie_info.get("available"):
        earned += 5
        gen = pcie_info.get("pcie_gen_current")
        width = pcie_info.get("pcie_width_current")
        positive.append(f"PCIe info collected: Gen{gen} x{width}")
        if not pcie_info.get("is_bottlenecked"):
            earned += 5
            positive.append("PCIe running at max capability")
        else:
            gen_max = pcie_info.get("pcie_gen_max")
            width_max = pcie_info.get("pcie_width_max")
            deductions.append({
                "category": "pcie_memory",
                "pts": -5,
                "reason": f"PCIe below max: current=Gen{gen}x{width}, max=Gen{gen_max}x{width_max}",
            })
    else:
        missing.append("pcie_info")
        deductions.append({"category": "pcie_memory", "pts": -5, "reason": "PCIe info unavailable"})

    if "memory_bandwidth" in results_map:
        r = results_map["memory_bandwidth"]
        if getattr(r, "error", None) is None:
            earned += 5
            positive.append("Memory bandwidth test completed")
        else:
            deductions.append({"category": "pcie_memory", "pts": -3, "reason": "Memory bandwidth test had errors"})
    else:
        missing.append("memory_bandwidth_test")

    # Transfer overhead check
    for r in getattr(session, "results", []):
        t_ms = getattr(r, "transfer_time_ms", None)
        e2e_ms = getattr(r, "end_to_end_time_ms", None)
        if t_ms and e2e_ms and e2e_ms > 0:
            if t_ms / e2e_ms < 0.40:
                earned += 5
                positive.append("Transfer overhead below 40% threshold")
            else:
                deductions.append({
                    "category": "pcie_memory",
                    "pts": -5,
                    "reason": f"Transfer overhead {t_ms/e2e_ms*100:.0f}% in {getattr(r,'test_name','')}",
                })
            break
    else:
        missing.append("transfer_overhead_data")

    return min(earned, MAX), missing, positive, deductions


def _check_kernel_correctness(session: Any) -> tuple[float, List[str], List[str], List[Dict]]:
    """
    Kernel correctness: 20 pts
    - Any GPU kernels ran: 5 pts
    - All ran correctness checks: 5 pts
    - All passed correctness: 10 pts
    """
    earned = 0.0
    missing: List[str] = []
    positive: List[str] = []
    deductions: List[Dict] = []
    MAX = _MAX_SCORES["kernel_correctness"]

    results = [r for r in getattr(session, "results", []) if not getattr(r, "test_name", "").startswith("cpu_")]

    if not results:
        missing.append("gpu_kernel_results")
        deductions.append({
            "category": "kernel_correctness",
            "pts": -20,
            "reason": "No GPU kernel results available",
        })
        return 0.0, missing, positive, deductions

    earned += 5
    positive.append(f"{len(results)} GPU kernel test(s) ran")

    checked = [r for r in results if getattr(r, "correctness_pass", None) is not None]
    if checked:
        earned += 5
        positive.append(f"{len(checked)} test(s) included correctness checks")
        passed = [r for r in checked if getattr(r, "correctness_pass", False)]
        if len(passed) == len(checked):
            earned += 10
            positive.append("All correctness checks passed")
        else:
            failed_count = len(checked) - len(passed)
            deductions.append({
                "category": "kernel_correctness",
                "pts": -10,
                "reason": f"{failed_count} kernel(s) failed correctness check",
            })
    else:
        missing.append("correctness_check_results")
        deductions.append({
            "category": "kernel_correctness",
            "pts": -5,
            "reason": "No correctness data in results",
        })

    return min(earned, MAX), missing, positive, deductions


def _check_performance_consistency(session: Any) -> tuple[float, List[str], List[str], List[Dict]]:
    """
    Kernel performance consistency: 15 pts
    - Any results with std_dev: 5 pts
    - All CoV < 10%: 5 pts
    - Speedup > 1x for at least one test: 5 pts
    """
    earned = 0.0
    missing: List[str] = []
    positive: List[str] = []
    deductions: List[Dict] = []
    MAX = _MAX_SCORES["kernel_performance_consistency"]

    results = [r for r in getattr(session, "results", [])
               if getattr(r, "standard_deviation", None) is not None
               and getattr(r, "mean", None) is not None]

    if not results:
        missing.append("performance_statistics")
        return 0.0, missing, positive, deductions

    earned += 5
    positive.append(f"{len(results)} result(s) have performance statistics")

    high_cov = []
    for r in results:
        mean_v = getattr(r, "mean", 0) or 0
        std_v = getattr(r, "standard_deviation", 0) or 0
        if mean_v > 0:
            cov = std_v / mean_v
            if cov > 0.10:
                high_cov.append(f"{getattr(r,'test_name','?')}(CoV={cov:.2f})")

    if not high_cov:
        earned += 5
        positive.append("All benchmarks have CoV < 10% (good consistency)")
    else:
        deductions.append({
            "category": "performance_consistency",
            "pts": -5,
            "reason": f"High variability (CoV>10%): {', '.join(high_cov)}",
        })

    # Speedup check
    speedup_results = [r for r in getattr(session, "results", [])
                       if getattr(r, "speedup", None) is not None
                       and getattr(r, "speedup", 0) > 1.0]
    if speedup_results:
        earned += 5
        best = max(speedup_results, key=lambda r: getattr(r, "speedup", 0))
        positive.append(
            f"GPU speedup detected: {getattr(best,'speedup',0):.1f}x in {getattr(best,'test_name','?')}"
        )
    else:
        missing.append("gpu_speedup_data")

    return min(earned, MAX), missing, positive, deductions


def _check_thermal_power(session: Any) -> tuple[float, List[str], List[str], List[Dict]]:
    """
    Thermal/power stability: 10 pts
    - Temperature data available: 3 pts
    - Temperature < 85°C: 4 pts
    - Performance state P0-P2: 3 pts
    """
    earned = 0.0
    missing: List[str] = []
    positive: List[str] = []
    deductions: List[Dict] = []
    MAX = _MAX_SCORES["thermal_power_stability"]

    nvidia_info = getattr(session, "nvidia_info", {}) or {}
    temp = nvidia_info.get("temperature_c")
    perf_state = nvidia_info.get("performance_state")

    if temp is None:
        missing.append("gpu_temperature")
        deductions.append({"category": "thermal_power", "pts": -7, "reason": "No temperature data"})
        return 0.0, missing, positive, deductions

    earned += 3
    positive.append(f"GPU temperature data available: {temp:.0f}°C")

    if float(temp) < 85.0:
        earned += 4
        positive.append(f"GPU temperature {temp:.0f}°C is below 85°C threshold")
    else:
        deductions.append({
            "category": "thermal_power",
            "pts": -4,
            "reason": f"GPU temperature {temp:.0f}°C above 85°C",
        })

    if perf_state:
        try:
            p_num = int(perf_state.lstrip("P"))
            if p_num <= 2:
                earned += 3
                positive.append(f"GPU performance state {perf_state} (high performance)")
            else:
                deductions.append({
                    "category": "thermal_power",
                    "pts": -3,
                    "reason": f"Performance state {perf_state} (possible throttling)",
                })
        except ValueError:
            missing.append("performance_state_parse")
    else:
        missing.append("gpu_performance_state")

    return min(earned, MAX), missing, positive, deductions


def compute_score(session: Any) -> ScoreResult:
    """
    Compute the GPU Insight Score (0-100).
    Never assumes normal when data is missing.
    """
    result = ScoreResult()
    total_score = 0.0
    all_missing: List[str] = []
    all_positive: List[str] = []
    all_deductions: List[Dict] = []

    checkers = [
        ("environment_readiness", _check_environment_readiness),
        ("gpu_runtime_availability", _check_gpu_runtime),
        ("pcie_memory_transfer", _check_pcie_memory),
        ("kernel_correctness", _check_kernel_correctness),
        ("kernel_performance_consistency", _check_performance_consistency),
        ("thermal_power_stability", _check_thermal_power),
    ]

    for category, fn in checkers:
        try:
            earned, missing, positive, deductions = fn(session)
            result.category_scores[category] = round(earned, 1)
            total_score += earned
            all_missing.extend(missing)
            all_positive.extend(positive)
            all_deductions.extend(deductions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Score category %s failed: %s", category, exc)
            result.category_scores[category] = 0.0
            all_missing.append(category)

    result.score = round(min(max(total_score, 0.0), 100.0), 1)
    result.missing_data = list(set(all_missing))
    result.positive_findings = all_positive
    result.deductions = all_deductions

    # Confidence: based on how much data we have
    data_completeness = 1.0 - min(len(all_missing) / 15.0, 1.0)
    result.confidence = round(max(0.1, data_completeness), 2)

    result.notes = (
        f"Score computed from {len(checkers)} categories. "
        f"Missing data items: {len(all_missing)}. "
        f"Confidence reflects data completeness, not GPU quality."
    )

    return result
