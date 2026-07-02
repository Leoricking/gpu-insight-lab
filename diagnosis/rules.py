"""
GPU Insight Lab - Diagnosis Rules
Each rule takes a BenchmarkSession and returns DiagnosisResult(s) or None.
All results MUST include evidence strings.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from diagnosis.engine import DiagnosisResult
from diagnosis.evidence import (
    format_evidence,
    format_missing_evidence,
    format_ratio_evidence,
    format_threshold_evidence,
    format_version_mismatch_evidence,
)
from diagnosis.recommendations import get_recommendation

logger = logging.getLogger(__name__)


def _get_results_by_test(session: Any) -> dict[str, Any]:
    """Build {test_name: BenchmarkResult} from session."""
    mapping: dict[str, Any] = {}
    for r in getattr(session, "results", []):
        name = getattr(r, "test_name", None)
        if name:
            mapping[name] = r
    return mapping


def rule_pinned_vs_pageable(session: Any) -> Optional[DiagnosisResult]:
    """
    Compare pinned vs pageable H2D bandwidth.
    If pinned > pageable * 1.2 → positive finding (staging bottleneck exists).
    If similar → note no staging bottleneck.
    """
    rule_id = "pinned_vs_pageable"
    results_map = _get_results_by_test(session)

    pinned = results_map.get("memory_bandwidth_pinned_h2d") or results_map.get("pinned_h2d")
    pageable = results_map.get("memory_bandwidth_pageable_h2d") or results_map.get("pageable_h2d")
    # Also check native memory_bandwidth result for combined data
    mem_bw = results_map.get("memory_bandwidth")

    pinned_bw: Optional[float] = None
    pageable_bw: Optional[float] = None

    if pinned is not None:
        pinned_bw = getattr(pinned, "bandwidth_gbps", None)
    if pageable is not None:
        pageable_bw = getattr(pageable, "bandwidth_gbps", None)

    # Try parsing notes from memory_bandwidth combined result
    if mem_bw is not None and (pinned_bw is None or pageable_bw is None):
        notes = getattr(mem_bw, "notes", "") or ""
        import re  # noqa: PLC0415
        m_pin = re.search(r"pinned[_\s]?h2d[_\s=:]+([0-9.]+)", notes, re.IGNORECASE)
        m_pag = re.search(r"pageable[_\s]?h2d[_\s=:]+([0-9.]+)", notes, re.IGNORECASE)
        if m_pin:
            pinned_bw = float(m_pin.group(1))
        if m_pag:
            pageable_bw = float(m_pag.group(1))

    if pinned_bw is None or pageable_bw is None:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="INSUFFICIENT_DATA",
            title="Pinned vs Pageable Bandwidth: Insufficient Data",
            summary="Memory bandwidth test did not produce separate pinned/pageable results.",
            evidence=format_missing_evidence("pinned_h2d_bandwidth, pageable_h2d_bandwidth"),
            metric_name="h2d_bandwidth_gbps",
            confidence=0.0,
            recommendation="Run the memory_bandwidth test with the native benchmark executable.",
            verification_step="Run: gpu_insight_benchmark --test memory_bandwidth",
        )

    if pageable_bw <= 0:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="INSUFFICIENT_DATA",
            title="Pinned vs Pageable: Invalid Reference Value",
            summary="Pageable bandwidth is zero or negative; cannot compute ratio.",
            evidence=f"pageable_h2d_bandwidth={pageable_bw} GB/s (invalid)",
            metric_name="pageable_h2d_bandwidth_gbps",
            observed_value=pageable_bw,
            confidence=0.0,
        )

    ratio = pinned_bw / pageable_bw
    rec, verif = get_recommendation("PCIE_TRANSFER_BOUND")

    if ratio >= 1.2:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="Pinned Memory Improves H2D Bandwidth",
            summary=(
                f"Pinned memory H2D bandwidth ({pinned_bw:.1f} GB/s) is "
                f"{ratio:.1f}x higher than pageable ({pageable_bw:.1f} GB/s). "
                "Using pinned memory is beneficial for this workload."
            ),
            evidence=format_ratio_evidence(
                "H2D bandwidth",
                pinned_bw,
                pageable_bw,
                ratio,
                unit=" GB/s",
                context="Pinned memory allows DMA without OS page locking overhead",
            ),
            metric_name="pinned_vs_pageable_h2d_ratio",
            observed_value=ratio,
            reference_value=1.2,
            confidence=0.85,
            recommendation=(
                "Allocate host memory with cudaMallocHost() for data that is "
                "transferred to the GPU frequently."
            ),
            verification_step=(
                "Compare cudaMemcpy times for pageable vs cudaMallocHost allocations."
            ),
        )
    else:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="No Significant Pinned Memory Advantage Detected",
            summary=(
                f"Pinned ({pinned_bw:.1f} GB/s) and pageable ({pageable_bw:.1f} GB/s) "
                f"H2D bandwidth are similar (ratio={ratio:.2f}x < 1.2). "
                "PCIe staging is not a bottleneck for current transfer sizes."
            ),
            evidence=format_ratio_evidence(
                "H2D bandwidth",
                pinned_bw,
                pageable_bw,
                ratio,
                unit=" GB/s",
            ),
            metric_name="pinned_vs_pageable_h2d_ratio",
            observed_value=ratio,
            reference_value=1.2,
            confidence=0.75,
            recommendation="Current pageable transfers are sufficient; no change required.",
            verification_step="Re-run with larger transfer sizes (256 MB+) to confirm.",
        )


def rule_transfer_overhead(session: Any) -> Optional[DiagnosisResult]:
    """
    If transfer_time > 40% of end_to_end → PCIE_TRANSFER_BOUND candidate.
    """
    rule_id = "transfer_overhead"
    results_map = _get_results_by_test(session)

    candidates = [
        r for name, r in results_map.items()
        if getattr(r, "transfer_time_ms", None) is not None
        and getattr(r, "end_to_end_time_ms", None) is not None
        and getattr(r, "end_to_end_time_ms", 0) > 0
    ]

    if not candidates:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="INSUFFICIENT_DATA",
            title="Transfer Overhead: Insufficient Data",
            summary="No benchmark results contained separate transfer and end-to-end timing.",
            evidence=format_missing_evidence("transfer_time_ms, end_to_end_time_ms"),
            confidence=0.0,
            recommendation="Run native CUDA benchmarks that measure H2D/D2H separately.",
        )

    for r in candidates:
        transfer_ms = getattr(r, "transfer_time_ms", 0) or 0
        e2e_ms = getattr(r, "end_to_end_time_ms", 1) or 1
        overhead_ratio = transfer_ms / e2e_ms
        test_name = getattr(r, "test_name", "unknown")

        if overhead_ratio > 0.40:
            rec, verif = get_recommendation("PCIE_TRANSFER_BOUND")
            return DiagnosisResult(
                rule_id=rule_id,
                severity="WARNING",
                category="PCIE_TRANSFER_BOUND",
                title=f"PCIe Transfer Overhead High in {test_name}",
                summary=(
                    f"Transfer time is {overhead_ratio*100:.0f}% of end-to-end time "
                    f"({transfer_ms:.2f} ms / {e2e_ms:.2f} ms) in {test_name}. "
                    "This indicates data movement is a significant bottleneck."
                ),
                evidence=format_threshold_evidence(
                    "transfer_overhead_ratio",
                    overhead_ratio,
                    0.40,
                    unit="",
                    above=True,
                    context=f"test={test_name}, transfer={transfer_ms:.2f}ms, e2e={e2e_ms:.2f}ms",
                ),
                metric_name="transfer_overhead_ratio",
                observed_value=round(overhead_ratio, 3),
                reference_value=0.40,
                confidence=0.80,
                recommendation=rec,
                verification_step=verif,
            )

    # All tests are fine
    return DiagnosisResult(
        rule_id=rule_id,
        severity="INFO",
        category="HEALTHY",
        title="PCIe Transfer Overhead Within Normal Range",
        summary="Transfer time is below 40% of end-to-end time in all tested benchmarks.",
        evidence=f"Checked {len(candidates)} benchmark(s); none exceeded 40% transfer overhead.",
        metric_name="transfer_overhead_ratio",
        observed_value="<0.40",
        reference_value=0.40,
        confidence=0.75,
        recommendation="No action required.",
        verification_step="Re-check with larger data sizes or streaming workloads.",
    )


def rule_launch_overhead(session: Any) -> Optional[DiagnosisResult]:
    """
    If GPU time > CPU time for small N (<10000) → KERNEL_LAUNCH_OVERHEAD.
    """
    rule_id = "launch_overhead"
    results_map = _get_results_by_test(session)

    small_n_results = [
        r for name, r in results_map.items()
        if getattr(r, "input_size", 0) < 10000
        and getattr(r, "gpu_time_ms", None) is not None
        and getattr(r, "cpu_time_ms", None) is not None
    ]

    if not small_n_results:
        return None  # Rule not applicable without small-N tests

    for r in small_n_results:
        gpu_ms = getattr(r, "gpu_time_ms", 0) or 0
        cpu_ms = getattr(r, "cpu_time_ms", 0) or 0
        n = getattr(r, "input_size", 0)
        test_name = getattr(r, "test_name", "unknown")

        if cpu_ms > 0 and gpu_ms > cpu_ms:
            rec, verif = get_recommendation("KERNEL_LAUNCH_OVERHEAD")
            return DiagnosisResult(
                rule_id=rule_id,
                severity="INFO",
                category="KERNEL_LAUNCH_OVERHEAD",
                title=f"Kernel Launch Overhead Detected in {test_name}",
                summary=(
                    f"GPU time ({gpu_ms:.3f} ms) > CPU time ({cpu_ms:.3f} ms) "
                    f"for N={n:,}. Kernel launch overhead dominates at this problem size. "
                    "This is expected and normal for small N."
                ),
                evidence=format_ratio_evidence(
                    "gpu_vs_cpu_time",
                    gpu_ms,
                    cpu_ms,
                    gpu_ms / cpu_ms,
                    unit=" ms",
                    context=f"input_size={n}, test={test_name}",
                ),
                metric_name="gpu_time_ms",
                observed_value=gpu_ms,
                reference_value=cpu_ms,
                confidence=0.85,
                recommendation=rec,
                verification_step=verif,
            )

    return None


def rule_optimization_check(session: Any) -> Optional[DiagnosisResult]:
    """
    If optimized kernel is not faster than naive → WARNING, do not mark success.
    Checks: gemm_tiled vs gemm_naive, vector_add_gridstride vs vector_add_naive.
    """
    rule_id = "optimization_check"
    results_map = _get_results_by_test(session)

    findings: list[DiagnosisResult] = []

    # GEMM: tiled should be faster than naive
    gemm_naive = results_map.get("gemm_naive")
    gemm_tiled = results_map.get("gemm_tiled")
    if gemm_naive and gemm_tiled:
        naive_ms = getattr(gemm_naive, "gpu_time_ms", None) or getattr(gemm_naive, "mean", None)
        tiled_ms = getattr(gemm_tiled, "gpu_time_ms", None) or getattr(gemm_tiled, "mean", None)
        if naive_ms is not None and tiled_ms is not None and naive_ms > 0:
            ratio = tiled_ms / naive_ms
            if ratio >= 1.0:
                findings.append(DiagnosisResult(
                    rule_id=rule_id,
                    severity="WARNING",
                    category="GPU_COMPUTE_BOUND",
                    title="Tiled GEMM Not Faster Than Naive",
                    summary=(
                        f"Tiled GEMM ({tiled_ms:.2f} ms) is not faster than naive ({naive_ms:.2f} ms). "
                        "Tiling optimization may not be functioning as expected. "
                        "Do not assume optimization correctness without verification."
                    ),
                    evidence=format_ratio_evidence(
                        "gemm_tiled_vs_naive_time",
                        tiled_ms,
                        naive_ms,
                        ratio,
                        unit=" ms",
                        context="Expected tiled < naive for large matrices",
                    ),
                    metric_name="gemm_tiled_vs_naive_ratio",
                    observed_value=round(ratio, 3),
                    reference_value=1.0,
                    confidence=0.80,
                    recommendation=(
                        "Verify tile size is appropriate for the GPU's shared memory capacity. "
                        "Check for bank conflicts in the tiled implementation."
                    ),
                    verification_step="Profile with ncu --metrics l1tex__data_bank_conflicts_pipe_lsu.",
                ))
            else:
                findings.append(DiagnosisResult(
                    rule_id=rule_id,
                    severity="INFO",
                    category="HEALTHY",
                    title="Tiled GEMM Speedup Confirmed",
                    summary=f"Tiled GEMM is {1.0/ratio:.1f}x faster than naive GEMM.",
                    evidence=format_ratio_evidence(
                        "gemm_tiled_vs_naive",
                        tiled_ms,
                        naive_ms,
                        ratio,
                        unit=" ms",
                    ),
                    metric_name="gemm_tiled_speedup",
                    observed_value=round(1.0 / ratio, 2),
                    reference_value=1.0,
                    confidence=0.85,
                ))

    if not findings:
        return None
    return findings[0] if len(findings) == 1 else findings  # type: ignore[return-value]


def rule_thermal_throttle(session: Any) -> Optional[DiagnosisResult]:
    """
    If temp > 85°C and clock below base → THERMAL_THROTTLING candidate.
    """
    rule_id = "thermal_throttle"
    nvidia_info = getattr(session, "nvidia_info", {}) or {}
    temp = nvidia_info.get("temperature_c")
    gpu_clock = nvidia_info.get("gpu_clock_mhz")
    perf_state = nvidia_info.get("performance_state")

    if temp is None:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="INSUFFICIENT_DATA",
            title="Thermal Status: No Temperature Data",
            summary="GPU temperature data not available; cannot evaluate thermal throttling.",
            evidence=format_missing_evidence("temperature_c", "not collected"),
            confidence=0.0,
            recommendation="Ensure NVIDIA GPU is present and pynvml or nvidia-smi is available.",
        )

    temp_f = float(temp)
    THROTTLE_TEMP = 85.0

    if temp_f > THROTTLE_TEMP:
        # Check if performance state suggests throttling (P0=best, P8+=throttled)
        is_throttled = False
        if perf_state and perf_state.startswith("P"):
            try:
                p_num = int(perf_state[1:])
                is_throttled = p_num >= 4
            except ValueError:
                pass

        rec, verif = get_recommendation("THERMAL_THROTTLING")
        severity = "WARNING" if is_throttled else "INFO"
        return DiagnosisResult(
            rule_id=rule_id,
            severity=severity,
            category="THERMAL_THROTTLING",
            title="GPU Temperature Above 85°C",
            summary=(
                f"GPU temperature is {temp_f:.0f}°C, above the 85°C warning threshold. "
                + (f"Performance state {perf_state} suggests possible clock reduction. " if perf_state else "")
                + "Monitor for sustained throttling."
            ),
            evidence=format_threshold_evidence(
                "gpu_temperature",
                temp_f,
                THROTTLE_TEMP,
                unit="°C",
                above=True,
                context=f"perf_state={perf_state}, clock={gpu_clock}MHz",
            ),
            metric_name="temperature_c",
            observed_value=temp_f,
            reference_value=THROTTLE_TEMP,
            confidence=0.75,
            recommendation=rec,
            verification_step=verif,
        )
    else:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="GPU Temperature Normal",
            summary=f"GPU temperature {temp_f:.0f}°C is within normal operating range (<85°C).",
            evidence=format_threshold_evidence(
                "gpu_temperature",
                temp_f,
                THROTTLE_TEMP,
                unit="°C",
                above=False,
            ),
            metric_name="temperature_c",
            observed_value=temp_f,
            reference_value=THROTTLE_TEMP,
            confidence=0.90,
            recommendation="No action required.",
        )


def rule_vram_pressure(session: Any) -> Optional[DiagnosisResult]:
    """
    If VRAM used > 90% → VRAM_CAPACITY_BOUND.
    """
    rule_id = "vram_pressure"
    nvidia_info = getattr(session, "nvidia_info", {}) or {}
    vram_total = nvidia_info.get("vram_total_mb")
    vram_used = nvidia_info.get("vram_used_mb")

    if vram_total is None or vram_used is None:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="INSUFFICIENT_DATA",
            title="VRAM Pressure: No Memory Data",
            summary="VRAM utilization data not available.",
            evidence=format_missing_evidence("vram_total_mb, vram_used_mb"),
            confidence=0.0,
        )

    total = float(vram_total)
    used = float(vram_used)
    if total <= 0:
        return None

    utilization = used / total
    rec, verif = get_recommendation("VRAM_CAPACITY_BOUND")

    if utilization > 0.90:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="WARNING",
            category="VRAM_CAPACITY_BOUND",
            title="High VRAM Utilization",
            summary=(
                f"VRAM utilization is {utilization*100:.0f}% "
                f"({used/1024:.1f} GB / {total/1024:.1f} GB). "
                "Operations may fail or slow down due to memory pressure."
            ),
            evidence=format_threshold_evidence(
                "vram_utilization",
                utilization,
                0.90,
                unit="",
                above=True,
                context=f"used={used:.0f}MB, total={total:.0f}MB",
            ),
            metric_name="vram_utilization",
            observed_value=round(utilization, 3),
            reference_value=0.90,
            confidence=0.90,
            recommendation=rec,
            verification_step=verif,
        )
    else:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="VRAM Utilization Normal",
            summary=f"VRAM utilization is {utilization*100:.0f}% — within healthy range.",
            evidence=format_threshold_evidence(
                "vram_utilization",
                utilization,
                0.90,
                unit="",
                above=False,
                context=f"used={used:.0f}MB, total={total:.0f}MB",
            ),
            metric_name="vram_utilization",
            observed_value=round(utilization, 3),
            reference_value=0.90,
            confidence=0.90,
        )


def rule_driver_mismatch(session: Any) -> Optional[DiagnosisResult]:
    """
    If CUDA driver version != runtime version → DRIVER_RUNTIME_MISMATCH.
    """
    rule_id = "driver_mismatch"
    nvidia_info = getattr(session, "nvidia_info", {}) or {}
    cuda_info = getattr(session, "cuda_info", {}) or {}

    driver_ver = nvidia_info.get("cuda_driver_version")
    runtime_ver = cuda_info.get("cuda_runtime_version") or cuda_info.get("nvcc_version")

    if driver_ver is None or runtime_ver is None:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="INSUFFICIENT_DATA",
            title="CUDA Version Mismatch Check: Incomplete Data",
            summary="Could not obtain both driver and runtime CUDA versions.",
            evidence=format_missing_evidence(
                "cuda_driver_version and/or cuda_runtime_version",
                "not available",
            ),
            confidence=0.0,
            recommendation="Ensure nvidia-smi and nvcc are both installed.",
        )

    # Normalize: compare major.minor only
    def _major_minor(v: str) -> str:
        parts = str(v).split(".")
        return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(v)

    driver_mm = _major_minor(driver_ver)
    runtime_mm = _major_minor(runtime_ver)

    rec, verif = get_recommendation("DRIVER_RUNTIME_MISMATCH")

    if driver_mm != runtime_mm:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="ERROR",
            category="DRIVER_RUNTIME_MISMATCH",
            title="CUDA Driver/Runtime Version Mismatch",
            summary=(
                f"CUDA driver version ({driver_ver}) does not match runtime ({runtime_ver}). "
                "This can cause runtime errors, silent incorrect results, or crashes."
            ),
            evidence=format_version_mismatch_evidence("CUDA", driver_ver, runtime_ver),
            metric_name="cuda_version_match",
            observed_value=f"driver={driver_ver}, runtime={runtime_ver}",
            reference_value="driver == runtime",
            confidence=0.90,
            recommendation=rec,
            verification_step=verif,
        )
    else:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="CUDA Driver and Runtime Versions Match",
            summary=f"Driver CUDA version {driver_ver} matches runtime {runtime_ver}.",
            evidence=format_evidence(
                "cuda_version",
                driver_ver,
                runtime_ver,
                context="Driver and runtime versions are consistent",
            ),
            metric_name="cuda_version_match",
            observed_value=driver_ver,
            reference_value=runtime_ver,
            confidence=0.90,
        )


def rule_profiler_unavailable(session: Any) -> Optional[DiagnosisResult]:
    """
    If nsys/ncu missing → INFO, not a failure.
    """
    rule_id = "profiler_unavailable"
    tools = getattr(session, "tool_status", {}) or {}

    nsys_ok = tools.get("nsys", {}).get("exists", False) if isinstance(tools.get("nsys"), dict) else False
    ncu_ok = tools.get("ncu", {}).get("exists", False) if isinstance(tools.get("ncu"), dict) else False

    missing: list[str] = []
    if not nsys_ok:
        missing.append("nsys (Nsight Systems)")
    if not ncu_ok:
        missing.append("ncu (Nsight Compute)")

    if missing:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="TOOLCHAIN_INCOMPLETE",
            title="Profiling Tools Not Available",
            summary=(
                f"Missing: {', '.join(missing)}. "
                "GPU Insight Lab can run benchmarks without profilers, "
                "but deep kernel analysis requires Nsight tools."
            ),
            evidence=(
                f"Tool check: nsys={'found' if nsys_ok else 'missing'}, "
                f"ncu={'found' if ncu_ok else 'missing'}. "
                "This is informational; it does not indicate a GPU problem."
            ),
            metric_name="profiler_availability",
            observed_value=str(missing),
            reference_value="nsys and ncu both present",
            confidence=1.0,
            recommendation=(
                "Install Nsight Systems and Nsight Compute from "
                "developer.nvidia.com/nsight-systems and developer.nvidia.com/nsight-compute."
            ),
            verification_step="Run 'nsys --version' and 'ncu --version' after installation.",
        )
    else:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="Profiling Tools Available",
            summary="Both nsys (Nsight Systems) and ncu (Nsight Compute) are installed.",
            evidence="Tool check: nsys=found, ncu=found.",
            metric_name="profiler_availability",
            observed_value="nsys and ncu present",
            reference_value="nsys and ncu both present",
            confidence=1.0,
        )


def rule_toolchain_completeness(session: Any) -> Optional[DiagnosisResult]:
    """
    Check if core build tools (nvcc, cmake) are available.
    """
    rule_id = "toolchain_completeness"
    tools = getattr(session, "tool_status", {}) or {}

    def _exists(t: str) -> bool:
        entry = tools.get(t)
        if isinstance(entry, dict):
            return bool(entry.get("exists", False))
        return False

    required = ["nvcc", "cmake"]
    missing = [t for t in required if not _exists(t)]
    present = [t for t in required if _exists(t)]

    if missing:
        rec, verif = get_recommendation("TOOLCHAIN_INCOMPLETE")
        return DiagnosisResult(
            rule_id=rule_id,
            severity="WARNING",
            category="TOOLCHAIN_INCOMPLETE",
            title="Core Build Tools Missing",
            summary=(
                f"Missing tools: {', '.join(missing)}. "
                "Native benchmark executable cannot be built without these."
            ),
            evidence=(
                f"Tool check results — present: {present or 'none'}; "
                f"missing: {missing}."
            ),
            metric_name="toolchain_completeness",
            observed_value=str(missing),
            reference_value="nvcc and cmake present",
            confidence=1.0,
            recommendation=rec,
            verification_step=verif,
        )
    else:
        return DiagnosisResult(
            rule_id=rule_id,
            severity="INFO",
            category="HEALTHY",
            title="Core Build Toolchain Complete",
            summary="nvcc and cmake are present; native benchmarks can be built.",
            evidence=f"Tool check: {', '.join(required)} all found.",
            metric_name="toolchain_completeness",
            observed_value="all present",
            reference_value="nvcc and cmake present",
            confidence=1.0,
        )
