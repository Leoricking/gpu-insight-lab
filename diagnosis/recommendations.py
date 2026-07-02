"""
GPU Insight Lab - Recommendation templates.
"""

from __future__ import annotations

# Keyed by category
RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "PCIE_TRANSFER_BOUND": {
        "recommendation": (
            "Transfer overhead exceeds 40% of end-to-end time. "
            "Consider: (1) use pinned host memory to improve H2D/D2H throughput, "
            "(2) overlap transfers with computation using CUDA streams, "
            "(3) increase computation size to amortize transfer cost, "
            "(4) batch multiple small transfers into fewer large ones."
        ),
        "verification": (
            "Re-run memory_bandwidth test comparing pageable vs pinned transfers. "
            "Use nsys timeline to visualize overlap between compute and copy engines."
        ),
    },
    "KERNEL_LAUNCH_OVERHEAD": {
        "recommendation": (
            "GPU time exceeds CPU time for this input size, indicating kernel launch overhead dominates. "
            "Consider: (1) increase problem size to amortize launch cost, "
            "(2) batch multiple small kernels into one, "
            "(3) use CUDA graphs to reduce per-launch overhead for repeated patterns."
        ),
        "verification": (
            "Profile with ncu --metrics launch__grid_size to quantify grid efficiency. "
            "Compare CPU and GPU times at 10x, 100x problem sizes."
        ),
    },
    "THERMAL_THROTTLING": {
        "recommendation": (
            "GPU temperature above 85°C with clock below base frequency. "
            "Check: (1) case airflow and cooling, "
            "(2) verify thermal paste is adequate, "
            "(3) review power limit settings, "
            "(4) consider enabling persistence mode to stabilize clocks."
        ),
        "verification": (
            "Monitor temperature with nvidia-smi dmon -s p during benchmark. "
            "Compare actual vs boost vs base clocks."
        ),
    },
    "VRAM_CAPACITY_BOUND": {
        "recommendation": (
            "VRAM utilization above 90%. "
            "Consider: (1) reduce batch size, "
            "(2) use fp16 or bf16 to reduce memory footprint, "
            "(3) implement activation checkpointing for training workloads, "
            "(4) evaluate multi-GPU or GPU with larger VRAM."
        ),
        "verification": (
            "Monitor VRAM with nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1"
        ),
    },
    "DRIVER_RUNTIME_MISMATCH": {
        "recommendation": (
            "CUDA driver and runtime versions do not match. "
            "This can cause undefined behavior, crashes, or silent incorrect results. "
            "Install matching CUDA Toolkit and driver from developer.nvidia.com/cuda-downloads."
        ),
        "verification": (
            "Run 'nvcc --version' (runtime) and 'nvidia-smi' (driver version). "
            "They should match the same CUDA toolkit version."
        ),
    },
    "GPU_COMPUTE_BOUND": {
        "recommendation": (
            "Kernel is compute-bound. "
            "To optimize: (1) reduce arithmetic intensity via algorithm change, "
            "(2) use Tensor Cores if applicable (fp16/bf16 GEMM), "
            "(3) profile with ncu to find instruction bottlenecks."
        ),
        "verification": "Profile with ncu --set full to identify compute bottleneck.",
    },
    "GPU_MEMORY_BOUND": {
        "recommendation": (
            "Kernel is memory-bandwidth bound. "
            "To optimize: (1) improve data reuse with shared memory tiling, "
            "(2) ensure coalesced memory access patterns, "
            "(3) use __ldg() for read-only data, "
            "(4) consider compression or mixed precision."
        ),
        "verification": "Profile with ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global to verify.",
    },
    "TOOLCHAIN_INCOMPLETE": {
        "recommendation": (
            "Required tools are missing. "
            "Install CUDA Toolkit from developer.nvidia.com/cuda-downloads. "
            "For profiling: install Nsight Systems and Nsight Compute."
        ),
        "verification": "Run 'nvcc --version', 'nsys --version', 'ncu --version' after installation.",
    },
    "HEALTHY": {
        "recommendation": "No action required. Continue monitoring during production workloads.",
        "verification": "Re-run benchmarks after driver updates or hardware changes.",
    },
    "LOW_CONFIDENCE": {
        "recommendation": (
            "Insufficient data to make a reliable diagnosis. "
            "Run the full benchmark suite to enable all diagnostic rules."
        ),
        "verification": "Run full-test and check for errors in all benchmark stages.",
    },
    "INSUFFICIENT_DATA": {
        "recommendation": "Run the full benchmark suite to collect required measurements.",
        "verification": "Ensure native benchmark executable is built and accessible in bin/.",
    },
}


def get_recommendation(category: str) -> tuple[str, str]:
    """Return (recommendation, verification_step) for a category."""
    entry = RECOMMENDATIONS.get(category, RECOMMENDATIONS["LOW_CONFIDENCE"])
    return entry["recommendation"], entry.get("verification", "")
