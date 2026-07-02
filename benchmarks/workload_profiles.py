"""
GPU Insight Lab - Workload Profiles
Defines named workload configurations for quick/full benchmark sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkloadConfig:
    """Configuration for a single workload/benchmark test."""
    name: str
    display_name: str
    category: str          # e.g. "memory", "compute", "transfer", "image"
    description: str
    default_size: int
    default_block_size: int
    default_repeat: int
    default_warmup: int
    include_in_quick: bool
    include_in_full: bool
    requires_native: bool
    timeout_seconds: int = 60
    extra_params: Dict[str, Any] = field(default_factory=dict)


WORKLOAD_PROFILES: List[WorkloadConfig] = [
    WorkloadConfig(
        name="device_info",
        display_name="Device Info",
        category="info",
        description="Collect CUDA device properties",
        default_size=0,
        default_block_size=0,
        default_repeat=1,
        default_warmup=0,
        include_in_quick=True,
        include_in_full=True,
        requires_native=True,
        timeout_seconds=15,
    ),
    WorkloadConfig(
        name="vector_add",
        display_name="Vector Add",
        category="compute",
        description="Element-wise vector addition, naive + grid-stride",
        default_size=16_777_216,
        default_block_size=256,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=True,
        include_in_full=True,
        requires_native=True,
    ),
    WorkloadConfig(
        name="reduction",
        display_name="Parallel Reduction",
        category="compute",
        description="Sum reduction: naive, shared-memory, atomicAdd",
        default_size=16_777_216,
        default_block_size=256,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=True,
        include_in_full=True,
        requires_native=True,
    ),
    WorkloadConfig(
        name="transpose",
        display_name="Matrix Transpose",
        category="memory",
        description="Naive vs tiled transpose, effective bandwidth",
        default_size=4096,
        default_block_size=32,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=False,
        include_in_full=True,
        requires_native=True,
    ),
    WorkloadConfig(
        name="gemm_naive",
        display_name="GEMM Naive",
        category="compute",
        description="Naive global-memory GEMM",
        default_size=512,
        default_block_size=16,
        default_repeat=5,
        default_warmup=2,
        include_in_quick=False,
        include_in_full=True,
        requires_native=True,
        timeout_seconds=120,
    ),
    WorkloadConfig(
        name="gemm_tiled",
        display_name="GEMM Tiled",
        category="compute",
        description="Shared-memory tiled GEMM",
        default_size=512,
        default_block_size=16,
        default_repeat=5,
        default_warmup=2,
        include_in_quick=False,
        include_in_full=True,
        requires_native=True,
        timeout_seconds=120,
    ),
    WorkloadConfig(
        name="memory_bandwidth",
        display_name="Memory Bandwidth",
        category="transfer",
        description="H2D/D2H pageable and pinned, D2D bandwidth",
        default_size=67_108_864,  # 64 MB in bytes
        default_block_size=256,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=True,
        include_in_full=True,
        requires_native=True,
    ),
    WorkloadConfig(
        name="stream_pipeline",
        display_name="CUDA Stream Pipeline",
        category="compute",
        description="Sync vs 2-stream vs 4-stream async pipeline",
        default_size=16_777_216,
        default_block_size=256,
        default_repeat=5,
        default_warmup=2,
        include_in_quick=False,
        include_in_full=True,
        requires_native=True,
    ),
    WorkloadConfig(
        name="image_grayscale",
        display_name="Image Grayscale",
        category="image",
        description="CPU vs CUDA RGB-to-grayscale, batch correctness",
        default_size=1920 * 1080,
        default_block_size=256,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=False,
        include_in_full=True,
        requires_native=True,
    ),
    WorkloadConfig(
        name="cpu_vector_add",
        display_name="CPU Vector Add (baseline)",
        category="cpu_baseline",
        description="Pure NumPy CPU vector addition baseline",
        default_size=1_000_000,
        default_block_size=0,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=True,
        include_in_full=True,
        requires_native=False,
    ),
    WorkloadConfig(
        name="cpu_matrix_multiply",
        display_name="CPU Matrix Multiply (baseline)",
        category="cpu_baseline",
        description="Pure NumPy CPU GEMM baseline",
        default_size=512,
        default_block_size=0,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=False,
        include_in_full=True,
        requires_native=False,
    ),
    WorkloadConfig(
        name="cpu_image_grayscale",
        display_name="CPU Grayscale (baseline)",
        category="cpu_baseline",
        description="Pure NumPy CPU grayscale baseline",
        default_size=1920 * 1080,
        default_block_size=0,
        default_repeat=10,
        default_warmup=3,
        include_in_quick=False,
        include_in_full=True,
        requires_native=False,
    ),
]

# Lookup by name
_PROFILE_MAP: Dict[str, WorkloadConfig] = {w.name: w for w in WORKLOAD_PROFILES}

# Type alias
Dict = dict  # re-export for type hints in this module


def get_profile(name: str) -> Optional[WorkloadConfig]:
    return _PROFILE_MAP.get(name)


def get_quick_profiles() -> List[WorkloadConfig]:
    return [w for w in WORKLOAD_PROFILES if w.include_in_quick]


def get_full_profiles() -> List[WorkloadConfig]:
    return [w for w in WORKLOAD_PROFILES if w.include_in_full]
