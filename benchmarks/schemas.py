"""
GPU Insight Lab - Benchmark Data Schemas
Dataclasses for benchmark results and sessions.
"""

from __future__ import annotations

import time
import platform
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


SCHEMA_VERSION = "1.0"


@dataclass
class BenchmarkResult:
    """Complete record of a single benchmark test execution."""

    # Schema and identity
    schema_version: str = SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)
    hostname: str = field(default_factory=platform.node)
    os: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")

    # GPU info
    gpu_name: str = ""
    gpu_uuid: str = ""
    driver_version: str = ""
    cuda_runtime_version: str = ""
    cuda_driver_version: str = ""
    compute_capability: str = ""

    # Test configuration
    test_name: str = ""
    data_type: str = "float32"
    input_size: int = 0
    block_size: int = 0
    grid_size: int = 0
    warmup_runs: int = 0
    measured_runs: int = 0

    # Timing results (all in milliseconds)
    cpu_time_ms: Optional[float] = None
    gpu_time_ms: Optional[float] = None
    transfer_time_ms: Optional[float] = None
    end_to_end_time_ms: Optional[float] = None

    # Performance metrics
    throughput: Optional[float] = None          # GFLOPS or elements/sec depending on test
    bandwidth_gbps: Optional[float] = None
    speedup: Optional[float] = None             # GPU vs CPU speedup factor

    # Correctness
    correctness_pass: Optional[bool] = None
    max_error: Optional[float] = None

    # Statistics (all from measured_runs timing)
    mean: Optional[float] = None
    median: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    standard_deviation: Optional[float] = None
    raw_measurements: List[float] = field(default_factory=list)

    # Metadata
    notes: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                d[k] = v
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Reconstruct from dict (e.g., parsed JSON)."""
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def is_valid(self) -> bool:
        """Return True if the result has at least a test name and no fatal error."""
        return bool(self.test_name) and self.error is None


@dataclass
class BenchmarkSession:
    """Container for a complete benchmark run session."""

    session_id: Optional[str] = None
    session_name: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # System snapshot
    system_info: Dict[str, Any] = field(default_factory=dict)
    nvidia_info: Dict[str, Any] = field(default_factory=dict)
    cuda_info: Dict[str, Any] = field(default_factory=dict)
    pcie_info: Dict[str, Any] = field(default_factory=dict)
    tool_status: Dict[str, Any] = field(default_factory=dict)
    amd_info: Dict[str, Any] = field(default_factory=dict)

    # Results
    results: List[BenchmarkResult] = field(default_factory=list)
    diagnosis_results: List[Dict[str, Any]] = field(default_factory=list)

    # Score
    health_score: Optional[float] = None
    score_confidence: Optional[float] = None
    score_details: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None

    def duration_seconds(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return round(self.completed_at - self.started_at, 2)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "results":
                d[k] = [r.to_dict() for r in v]
            else:
                d[k] = v
        return d
