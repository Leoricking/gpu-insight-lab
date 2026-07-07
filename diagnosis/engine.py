"""
GPU Insight Lab - Diagnosis Engine
Evidence-based rule engine. Runs all rules against a BenchmarkSession.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    rule_id: str = ""
    severity: str = "INFO"          # INFO, WARNING, ERROR, CRITICAL
    category: str = "LOW_CONFIDENCE"  # from CATEGORIES below
    title: str = ""
    summary: str = ""
    evidence: str = ""
    metric_name: str = ""
    observed_value: Any = None       # float | str
    reference_value: Any = None      # float | str
    confidence: float = 0.0          # 0.0-1.0
    recommendation: str = ""
    verification_step: str = ""

    # Extended fields for interview demo readiness
    # These are populated by the scoring engine at the session level,
    # but included here for schema completeness so report consumers
    # can always reference them without key errors.
    gpu_insight_score: Optional[float] = None      # 0-100 composite score
    health_score: Optional[float] = None           # alias for gpu_insight_score
    bottleneck_classification: str = ""            # e.g. PCIE_TRANSFER_BOUND
    missing_data: List[str] = field(default_factory=list)
    deductions: List[Dict[str, Any]] = field(default_factory=list)
    positive_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)


CATEGORIES = {
    "HEALTHY",
    "CPU_BOUND",
    "GPU_COMPUTE_BOUND",
    "GPU_MEMORY_BOUND",
    "PCIE_TRANSFER_BOUND",
    "VRAM_CAPACITY_BOUND",
    "KERNEL_LAUNCH_OVERHEAD",
    "SYNCHRONIZATION_BOUND",
    "DISK_IO_BOUND",
    "THERMAL_THROTTLING",
    "POWER_LIMITED",
    "DRIVER_RUNTIME_MISMATCH",
    "TOOLCHAIN_INCOMPLETE",
    "LOW_CONFIDENCE",
    "INSUFFICIENT_DATA",
}


def run_diagnosis(session: Any) -> List[Dict[str, Any]]:
    """
    Run all diagnosis rules against a BenchmarkSession.
    Returns list of DiagnosisResult as dicts.
    Never raises.
    """
    from diagnosis import rules  # noqa: PLC0415

    results: List[DiagnosisResult] = []

    rule_funcs = [
        rules.rule_pinned_vs_pageable,
        rules.rule_transfer_overhead,
        rules.rule_launch_overhead,
        rules.rule_optimization_check,
        rules.rule_thermal_throttle,
        rules.rule_vram_pressure,
        rules.rule_driver_mismatch,
        rules.rule_profiler_unavailable,
        rules.rule_toolchain_completeness,
    ]

    for rule_fn in rule_funcs:
        try:
            findings = rule_fn(session)
            if findings is None:
                continue
            if isinstance(findings, DiagnosisResult):
                results.append(findings)
            elif isinstance(findings, list):
                results.extend(f for f in findings if isinstance(f, DiagnosisResult))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diagnosis rule %s failed: %s", rule_fn.__name__, exc)
            results.append(DiagnosisResult(
                rule_id=rule_fn.__name__,
                severity="INFO",
                category="LOW_CONFIDENCE",
                title=f"Rule {rule_fn.__name__} could not be evaluated",
                summary=str(exc),
                evidence=f"Rule raised exception: {exc}",
                confidence=0.0,
            ))

    return [dataclasses.asdict(r) for r in results]
