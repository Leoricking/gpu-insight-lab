"""
GPU Insight Lab - Evidence helpers
Utilities for building evidence strings for diagnosis results.
"""

from __future__ import annotations

from typing import Any, Optional


def format_evidence(
    metric: str,
    observed: Any,
    reference: Any,
    unit: str = "",
    context: str = "",
) -> str:
    """Build a readable evidence string."""
    obs_str = f"{observed}{unit}" if unit else str(observed)
    ref_str = f"{reference}{unit}" if unit else str(reference)
    base = f"{metric}: observed={obs_str}, reference={ref_str}"
    if context:
        base += f". {context}"
    return base


def format_ratio_evidence(
    metric: str,
    observed: float,
    reference: float,
    ratio: float,
    unit: str = "",
    context: str = "",
) -> str:
    """Evidence string for ratio-based comparisons."""
    obs_str = f"{observed:.3f}{unit}"
    ref_str = f"{reference:.3f}{unit}"
    base = f"{metric}: observed={obs_str}, reference={ref_str}, ratio={ratio:.2f}x"
    if context:
        base += f". {context}"
    return base


def format_threshold_evidence(
    metric: str,
    observed: float,
    threshold: float,
    unit: str = "",
    above: bool = True,
    context: str = "",
) -> str:
    """Evidence string for threshold violations."""
    direction = "above" if above else "below"
    obs_str = f"{observed:.3f}{unit}"
    thr_str = f"{threshold:.3f}{unit}"
    base = f"{metric}: {obs_str} is {direction} threshold {thr_str}"
    if context:
        base += f". {context}"
    return base


def format_missing_evidence(metric: str, reason: str = "not measured") -> str:
    """Evidence string when data is missing."""
    return f"{metric}: data {reason}; cannot evaluate this rule"


def format_version_mismatch_evidence(
    component: str,
    driver_ver: str,
    runtime_ver: str,
) -> str:
    return (
        f"{component} version mismatch: driver={driver_ver}, runtime={runtime_ver}. "
        "Driver and runtime versions should match to avoid undefined behavior."
    )
