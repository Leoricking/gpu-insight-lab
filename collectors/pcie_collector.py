"""
GPU Insight Lab - PCIe Link Collector
Queries PCIe generation and width via nvidia-smi.
Marks unknown if unavailable. Never guesses.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PCIeInfo:
    available: bool = False
    pcie_gen_current: Optional[int] = None
    pcie_width_current: Optional[int] = None
    pcie_gen_max: Optional[int] = None
    pcie_width_max: Optional[int] = None
    bandwidth_gbps_theoretical: Optional[float] = None
    bandwidth_gbps_current: Optional[float] = None
    is_bottlenecked: Optional[bool] = None
    error: Optional[str] = None


# Theoretical PCIe bandwidth per lane per direction (GB/s)
_PCIE_LANE_BW: dict[int, float] = {
    1: 0.25,   # PCIe Gen 1: 2.5 GT/s, ~0.25 GB/s/lane
    2: 0.5,    # PCIe Gen 2: 5 GT/s
    3: 0.985,  # PCIe Gen 3: 8 GT/s, ~0.985 GB/s/lane
    4: 1.97,   # PCIe Gen 4: 16 GT/s
    5: 3.94,   # PCIe Gen 5: 32 GT/s
}


def _compute_bandwidth(gen: int, width: int) -> Optional[float]:
    """Return theoretical one-directional bandwidth in GB/s."""
    lane_bw = _PCIE_LANE_BW.get(gen)
    if lane_bw is None:
        return None
    return round(lane_bw * width, 2)


def collect() -> PCIeInfo:
    """Collect PCIe link info from nvidia-smi. Never raises."""
    info = PCIeInfo()

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.current,pcie.link.width.current,"
                "pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        info.error = f"nvidia-smi unavailable: {exc}"
        logger.debug("%s", info.error)
        return info

    if result.returncode != 0:
        info.error = f"nvidia-smi error: {result.stderr.strip()}"
        logger.debug("%s", info.error)
        return info

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        info.error = "nvidia-smi returned no data"
        return info

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        info.error = f"Unexpected nvidia-smi output: {line}"
        return info

    def _i(s: str) -> Optional[int]:
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    info.available = True
    info.pcie_gen_current = _i(parts[0])
    info.pcie_width_current = _i(parts[1])
    info.pcie_gen_max = _i(parts[2])
    info.pcie_width_max = _i(parts[3])

    # Compute theoretical bandwidths
    if info.pcie_gen_max is not None and info.pcie_width_max is not None:
        info.bandwidth_gbps_theoretical = _compute_bandwidth(info.pcie_gen_max, info.pcie_width_max)
    if info.pcie_gen_current is not None and info.pcie_width_current is not None:
        info.bandwidth_gbps_current = _compute_bandwidth(info.pcie_gen_current, info.pcie_width_current)

    # Flag bottleneck: operating below max capability
    if (info.pcie_gen_current is not None and info.pcie_gen_max is not None
            and info.pcie_width_current is not None and info.pcie_width_max is not None):
        info.is_bottlenecked = (
            info.pcie_gen_current < info.pcie_gen_max
            or info.pcie_width_current < info.pcie_width_max
        )

    return info
