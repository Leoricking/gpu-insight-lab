"""
GPU Insight Lab - Toolchain Availability Checker
Checks existence and version of: nvcc, cmake, cl.exe, nvidia-smi,
nsys, ncu, rocminfo, rocm-smi, hipcc.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class ToolStatus:
    exists: bool = False
    version: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None


def _probe_tool(
    cmd: str,
    version_args: list[str],
    version_pattern: Optional[str] = None,
    timeout: int = 8,
) -> ToolStatus:
    """Run cmd with version_args, parse version, return ToolStatus."""
    import re  # noqa: PLC0415

    # Find path first
    which_cmd = ["where", cmd] if sys.platform == "win32" else ["which", cmd]
    try:
        which_result = subprocess.run(
            which_cmd, capture_output=True, text=True, timeout=5, encoding="utf-8"
        )
        if which_result.returncode != 0:
            return ToolStatus(exists=False, error="not in PATH")
        tool_path = which_result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return ToolStatus(exists=False, error=str(exc))

    # Run version command
    try:
        result = subprocess.run(
            [cmd] + version_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )
        output = (result.stdout + result.stderr).strip()
        version: Optional[str] = None
        if version_pattern:
            m = re.search(version_pattern, output)
            version = m.group(1) if m else output.splitlines()[0][:80] if output else None
        else:
            version = output.splitlines()[0][:80] if output else None
        return ToolStatus(exists=True, version=version, path=tool_path)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return ToolStatus(exists=True, path=tool_path, error=str(exc))


# Tool definitions: {tool_name: (version_args, version_regex)}
_TOOLS: dict[str, tuple[list[str], Optional[str]]] = {
    "nvcc": (["--version"], r"release\s+([\d.]+)"),
    "cmake": (["--version"], r"cmake version\s+([\d.]+)"),
    "cl": (["/help"], r"Version\s+([\d.]+)"),
    "nvidia-smi": (["--version"], r"NVIDIA-SMI\s+([\d.]+)"),
    "nsys": (["--version"], r"(\d+\.\d+[\.\d]*)"),
    "ncu": (["--version"], r"(\d+\.\d+[\.\d]*)"),
    "rocminfo": (["--version"], r"(\d+[\.\d]*)"),
    "rocm-smi": (["--version"], r"(\d+[\.\d]*)"),
    "hipcc": (["--version"], r"(\d+[\.\d]*)"),
}


def collect() -> Dict[str, ToolStatus]:
    """Check all tools and return {tool_name: ToolStatus}. Never raises."""
    results: Dict[str, ToolStatus] = {}
    for tool_name, (ver_args, ver_pattern) in _TOOLS.items():
        try:
            results[tool_name] = _probe_tool(tool_name, ver_args, ver_pattern)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tool probe error for %s: %s", tool_name, exc)
            results[tool_name] = ToolStatus(exists=False, error=str(exc))
    return results


def get_tool_summary(tools: Dict[str, ToolStatus]) -> dict[str, bool]:
    """Return simplified {tool: available} dict."""
    return {k: v.exists for k, v in tools.items()}
