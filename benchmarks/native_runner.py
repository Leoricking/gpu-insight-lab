"""
GPU Insight Lab - Native Benchmark Runner
Manages subprocess execution of gpu_insight_benchmark executable.
Never crashes if executable is missing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120


def find_executable() -> Optional[Path]:
    """Search for gpu_insight_benchmark in bin/ and PATH. Returns None if not found."""
    from app.branding import NATIVE_EXECUTABLE  # noqa: PLC0415

    exe_name = NATIVE_EXECUTABLE + (".exe" if sys.platform == "win32" else "")
    root = Path(__file__).parent.parent

    candidates = [
        root / "bin" / exe_name,
        root / "build" / exe_name,
        root / "build" / "Release" / exe_name,
        root / "build" / "Debug" / exe_name,
    ]
    for c in candidates:
        if c.exists():
            logger.debug("Found native executable at %s", c)
            return c

    # Check PATH
    try:
        which_cmd = ["where", exe_name] if sys.platform == "win32" else ["which", exe_name]
        result = subprocess.run(
            which_cmd, capture_output=True, text=True, timeout=5, encoding="utf-8"
        )
        if result.returncode == 0:
            path = Path(result.stdout.strip().splitlines()[0])
            if path.exists():
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _run_executable(
    args: List[str],
    timeout: int = _DEFAULT_TIMEOUT,
    output_file: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the native executable with given args.
    Returns parsed JSON dict, or None on failure.
    """
    exe = find_executable()
    if exe is None:
        logger.info("Native benchmark executable not found; skipping native test")
        return None

    cmd = [str(exe)] + args
    if output_file:
        cmd += ["--output", str(output_file)]

    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        logger.warning("Native benchmark timed out after %ds", timeout)
        return {"error": f"timeout after {timeout}s"}
    except FileNotFoundError as exc:
        logger.warning("Native executable not runnable: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "Native benchmark exited with %d: %s",
            result.returncode,
            result.stderr.strip()[:200],
        )

    # Try stdout JSON first
    stdout = result.stdout.strip()
    if stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

    # Try output file
    if output_file and output_file.exists():
        try:
            return json.loads(output_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to parse output file: %s", exc)

    # Return stderr as error info
    if result.returncode != 0 and result.stderr:
        return {"error": result.stderr.strip()[:500], "returncode": result.returncode}

    return None


def run_device_info() -> Optional[Dict[str, Any]]:
    """Run --device-info and return parsed dict."""
    return _run_executable(["--device-info"], timeout=30)


def run_test(
    test_name: str,
    repeat: int = 10,
    warmup: int = 3,
    block_size: Optional[int] = None,
    data_size: Optional[int] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Run a named test. Returns parsed JSON or None."""
    args = ["--test", test_name, "--repeat", str(repeat), "--warmup", str(warmup)]
    if block_size is not None:
        args += ["--block-size", str(block_size)]
    if data_size is not None:
        args += ["--size", str(data_size)]
    return _run_executable(args, timeout=timeout)


def run_quick() -> List[Dict[str, Any]]:
    """Run --quick and return list of result dicts."""
    data = _run_executable(["--quick"], timeout=120)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, dict) and "error" not in data:
        return [data]
    return []


def run_full() -> List[Dict[str, Any]]:
    """Run --full and return list of result dicts."""
    data = _run_executable(["--full"], timeout=300)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, dict) and "error" not in data:
        return [data]
    return []


def is_available() -> bool:
    """Return True if the native executable exists and is executable."""
    exe = find_executable()
    return exe is not None
