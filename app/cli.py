"""
GPU Insight Lab - Command Line Interface
Cross-Vendor GPU Performance, Validation and Workload Diagnostics

Exit codes: 0=success, 1=error, 2=partial/warning
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from app.branding import APP_NAME, APP_SUBTITLE, APP_VERSION

_BANNER = f"""
{APP_NAME} CLI
{APP_SUBTITLE}
Version: {APP_VERSION}
"""

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _print_table(rows: list[dict], columns: list[str]) -> None:
    """Simple ASCII table printer."""
    col_widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in columns}
    header = "  ".join(c.ljust(col_widths[c]) for c in columns)
    sep = "  ".join("-" * col_widths[c] for c in columns)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in columns))


# ────────────────────────────────────────────────────────────────────────────
# Command handlers
# ────────────────────────────────────────────────────────────────────────────

def cmd_system_info(args: argparse.Namespace) -> int:
    """Collect and display system information."""
    import dataclasses as dc  # noqa: PLC0415
    from collectors import system_collector, nvidia_collector  # noqa: PLC0415
    from collectors import cuda_collector, pcie_collector, tool_collector, amd_collector  # noqa: PLC0415

    logger.info("Collecting system information...")
    sys_info = system_collector.collect()
    nv_info = nvidia_collector.collect()
    cuda_info = cuda_collector.collect()
    pcie_info = pcie_collector.collect()
    tools = tool_collector.collect()
    amd_info = amd_collector.collect()

    data: Dict[str, Any] = {
        "system": dc.asdict(sys_info),
        "nvidia": dc.asdict(nv_info),
        "cuda": dc.asdict(cuda_info),
        "pcie": dc.asdict(pcie_info),
        "tools": {k: dc.asdict(v) for k, v in tools.items()},
        "amd": dc.asdict(amd_info),
    }

    if args.json:
        _print_json(data)
        return 0

    s = data["system"]
    nv = data["nvidia"]
    print(_BANNER)
    print("=== System ===")
    print(f"  Hostname:         {s.get('hostname', 'N/A')}")
    print(f"  OS:               {s.get('os_name', '')} {s.get('os_release', '')}")
    print(f"  CPU:              {s.get('cpu_model', 'N/A')}")
    print(f"  CPU Cores:        {s.get('cpu_physical_cores', '?')} physical / {s.get('cpu_logical_cores', '?')} logical")
    print(f"  RAM:              {s.get('ram_total_gb', 0):.1f} GB total, {s.get('ram_available_gb', 0):.1f} GB available")
    print(f"  Python:           {s.get('python_version', 'N/A')}")
    print()

    print("=== GPU ===")
    if nv.get("available"):
        print(f"  Name:             {nv.get('gpu_name', 'N/A')}")
        print(f"  Driver:           {nv.get('driver_version', 'N/A')}")
        print(f"  CUDA Driver:      {nv.get('cuda_driver_version', 'N/A')}")
        print(f"  Compute Cap:      {nv.get('compute_capability', 'N/A')}")
        print(f"  VRAM:             {nv.get('vram_total_mb', 0):.0f} MB total, {nv.get('vram_used_mb', 0):.0f} MB used")
        print(f"  Temperature:      {nv.get('temperature_c', 'N/A')} °C")
        print(f"  GPU Utilization:  {nv.get('gpu_utilization_pct', 'N/A')} %")
        print(f"  Power:            {nv.get('power_draw_w', 'N/A')} W / {nv.get('power_limit_w', 'N/A')} W")
        print(f"  Perf State:       {nv.get('performance_state', 'N/A')}")
        p = data["pcie"]
        if p.get("available"):
            print(f"  PCIe Link:        Gen{p.get('pcie_gen_current')} x{p.get('pcie_width_current')} (max Gen{p.get('pcie_gen_max')} x{p.get('pcie_width_max')})")
    else:
        print("  No NVIDIA GPU detected.")
    print()

    print("=== CUDA Toolchain ===")
    c = data["cuda"]
    print(f"  nvcc:             {'Yes (' + str(c.get('nvcc_version','?')) + ')' if c.get('nvcc_available') else 'Not found'}")
    print(f"  CUDA Runtime:     {'Yes (' + str(c.get('cuda_runtime_version','?')) + ')' if c.get('cuda_runtime_available') else 'Not detected'}")
    print(f"  Native Benchmark: {'Yes — ' + str(c.get('native_benchmark_path','')) if c.get('native_benchmark_available') else 'Not built'}")
    print()

    print("=== Tools ===")
    for tool_name, status_d in data["tools"].items():
        avail = "Yes" if status_d.get("exists") else "No"
        ver = status_d.get("version") or ""
        print(f"  {tool_name:<16} {avail:<5} {ver}")
    print()

    amd = data["amd"]
    if amd.get("available") or amd.get("rocm_available"):
        print("=== AMD/ROCm ===")
        print(f"  Status:           {amd.get('validation_status', 'NOT_VALIDATED')}")
        print(f"  ROCm Available:   {amd.get('rocm_available', False)}")
        print(f"  ROCm Version:     {amd.get('rocm_version', 'N/A')}")
        print()

    return 0


def cmd_quick_test(args: argparse.Namespace) -> int:
    """Run quick benchmark suite."""
    print(_BANNER)
    print("[GPU Insight Lab] Starting quick test...")

    from benchmarks.runner import run_quick_test  # noqa: PLC0415

    def _progress(pct: int, msg: str) -> None:
        if not args.json:
            print(f"  [{pct:3d}%] {msg}")

    session = run_quick_test(progress_callback=_progress)

    if args.json:
        _print_json(session.to_dict())
        return 0 if session.status == "completed" else 1

    print()
    score = session.health_score
    print(f"=== GPU Insight Score: {score:.1f}/100 ===" if score else "=== Score: N/A ===")
    print()

    print(f"  Status:   {session.status}")
    print(f"  Duration: {session.duration_seconds():.1f}s" if session.duration_seconds() else "  Duration: N/A")
    print(f"  Tests:    {len(session.results)}")
    print()

    if session.results:
        print("=== Benchmark Results ===")
        for r in session.results:
            name = getattr(r, "test_name", "")
            mean = getattr(r, "mean", None)
            bw = getattr(r, "bandwidth_gbps", None)
            correct = getattr(r, "correctness_pass", None)
            line = f"  {name:<30}"
            if mean is not None:
                line += f"  mean={mean:.3f}ms"
            if bw is not None:
                line += f"  bw={bw:.2f}GB/s"
            if correct is not None:
                line += f"  correct={'Yes' if correct else 'No'}"
            print(line)
        print()

    if session.diagnosis_results:
        print("=== Diagnosis ===")
        for d in session.diagnosis_results:
            sev = d.get("severity", "INFO")
            title = d.get("title", "")
            cat = d.get("category", "")
            print(f"  [{sev}] {title} ({cat})")
        print()

    # Save session
    if not args.no_save:
        try:
            from storage.database import get_database  # noqa: PLC0415
            db = get_database()
            session_id = db.save_session(session)
            if session_id:
                print(f"  Session saved with ID: {session_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not save session: {exc}", file=sys.stderr)

    return 0 if session.status == "completed" else 2


def cmd_full_test(args: argparse.Namespace) -> int:
    """Run full benchmark suite."""
    print(_BANNER)
    print("[GPU Insight Lab] Starting full test...")

    from benchmarks.runner import run_full_test  # noqa: PLC0415

    def _progress(pct: int, msg: str) -> None:
        if not args.json:
            print(f"  [{pct:3d}%] {msg}")

    session = run_full_test(progress_callback=_progress)

    if args.json:
        _print_json(session.to_dict())
        return 0 if session.status == "completed" else 1

    print()
    score = session.health_score
    print(f"=== GPU Insight Score: {score:.1f}/100 ===" if score else "=== Score: N/A ===")
    print(f"  Tests: {len(session.results)}  Duration: {session.duration_seconds():.1f}s")
    print()

    if not args.no_save:
        try:
            from storage.database import get_database  # noqa: PLC0415
            db = get_database()
            session_id = db.save_session(session)
            if session_id:
                print(f"  Session saved with ID: {session_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not save session: {exc}", file=sys.stderr)

    return 0 if session.status == "completed" else 2


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run a specific benchmark test."""
    from benchmarks.runner import run_single_test  # noqa: PLC0415

    # Normalize hyphen-to-underscore aliases for convenience
    _TEST_ALIASES: Dict[str, str] = {
        "vector-add": "vector_add",
        "memory-bandwidth": "memory_bandwidth",
        "memory": "memory_bandwidth",
        "pcie": "memory_bandwidth",
        "streams": "stream_pipeline",
        "stream-pipeline": "stream_pipeline",
        "gemm": "gemm_naive",
        "image-grayscale": "image_grayscale",
        "prefix-sum": "prefix_sum",
        "convolution-2d": "convolution_2d",
    }
    test_name = _TEST_ALIASES.get(args.test, args.test)

    print(f"[GPU Insight Lab] Running benchmark: {test_name}")
    result = run_single_test(test_name)

    if args.json:
        d = result.to_dict() if hasattr(result, "to_dict") else {}
        _print_json(d)
    else:
        print(f"  Test:      {getattr(result, 'test_name', args.test)}")
        print(f"  Mean:      {getattr(result, 'mean', 'N/A')} ms")
        print(f"  Bandwidth: {getattr(result, 'bandwidth_gbps', 'N/A')} GB/s")
        print(f"  Correct:   {getattr(result, 'correctness_pass', 'N/A')}")
        if getattr(result, "error", None):
            print(f"  Error:     {result.error}")
            return 1

    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """List stored benchmark sessions."""
    try:
        from storage.database import get_database  # noqa: PLC0415
        db = get_database()
        sessions = db.get_sessions()
    except Exception as exc:  # noqa: BLE001
        print(f"Error accessing database: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _print_json(sessions)
        return 0

    if not sessions:
        print("No sessions found.")
        return 0

    print(f"Found {len(sessions)} session(s):\n")
    _print_table(
        sessions,
        ["id", "session_name", "started_at", "gpu_name", "health_score", "status"],
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two sessions."""
    try:
        from storage.database import get_database  # noqa: PLC0415
        db = get_database()
        result = db.compare_sessions(int(args.session_a), int(args.session_b))
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result is None:
        print(f"Could not compare sessions {args.session_a} and {args.session_b}", file=sys.stderr)
        return 1

    if args.json:
        _print_json(result)
        return 0

    print(f"Comparing Session {args.session_a} vs Session {args.session_b}")
    print(f"  A: {result.get('session_a_name')} — GPU: {result.get('session_a_gpu')} — Score: {result.get('session_a_score')}")
    print(f"  B: {result.get('session_b_name')} — GPU: {result.get('session_b_gpu')} — Score: {result.get('session_b_score')}")
    print()
    print("Benchmarks:")
    for bm in result.get("benchmarks", []):
        a_ms = bm.get("a_mean_ms")
        b_ms = bm.get("b_mean_ms")
        delta = bm.get("delta_pct")
        line = f"  {bm.get('test_name', ''):<30}  A={a_ms}ms  B={b_ms}ms"
        if delta is not None:
            line += f"  delta={delta:+.1f}%"
        print(line)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a session to a report file."""
    from pathlib import Path as _Path  # noqa: PLC0415
    try:
        from storage.database import get_database  # noqa: PLC0415
        db = get_database()
        if getattr(args, "latest", False):
            session_data = _resolve_latest_session()
            if session_data is None:
                print("No sessions found in database. Run quick-test first.", file=sys.stderr)
                return 1
        else:
            session_id = getattr(args, "session", None)
            if session_id is None:
                print("Provide --session SESSION_ID or --latest", file=sys.stderr)
                return 1
            session_data = db.get_session(int(session_id))
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading session: {exc}", file=sys.stderr)
        return 1

    if session_data is None:
        print("Session not found", file=sys.stderr)
        return 1

    # Normalize flat DB structure to nested report structure
    session_data = _normalize_session_for_report(session_data)

    fmt = (args.format or "json").lower()
    output_dir = _Path(args.output_dir) if args.output_dir else None

    try:
        if fmt == "json":
            from reports.json_report import generate  # noqa: PLC0415
        elif fmt == "csv":
            from reports.csv_report import generate  # noqa: PLC0415
        elif fmt in ("markdown", "md"):
            from reports.markdown_report import generate  # noqa: PLC0415
        elif fmt == "html":
            from reports.html_report import generate  # noqa: PLC0415
        elif fmt in ("excel", "xlsx"):
            from reports.excel_report import generate  # noqa: PLC0415
        else:
            print(f"Unknown format: {fmt}", file=sys.stderr)
            return 1

        file_path = generate(session_data, output_dir=output_dir)
        if args.json:
            _print_json({"file_path": str(file_path), "format": fmt})
        else:
            print(f"Report generated: {file_path}")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1


def _resolve_latest_session() -> Optional[Dict[str, Any]]:
    """Return the most recent session from the DB, or None."""
    from storage.database import get_database  # noqa: PLC0415
    db = get_database()
    sessions = db.get_sessions(limit=1)
    if not sessions:
        return None
    return db.get_session(int(sessions[0]["id"]))


def _normalize_session_for_report(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    When a session is loaded from the DB it has flat system fields (hostname, cpu, ram_gb).
    Report generators expect nested dicts (system_info, nvidia_info, etc.).
    This function promotes flat DB fields into the expected nested structure if needed.
    """
    if session_data is None:
        return {}
    # If already nested, return as-is
    if "system_info" in session_data and isinstance(session_data["system_info"], dict):
        return session_data

    # Build nested dicts from flat DB fields
    normalized = dict(session_data)
    normalized["system_info"] = {
        "hostname": session_data.get("hostname", ""),
        "os_name": session_data.get("os", ""),
        "os_release": "",
        "cpu_model": session_data.get("cpu", ""),
        "cpu_physical_cores": None,
        "cpu_logical_cores": None,
        "ram_total_gb": session_data.get("ram_gb", 0.0),
        "ram_available_gb": None,
        "python_version": "",
    }
    normalized["nvidia_info"] = {
        "available": bool(session_data.get("gpu_name")),
        "gpu_name": session_data.get("gpu_name", ""),
        "gpu_uuid": "",
        "driver_version": session_data.get("driver_version", ""),
        "cuda_driver_version": session_data.get("cuda_version", ""),
        "compute_capability": "",
        "vram_total_mb": 0.0,
        "vram_used_mb": 0.0,
        "vram_free_mb": 0.0,
        "temperature_c": 0.0,
        "gpu_utilization_pct": None,
        "power_draw_w": None,
        "power_limit_w": None,
        "performance_state": "",
        "gpu_clock_mhz": None,
        "mem_clock_mhz": None,
    }
    normalized["cuda_info"] = {
        "nvcc_available": False,
        "nvcc_version": session_data.get("cuda_version", ""),
        "cuda_runtime_available": bool(session_data.get("cuda_version")),
        "cuda_runtime_version": session_data.get("cuda_version", ""),
        "native_benchmark_available": False,
    }
    normalized.setdefault("pcie_info", {})
    normalized.setdefault("tool_status", {})
    normalized.setdefault("amd_info", {})
    return normalized


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Show diagnosis for a session."""
    try:
        from storage.database import get_database  # noqa: PLC0415
        db = get_database()
        if getattr(args, "latest", False):
            session_data = _resolve_latest_session()
            if session_data is None:
                print("No sessions found in database.", file=sys.stderr)
                return 1
        else:
            session_id = getattr(args, "session", None)
            if session_id is None:
                print("Provide --session SESSION_ID or --latest", file=sys.stderr)
                return 1
            session_data = db.get_session(int(session_id))
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading session: {exc}", file=sys.stderr)
        return 1

    if session_data is None:
        print("Session not found", file=sys.stderr)
        return 1

    diag_results = session_data.get("diagnosis_results", []) or []

    if args.json:
        _print_json(diag_results)
        return 0

    shown_id = session_data.get("id", getattr(args, "session", "latest"))
    print(f"Diagnosis for session {shown_id}:\n")
    if not diag_results:
        print("  No diagnosis results.")
        return 0

    for d in diag_results:
        sev = d.get("severity", "INFO")
        title = d.get("title", "")
        cat = d.get("category", "")
        summary = d.get("summary", "")
        evidence = d.get("evidence", "")
        rec = d.get("recommendation", "")
        conf = d.get("confidence", 0)
        print(f"[{sev}] {title}")
        print(f"  Category:   {cat}  Confidence: {conf*100:.0f}%")
        print(f"  Summary:    {summary}")
        print(f"  Evidence:   {evidence}")
        if rec:
            print(f"  Recommend.: {rec}")
        print()

    return 0


def cmd_demo_report(args: argparse.Namespace) -> int:
    """Generate all 4 sample reports from mock data into output/."""
    from pathlib import Path as _Path  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    output_dir_str = getattr(args, "output_dir", "") or ""
    if output_dir_str:
        output_dir = _Path(output_dir_str)
    else:
        output_dir = _Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading examples/sample_session.json; fall back to inline mock data
    sample_path = _Path(__file__).parent.parent / "examples" / "sample_session.json"
    if sample_path.exists():
        try:
            with sample_path.open(encoding="utf-8") as fh:
                session_data = _json.load(fh)
        except Exception:  # noqa: BLE001
            session_data = None
    else:
        session_data = None

    if session_data is None:
        # Inline mock session for demo
        session_data = {
            "session_id": "demo-sample-001",
            "session_name": "Demo Sample Session",
            "started_at": "2026-07-07T00:00:00+00:00",
            "completed_at": "2026-07-07T00:01:30+00:00",
            "status": "completed",
            "health_score": 72.5,
            "score_confidence": 0.65,
            "score_details": {
                "positive_findings": [
                    "Python environment operational",
                    "nvidia-smi available",
                    "GPU temperature data available: 58.0 C",
                ],
                "missing_data": [
                    "native_benchmark_executable",
                    "memory_bandwidth_test",
                ],
                "deductions": [
                    {"category": "environment_readiness", "pts": -3,
                     "reason": "native executable not built"},
                ],
                "category_scores": {
                    "environment_readiness": 13.0,
                    "gpu_runtime_availability": 15.0,
                    "pcie_memory_transfer": 10.0,
                    "kernel_correctness": 20.0,
                    "kernel_performance_consistency": 5.0,
                    "thermal_power_stability": 9.5,
                },
            },
            "system_info": {
                "hostname": "demo-machine",
                "os_name": "Windows",
                "os_release": "10",
                "cpu_model": "Intel Core i9-12900K",
                "cpu_physical_cores": 16,
                "cpu_logical_cores": 24,
                "ram_total_gb": 64.0,
                "ram_available_gb": 48.0,
                "python_version": "3.11.9",
            },
            "nvidia_info": {
                "available": True,
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "driver_version": "545.84",
                "cuda_driver_version": "12.3",
                "compute_capability": "8.6",
                "vram_total_mb": 24576,
                "vram_used_mb": 512,
                "vram_free_mb": 24064,
                "temperature_c": 58.0,
                "gpu_utilization_pct": 0,
                "power_draw_w": 35.0,
                "power_limit_w": 370.0,
                "performance_state": "P8",
            },
            "cuda_info": {
                "nvcc_available": True,
                "nvcc_version": "12.3",
                "cuda_runtime_available": True,
                "cuda_runtime_version": "12.3",
                "native_benchmark_available": False,
                "native_benchmark_path": "",
            },
            "pcie_info": {
                "available": True,
                "pcie_gen_current": 4,
                "pcie_width_current": 16,
                "pcie_gen_max": 4,
                "pcie_width_max": 16,
                "bandwidth_gbps_current": 31.5,
                "bandwidth_gbps_theoretical": 32.0,
                "is_bottlenecked": False,
            },
            "tool_status": {
                "nvcc": {"exists": True, "version": "12.3", "path": "nvcc"},
                "cmake": {"exists": True, "version": "3.28.0", "path": "cmake"},
                "nvidia-smi": {"exists": True, "version": "545.84", "path": "nvidia-smi"},
                "nsys": {"exists": False, "version": None, "path": None},
                "ncu": {"exists": False, "version": None, "path": None},
            },
            "amd_info": {"available": False, "rocm_available": False},
            "results": [
                {
                    "test_name": "cpu_vector_add",
                    "data_type": "float32",
                    "input_size": 1000000,
                    "cpu_time_ms": 2.341,
                    "gpu_time_ms": None,
                    "bandwidth_gbps": None,
                    "speedup": None,
                    "correctness_pass": True,
                    "mean": 2.341,
                    "standard_deviation": 0.15,
                    "notes": "CPU baseline — NumPy",
                    "error": None,
                    "backend": "cpu",
                    "status": "PASS",
                },
                {
                    "test_name": "vector_add",
                    "data_type": "float32",
                    "input_size": 16777216,
                    "cpu_time_ms": 18.2,
                    "gpu_time_ms": 0.82,
                    "bandwidth_gbps": 245.3,
                    "speedup": 22.2,
                    "correctness_pass": True,
                    "mean": 0.82,
                    "standard_deviation": 0.03,
                    "notes": "Sample data — NOT real measurement",
                    "error": None,
                    "backend": "cuda",
                    "status": "PASS",
                },
                {
                    "test_name": "gemm_tiled",
                    "data_type": "float32",
                    "input_size": 512,
                    "cpu_time_ms": 1850.0,
                    "gpu_time_ms": 4.7,
                    "bandwidth_gbps": None,
                    "speedup": 393.6,
                    "correctness_pass": True,
                    "mean": 4.7,
                    "standard_deviation": 0.21,
                    "notes": "Sample data — NOT real measurement",
                    "error": None,
                    "backend": "cuda",
                    "status": "PASS",
                },
            ],
            "diagnosis_results": [
                {
                    "rule_id": "toolchain_completeness",
                    "severity": "INFO",
                    "category": "HEALTHY",
                    "title": "Core Build Toolchain Complete",
                    "summary": "nvcc and cmake are present; native benchmarks can be built.",
                    "evidence": "Tool check: nvcc, cmake all found.",
                    "confidence": 1.0,
                    "recommendation": "No action required.",
                    "verification_step": "",
                },
                {
                    "rule_id": "profiler_unavailable",
                    "severity": "INFO",
                    "category": "TOOLCHAIN_INCOMPLETE",
                    "title": "Profiling Tools Not Available",
                    "summary": "Missing: nsys (Nsight Systems), ncu (Nsight Compute).",
                    "evidence": "Tool check: nsys=missing, ncu=missing.",
                    "confidence": 1.0,
                    "recommendation": "Install Nsight Systems and Nsight Compute from developer.nvidia.com.",
                    "verification_step": "Run nsys --version and ncu --version after installation.",
                },
            ],
        }

    # Generate all 4 formats
    results: Dict[str, Any] = {}
    formats = [
        ("json", "reports.json_report"),
        ("md", "reports.markdown_report"),
        ("html", "reports.html_report"),
        ("xlsx", "reports.excel_report"),
    ]

    for fmt, module_path in formats:
        try:
            import importlib  # noqa: PLC0415
            mod = importlib.import_module(module_path)
            # Override filename to use "sample" tag
            file_path = mod.generate(session_data, output_dir=output_dir)
            # Rename to sample name
            new_name = f"gpu_insight_report_sample.{fmt}"
            new_path = output_dir / new_name
            if file_path.name != new_name:
                import shutil  # noqa: PLC0415
                shutil.copy2(str(file_path), str(new_path))
                try:
                    file_path.unlink()
                except OSError:
                    pass
                file_path = new_path
            results[fmt] = str(file_path)
            print(f"  [{fmt.upper():4s}] {file_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{fmt.upper():4s}] FAILED: {exc}", file=sys.stderr)
            results[fmt] = f"ERROR: {exc}"

    print(f"\nDemo reports generated in: {output_dir.resolve()}")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-insight",
        description=f"{APP_NAME} CLI\n{APP_SUBTITLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"GPU Insight Lab {APP_VERSION}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", help="Command")

    # system-info
    p_sys = sub.add_parser("system-info", help="Collect and display system info")
    p_sys.add_argument("--json", action="store_true", help="Output as JSON")

    # quick-test
    p_quick = sub.add_parser("quick-test", help="Run quick benchmark")
    p_quick.add_argument("--json", action="store_true", help="Output as JSON")
    p_quick.add_argument("--output-dir", default="", help="Output directory for reports")
    p_quick.add_argument("--no-save", action="store_true", help="Do not save session to DB")

    # full-test
    p_full = sub.add_parser("full-test", help="Run full benchmark")
    p_full.add_argument("--json", action="store_true", help="Output as JSON")
    p_full.add_argument("--output-dir", default="", help="Output directory")
    p_full.add_argument("--no-save", action="store_true", help="Do not save session to DB")

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Run specific benchmark test")
    p_bm.add_argument("--test", required=True, help="Test name (e.g. vector_add, gemm_tiled)")
    p_bm.add_argument("--json", action="store_true", help="Output as JSON")

    # history
    p_hist = sub.add_parser("history", help="List stored sessions")
    p_hist.add_argument("--json", action="store_true", help="Output as JSON")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two sessions")
    p_cmp.add_argument("--session-a", required=True, help="Session A ID")
    p_cmp.add_argument("--session-b", required=True, help="Session B ID")
    p_cmp.add_argument("--json", action="store_true", help="Output as JSON")

    # export
    p_exp = sub.add_parser("export", help="Export session to report")
    p_exp_grp = p_exp.add_mutually_exclusive_group(required=True)
    p_exp_grp.add_argument("--session", help="Session ID")
    p_exp_grp.add_argument("--latest", action="store_true", help="Use the most recent session")
    p_exp.add_argument("--format", default="json",
                       choices=["json", "csv", "markdown", "md", "html", "excel", "xlsx"],
                       help="Report format")
    p_exp.add_argument("--output-dir", default="", help="Output directory")
    p_exp.add_argument("--json", action="store_true", help="JSON output (status only)")

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Show diagnosis for a session")
    p_diag_grp = p_diag.add_mutually_exclusive_group(required=True)
    p_diag_grp.add_argument("--session", help="Session ID")
    p_diag_grp.add_argument("--latest", action="store_true", help="Use the most recent session")
    p_diag.add_argument("--json", action="store_true", help="Output as JSON")

    # demo-report
    p_demo = sub.add_parser("demo-report", help="Generate sample reports from mock data into output/")
    p_demo.add_argument("--output-dir", default="", help="Output directory (default: output/)")

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "system-info": cmd_system_info,
        "quick-test": cmd_quick_test,
        "full-test": cmd_full_test,
        "benchmark": cmd_benchmark,
        "history": cmd_history,
        "compare": cmd_compare,
        "export": cmd_export,
        "diagnose": cmd_diagnose,
        "demo-report": cmd_demo_report,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("CLI command %s failed", args.command)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
