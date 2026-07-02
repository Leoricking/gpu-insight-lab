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

    print(f"[GPU Insight Lab] Running benchmark: {args.test}")
    result = run_single_test(args.test)

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
        session_data = db.get_session(int(args.session))
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading session: {exc}", file=sys.stderr)
        return 1

    if session_data is None:
        print(f"Session {args.session} not found", file=sys.stderr)
        return 1

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


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Show diagnosis for a session."""
    try:
        from storage.database import get_database  # noqa: PLC0415
        db = get_database()
        session_data = db.get_session(int(args.session))
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading session: {exc}", file=sys.stderr)
        return 1

    if session_data is None:
        print(f"Session {args.session} not found", file=sys.stderr)
        return 1

    diag_results = session_data.get("diagnosis_results", []) or []

    if args.json:
        _print_json(diag_results)
        return 0

    print(f"Diagnosis for session {args.session}:\n")
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
    p_exp.add_argument("--session", required=True, help="Session ID")
    p_exp.add_argument("--format", default="json",
                       choices=["json", "csv", "markdown", "html", "excel"],
                       help="Report format")
    p_exp.add_argument("--output-dir", default="", help="Output directory")
    p_exp.add_argument("--json", action="store_true", help="JSON output (status only)")

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Show diagnosis for a session")
    p_diag.add_argument("--session", required=True, help="Session ID")
    p_diag.add_argument("--json", action="store_true", help="Output as JSON")

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
