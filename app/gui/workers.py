"""
GPU Insight Lab - QThread Workers
Background workers for all long-running operations.
All emit: progress(int), status(str), finished(dict), error(str)
"""

from __future__ import annotations

import dataclasses
import logging
import traceback
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import QThread, Signal  # type: ignore
    _PYSIDE_AVAILABLE = True
except ImportError:
    _PYSIDE_AVAILABLE = False
    # Stubs for import without PySide6
    class QThread:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: ...
        def start(self) -> None: ...
        def quit(self) -> None: ...
        def wait(self, ms: int = 0) -> bool: return True

    class Signal:  # type: ignore
        def __init__(self, *args: Any) -> None: ...
        def emit(self, *args: Any) -> None: ...
        def connect(self, *args: Any) -> None: ...


if _PYSIDE_AVAILABLE:
    class SystemInfoWorker(QThread):
        progress = Signal(int)
        status = Signal(str)
        finished = Signal(dict)
        error = Signal(str)

        def run(self) -> None:
            try:
                self.progress.emit(10)
                self.status.emit("Collecting system info...")
                import dataclasses as dc  # noqa: PLC0415
                from collectors import system_collector, nvidia_collector  # noqa: PLC0415
                from collectors import cuda_collector, pcie_collector, tool_collector, amd_collector  # noqa: PLC0415

                self.progress.emit(20)
                sys_info = system_collector.collect()
                self.progress.emit(40)
                nv_info = nvidia_collector.collect()
                self.progress.emit(55)
                cuda_info = cuda_collector.collect()
                self.progress.emit(65)
                pcie_info = pcie_collector.collect()
                self.progress.emit(78)
                tools = tool_collector.collect()
                self.progress.emit(88)
                amd_info = amd_collector.collect()

                data = {
                    "system": dc.asdict(sys_info),
                    "nvidia": dc.asdict(nv_info),
                    "cuda": dc.asdict(cuda_info),
                    "pcie": dc.asdict(pcie_info),
                    "tools": {k: dc.asdict(v) for k, v in tools.items()},
                    "amd": dc.asdict(amd_info),
                }
                self.progress.emit(100)
                self.status.emit("System info collected")
                self.finished.emit(data)
            except Exception as exc:  # noqa: BLE001
                logger.exception("SystemInfoWorker failed")
                self.error.emit(str(exc))

    class BenchmarkWorker(QThread):
        progress = Signal(int)
        status = Signal(str)
        finished = Signal(dict)
        error = Signal(str)

        def __init__(
            self,
            test_name: str = "quick",
            repeat: int = 10,
            data_size: int = 0,
            block_size: int = 0,
            output_dir: str = "",
        ) -> None:
            super().__init__()
            self.test_name = test_name
            self.repeat = repeat
            self.data_size = data_size
            self.block_size = block_size
            self.output_dir = output_dir
            self._cancelled = False

        def cancel(self) -> None:
            self._cancelled = True

        def run(self) -> None:
            try:
                from benchmarks.runner import run_quick_test, run_full_test, run_single_test  # noqa: PLC0415

                def _prog(pct: int, msg: str) -> None:
                    if not self._cancelled:
                        self.progress.emit(pct)
                        self.status.emit(msg)

                if self._cancelled:
                    return

                if self.test_name == "quick":
                    session = run_quick_test(progress_callback=_prog)
                elif self.test_name == "full":
                    session = run_full_test(progress_callback=_prog)
                else:
                    self.status.emit(f"Running {self.test_name}...")
                    result = run_single_test(
                        self.test_name,
                        repeat=self.repeat,
                        block_size=self.block_size if self.block_size else None,
                        data_size=self.data_size if self.data_size else None,
                        progress_callback=_prog,
                    )
                    # Wrap in a dict
                    self.progress.emit(100)
                    self.finished.emit({"single_result": result.to_dict() if hasattr(result, "to_dict") else {}})
                    return

                data = session.to_dict() if hasattr(session, "to_dict") else {}
                self.finished.emit(data)
            except Exception as exc:  # noqa: BLE001
                logger.exception("BenchmarkWorker failed")
                self.error.emit(f"{type(exc).__name__}: {exc}")

    class DiagnosisWorker(QThread):
        progress = Signal(int)
        status = Signal(str)
        finished = Signal(dict)
        error = Signal(str)

        def __init__(self, session_data: Dict[str, Any]) -> None:
            super().__init__()
            self.session_data = session_data

        def run(self) -> None:
            try:
                self.progress.emit(10)
                self.status.emit("Running diagnosis engine...")
                from diagnosis.engine import run_diagnosis  # noqa: PLC0415
                from diagnosis.scoring import compute_score  # noqa: PLC0415

                class _FakeSession:
                    def __init__(self, d: Dict[str, Any]) -> None:
                        self.results = []
                        self.system_info = d.get("system_info", {}) or {}
                        self.nvidia_info = d.get("nvidia_info", {}) or {}
                        self.cuda_info = d.get("cuda_info", {}) or {}
                        self.pcie_info = d.get("pcie_info", {}) or {}
                        self.tool_status = d.get("tool_status", {}) or {}
                        self.amd_info = d.get("amd_info", {}) or {}
                        self.diagnosis_results = []

                        for r in d.get("results", []):
                            from benchmarks.schemas import BenchmarkResult  # noqa: PLC0415
                            if isinstance(r, dict):
                                self.results.append(BenchmarkResult.from_dict(r))

                session = _FakeSession(self.session_data)
                self.progress.emit(40)
                diag = run_diagnosis(session)
                self.progress.emit(75)
                score = compute_score(session)
                self.progress.emit(100)
                self.finished.emit({
                    "diagnosis_results": diag,
                    "score": score.score,
                    "confidence": score.confidence,
                    "score_details": dataclasses.asdict(score),
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("DiagnosisWorker failed")
                self.error.emit(str(exc))

    class ReportWorker(QThread):
        progress = Signal(int)
        status = Signal(str)
        finished = Signal(dict)
        error = Signal(str)

        def __init__(
            self,
            session_data: Dict[str, Any],
            fmt: str = "json",
            output_dir: str = "",
        ) -> None:
            super().__init__()
            self.session_data = session_data
            self.fmt = fmt
            self.output_dir = output_dir

        def run(self) -> None:
            try:
                from pathlib import Path  # noqa: PLC0415
                self.progress.emit(10)
                self.status.emit(f"Generating {self.fmt} report...")

                output_dir = Path(self.output_dir) if self.output_dir else None
                fmt = self.fmt.lower()

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
                    self.error.emit(f"Unknown format: {self.fmt}")
                    return

                self.progress.emit(50)
                file_path = generate(self.session_data, output_dir=output_dir)
                self.progress.emit(100)
                self.status.emit(f"Report saved: {file_path}")
                self.finished.emit({"file_path": str(file_path), "format": self.fmt})
            except Exception as exc:  # noqa: BLE001
                logger.exception("ReportWorker failed")
                self.error.emit(str(exc))

else:
    # Stub classes when PySide6 not installed
    class SystemInfoWorker:  # type: ignore
        def __init__(self) -> None: pass

    class BenchmarkWorker:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass

    class DiagnosisWorker:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass

    class ReportWorker:  # type: ignore
        def __init__(self, *a: Any, **kw: Any) -> None: pass
