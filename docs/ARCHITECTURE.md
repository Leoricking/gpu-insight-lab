# GPU Insight Lab — System Architecture

## Overview

GPU Insight Lab is a layered Python application with an optional native C++/CUDA binary for kernel-level benchmarks. The architecture separates concerns cleanly: collection, benchmarking, diagnosis, storage, reporting, and presentation.

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `app/branding.py` | Central constants (APP_NAME, VERSION, etc.) imported everywhere |
| `app/config.py` | Load/save config from `~/.gpu_insight_lab/config.json` |
| `app/features.py` | Feature registry (Free/Pro/Lab editions) |
| `app/cli.py` | argparse CLI with 8 commands |
| `app/main.py` | PySide6 GUI entry point |
| `app/gui/` | QMainWindow, pages, QThread workers |
| `app/controllers/` | Thin adapters between GUI/CLI and business logic |
| `collectors/` | Collect system/GPU/toolchain info; all gracefully degrade |
| `benchmarks/` | CPU baselines, native runner, session orchestrator, schemas |
| `diagnosis/` | Evidence-based rule engine + GPU Insight Score |
| `storage/` | SQLite persistence with auto-migration |
| `reports/` | JSON, CSV, Markdown, HTML, Excel output |
| `profilers/` | Nsight Systems, Nsight Compute, nvidia-smi monitor, ROCm stub |
| `workloads/` | Image batch, media pipeline, LLM placeholder, custom command |
| `native/` | CUDA C++ benchmark kernels (separate binary) |

## Data Flow

```
User
  │
  ├─ CLI (app/cli.py) ─────────────────────────────────────────────────────┐
  └─ GUI (app/gui/main_window.py)                                          │
       │                                                                   │
       ↓ (via controllers or direct)                                       │
  Benchmark Runner (benchmarks/runner.py)                                  │
       │                                                                   │
       ├─ Collectors ──────────────────────────────────────────────────────┤
       │   ├─ system_collector.py (psutil + platform)                      │
       │   ├─ nvidia_collector.py (pynvml → nvidia-smi fallback)           │
       │   ├─ cuda_collector.py (nvcc, CUDA_HOME)                          │
       │   ├─ pcie_collector.py (nvidia-smi)                               │
       │   ├─ tool_collector.py (which/where for each tool)               │
       │   └─ amd_collector.py (rocm-smi, rocminfo)                       │
       │                                                                   │
       ├─ CPU Baselines (benchmarks/cpu_baselines.py)                      │
       │                                                                   │
       ├─ Native Runner (benchmarks/native_runner.py)                      │
       │   └─ gpu_insight_benchmark.exe (native/cuda/)                    │
       │       ├─ vector_add.cu                                            │
       │       ├─ reduction.cu                                             │
       │       ├─ transpose.cu                                             │
       │       ├─ gemm_naive.cu + gemm_tiled.cu                           │
       │       ├─ memory_bandwidth.cu                                      │
       │       ├─ stream_pipeline.cu                                       │
       │       └─ image_grayscale.cu                                       │
       │                                                                   │
       ↓                                                                   │
  BenchmarkSession (benchmarks/schemas.py)                                 │
       │                                                                   │
       ├─ Diagnosis Engine (diagnosis/engine.py + rules.py)               │
       │   └─ GPU Insight Score (diagnosis/scoring.py)                    │
       │                                                                   │
       ├─ Storage (storage/database.py + migrations.py)                   │
       │   └─ gpu_insight.sqlite                                           │
       │                                                                   │
       └─ Reports (reports/)                    ──────────────────────────┘
           ├─ json_report.py
           ├─ csv_report.py
           ├─ markdown_report.py
           ├─ html_report.py (Jinja2)
           └─ excel_report.py (openpyxl)
```

## Key Design Decisions

1. **Graceful degradation everywhere**: Every collector returns partial data rather than crashing. Missing tools return `exists=False`, not exceptions.

2. **Evidence-based diagnosis**: Every `DiagnosisResult` has an `evidence` string. No finding is made without data. Low confidence is explicit, not hidden.

3. **Native binary separation**: CUDA kernels live in a separate C++/CUDA binary (`gpu_insight_benchmark`). Python orchestrates it via subprocess and parses JSON stdout. This means Python works even without a CUDA compiler.

4. **No shell=True**: All subprocess calls use list arguments to avoid shell injection.

5. **QThread for all I/O**: GUI never blocks the event loop. Every benchmark, collection, and report generation runs in a QThread worker with signals for progress.

6. **Pathlib everywhere**: All file paths use `pathlib.Path`, not `os.path`.

7. **SQLite for persistence**: Session history in `gpu_insight.sqlite` with versioned migrations.
