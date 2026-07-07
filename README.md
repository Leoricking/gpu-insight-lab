# GPU Insight Lab

**Cross-Vendor GPU Performance, Validation and Workload Diagnostics**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)]()

GPU Insight Lab is a Python-based GPU performance diagnostics platform that collects
system and GPU telemetry, runs CUDA benchmark kernels, applies an evidence-based diagnosis
engine, and produces multi-format reports — all from a single CLI command or a PySide6 GUI.

---

## Quick Interview Demo

Run the one-command demo script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_interview_demo.ps1
```

Or run individual commands:

```bash
python -m app.cli system-info
python -m app.cli quick-test
python -m app.cli benchmark --test memory
python -m app.cli benchmark --test pcie
python -m app.cli benchmark --test gemm
python -m app.cli diagnose --latest
python -m app.cli demo-report
```

> **No GPU required** for `system-info`, `diagnose`, `demo-report`. Benchmarks return `SKIPPED`/`NOT_VALIDATED` when CUDA hardware is unavailable — this is expected behavior, not an error.

| Document | Purpose |
|----------|---------|
| [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) | Live demo script, talking points, role-specific positioning |
| [docs/CODE_COVERAGE_AUDIT.md](docs/CODE_COVERAGE_AUDIT.md) | Honest implemented / partial / roadmap breakdown |
| [docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md) | Technical claims, CUDA question bank, portfolio projects |
| [docs/12_WEEK_CUDA_JOB_ROADMAP.md](docs/12_WEEK_CUDA_JOB_ROADMAP.md) | 12-week CUDA → NVIDIA/AMD job readiness roadmap |

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [CLI Usage](#cli-usage)
5. [GUI Usage](#gui-usage)
6. [Benchmark Suite](#benchmark-suite)
7. [Diagnosis Engine](#diagnosis-engine)
8. [GPU Insight Score](#gpu-insight-score)
9. [Report Formats](#report-formats)
10. [Session History](#session-history)
11. [Native CUDA Binary](#native-cuda-binary)
12. [Configuration](#configuration)
13. [Development and Testing](#development-and-testing)
14. [Architecture](#architecture)
15. [Limitations and Disclaimer](#limitations-and-disclaimer)
16. [License and Trademark Notice](#license-and-trademark-notice)

---

## Features

- **System and GPU collection** — CPU, RAM, OS, NVIDIA GPU info via pynvml with
  nvidia-smi fallback, CUDA/ROCm toolchain detection, PCIe link status, AMD GPU stub
- **Native CUDA benchmarks** — 7 C++ CUDA kernels (vector add, reduction, transpose,
  GEMM naive + tiled, memory bandwidth, stream pipeline, image grayscale) compiled for
  sm_75 through sm_90
- **CPU baselines** — NumPy-based reference implementations for vector add, matrix
  multiply, and image grayscale with warmup + repeat statistics
- **Evidence-based diagnosis** — 9 rules, every finding has a non-empty evidence string
  citing specific measured values; no speculation
- **GPU Insight Score** — 0–100 composite score across 6 categories with confidence rating
- **Multi-format reports** — JSON, CSV, Markdown, HTML (Jinja2), Excel (openpyxl, 7 sheets)
- **SQLite session history** — versioned schema migrations, session comparison with delta %
- **PySide6 GUI** — QMainWindow with sidebar navigation, QThread workers (GUI never blocks)
- **argparse CLI** — 10 commands with `--json` flag and exit codes 0/1/2
- **Graceful degradation** — every collector returns partial data; missing hardware or
  tools produce informative findings, not crashes

### Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| System Inspector | Implemented | Works without GPU |
| Memory Benchmark | Implemented | Requires CUDA GPU + native binary |
| Kernel Lab (7 kernels) | Implemented | Requires CUDA GPU + native binary |
| Diagnosis Engine | Implemented | Works without GPU |
| GPU Insight Score | Implemented | Works without GPU |
| Multi-format Reports | Implemented | JSON/CSV/MD always; HTML needs jinja2; Excel needs openpyxl |
| SQLite History | Implemented | |
| PySide6 GUI | Partial | Requires PySide6; pages are stubs |
| CLI Automation | Implemented | 10 commands |
| CUDA to HIP Demo | Partial | vector_add_hip, reduction_hip, gemm_naive_hip — NOT_VALIDATED on AMD |
| prefix_sum kernel | Skeleton / NOT_VALIDATED | Blelloch scan skeleton, not performance-optimized |
| convolution_2d kernel | Skeleton / NOT_VALIDATED | Tiled 2D conv skeleton, boundary conditions incomplete |
| softmax / layer_norm / GELU | **ROADMAP** | Not implemented |
| Flash Attention | **ROADMAP** | Not implemented |
| INT8 quantization | **ROADMAP** | Not implemented |
| PyTorch extension | **ROADMAP** | Not implemented |
| TensorRT plugin | **ROADMAP** | Not implemented |
| cuFFT / cuBLAS benchmarks | **ROADMAP** | Not implemented |
| Streamlit dashboard | **ROADMAP** | Not implemented |
| Multi-machine import | **ROADMAP** | Not implemented |
| Company report templates | **ROADMAP** | Not implemented |
| Batch execution | **ROADMAP** | Not implemented |
| AMD HIP real benchmark | NOT_VALIDATED | Requires AMD GPU + ROCm |

---

## Quick Start

```bash
# Install (all optional dependencies)
pip install -e ".[all]"

# Run a quick benchmark and show results
gpu-insight quick-test

# Get system info as JSON
gpu-insight system-info --json

# Run full benchmark suite and save HTML report
gpu-insight full-test --output-dir ~/gpu_reports

# Launch GUI
gpu-insight gui
```

---

## Installation

### Requirements

- Python 3.11 or 3.12
- Windows 10/11 or Linux (Ubuntu 20.04+)
- NVIDIA GPU with driver 520+ recommended (tool works without GPU for CPU baselines)
- CUDA Toolkit 11.8+ optional (required only to recompile native kernels)

### Install from Source

```bash
git clone https://github.com/yourusername/gpu-insight-lab.git
cd gpu-insight-lab

# Minimal install (CLI + collectors + diagnosis)
pip install -e .

# Full install (adds GUI, reports, pynvml, Pillow)
pip install -e ".[all]"

# Development install (adds pytest)
pip install -e ".[all,dev]"
```

### Verify Installation

```bash
python -m compileall app collectors benchmarks diagnosis storage reports profilers workloads -q
python -m pytest tests/ -v
gpu-insight --version
```

---

## CLI Usage

```
usage: gpu-insight [-h] [--version] {system-info,quick-test,full-test,benchmark,
                                      diagnose,export,history,compare,demo-report} ...
```

### Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `system-info` | Collect and display system + GPU info | `--json` |
| `quick-test` | Run quick benchmark suite | `--json`, `--no-save`, `--output-dir` |
| `full-test` | Run full benchmark suite | `--json`, `--no-save`, `--output-dir` |
| `benchmark` | Run one specific benchmark | `--test NAME`, `--json` |
| `diagnose` | Show diagnosis for a session | `--session ID` or `--latest`, `--json` |
| `export` | Generate report from session | `--session ID` or `--latest`, `--format json/csv/md/html/xlsx` |
| `history` | List stored sessions | `--json` |
| `compare` | Compare two sessions | `--session-a ID`, `--session-b ID`, `--json` |
| `demo-report` | Generate sample reports from mock data | `--output-dir DIR` |

> **Note:** The `gui` command requires PySide6 to be installed (`pip install PySide6`).
> It is not registered in the CLI parser by default in v0.1.0 but can be launched via `python -m app.main`.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (collection or benchmark failure) |
| 2 | Partial success (some data missing, tool continued) |

### Examples

```bash
# Show all system + GPU info as JSON
gpu-insight system-info --json

# Run a single vector_add benchmark
gpu-insight benchmark --test vector_add

# Run a single memory bandwidth benchmark (alias: pcie)
gpu-insight benchmark --test memory

# Generate an HTML report from the most recent session
gpu-insight export --latest --format html

# Full test, don't save to database
gpu-insight full-test --no-save

# Run diagnosis on the most recent stored session
gpu-insight diagnose --latest

# Generate demo reports from mock data (no GPU required)
gpu-insight demo-report --output-dir output/

# List all stored sessions
gpu-insight history

# Compare two sessions
gpu-insight compare --session-a 1 --session-b 2
```

---

## GUI Usage

```bash
gpu-insight gui
```

The PySide6 GUI provides:
- **Dashboard** — GPU Insight Score, key metrics, quick-action buttons
- **System Info** — All collected system + GPU data in a tree view
- **Benchmarks** — Run quick/full/single benchmarks with progress bar
- **Diagnosis** — Severity-filtered diagnosis results with evidence
- **History** — Session timeline, session comparison, trend charts
- Plus 6 placeholder pages for Pro/Lab edition features

All benchmark runs execute in a QThread worker; the GUI never blocks during measurement.

---

## Benchmark Suite

### Native CUDA Kernels

| Kernel | Measures | Output Metric |
|--------|----------|---------------|
| `vector_add` | Memory bandwidth (bound) | GB/s |
| `reduction` | Parallel reduction efficiency | GB/s input |
| `transpose` | Coalesced memory access | GB/s |
| `gemm_naive` | Unoptimized GEMM baseline | GFLOP/s |
| `gemm_tiled` | Shared-memory tiled GEMM | GFLOP/s |
| `memory_bandwidth` | Peak device memory bandwidth | GB/s |
| `stream_pipeline` | Async concurrent execution | GB/s + ms latency |
| `image_grayscale` | 2D pixel processing | Mpix/s |

### CPU Python Baselines

| Benchmark | Implementation | Metric |
|-----------|---------------|--------|
| `cpu_vector_add` | NumPy element-wise | GB/s |
| `cpu_matrix_multiply` | NumPy `@` (BLAS) | GFLOP/s |
| `cpu_image_grayscale` | NumPy weighted sum | Mpix/s |

### Statistics

Every benchmark produces: mean, median, min, max, standard deviation (population),
coefficient of variation (CV). High CV (> 10%) triggers the `HIGH_VARIANCE` diagnosis rule.

### Warmup

3 warmup iterations are discarded before measurement begins. Default repeat count is 10.
Use `--repeat N` to increase for more stable results.

---

## Diagnosis Engine

The diagnosis engine applies 9 evidence-based rules to each session:

| Rule ID | Category | What It Detects |
|---------|----------|-----------------|
| `DRIVER_MISSING` | Environment | No NVIDIA driver installed / detectable |
| `PCIE_BOTTLENECK` | PCIe/Memory | GPU running at lower link width/gen than maximum |
| `THERMAL_THROTTLE` | Thermal/Power | Temperature >= 83°C or active throttle reasons |
| `LOW_MEMORY_BANDWIDTH` | Performance | Bandwidth significantly below expected for GPU model |
| `LOW_COMPUTE_THROUGHPUT` | Performance | FP32 GFLOP/s below expected for GPU model |
| `CORRECTNESS_FAILURE` | Kernel Correctness | Benchmark output differs from CPU reference |
| `HIGH_VARIANCE` | Consistency | Coefficient of variation > 10% across repeats |
| `CUDA_TOOLKIT_MISSING` | Environment | nvcc not found on PATH or CUDA_HOME |
| `AMD_NOT_VALIDATED` | Compatibility | AMD GPU detected; results are NOT_VALIDATED |

**Evidence policy**: Every finding includes a non-empty `evidence` string citing specific
measured values. Rules that cannot obtain sufficient data return `None` (not a low-confidence
finding). See [docs/DIAGNOSIS_RULES.md](docs/DIAGNOSIS_RULES.md) for full rule documentation.

---

## GPU Insight Score

The GPU Insight Score is a 0–100 composite score across 6 categories:

| Category | Max Points | What It Measures |
|----------|-----------|-----------------|
| Environment Readiness | 20 | Driver, toolkit, toolchain present |
| GPU Runtime | 15 | GPU detected, pynvml functional, device accessible |
| PCIe / Memory | 20 | Link width/gen at maximum, bandwidth on target |
| Kernel Correctness | 20 | All benchmark correctness checks pass |
| Performance Consistency | 15 | Low CV across benchmark repeats |
| Thermal / Power | 10 | Temperature within safe range, no throttle |

Score confidence is reported as HIGH, MEDIUM, or LOW based on how much data was available.
A score computed with missing hardware data carries LOW confidence and is marked accordingly.

---

## Report Formats

| Format | Command | Notes |
|--------|---------|-------|
| JSON | `--format json` | Full session data, machine-readable |
| CSV | `--format csv` | Benchmark results table, Excel-compatible |
| Markdown | `--format md` | Human-readable, suitable for GitHub/Confluence |
| HTML | `--format html` | Self-contained (inline CSS), no CDN required |
| Excel | `--format excel` | 7 sheets: Summary, System, GPU, Benchmarks, Diagnosis, Score, Raw |

HTML reports use Jinja2 templating and work in air-gapped environments (no external
dependencies). Excel reports use openpyxl with freeze_panes, auto_filter, and severity
color coding; no merged cells for programmatic access.

---

## Session History

Benchmark sessions are stored in `~/.gpu_insight_lab/gpu_insight.sqlite` using SQLite
with WAL mode and foreign key enforcement.

```bash
# List all stored sessions
gpu-insight history

# Compare two sessions (shows delta %)
gpu-insight compare --session-a 1 --session-b 2

# Generate report from a stored session
gpu-insight export --session 1 --format html
```

Schema migrations run automatically. The database version is tracked in `schema_version`.

---

## Native CUDA Binary

The native benchmark binary (`gpu_insight_benchmark` / `gpu_insight_benchmark.exe`)
provides GPU-side kernel timing using CUDA Events. Python orchestrates it via subprocess
and parses JSON stdout.

### Building from Source

```bash
cd native
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Requires CMake 3.18+, CUDA Toolkit 11.8+, C++17 compiler.

Compiled for compute capabilities: sm_75 (Turing), sm_80 (Ampere), sm_86 (Ampere),
sm_89 (Ada), sm_90 (Hopper).

### Running Standalone

```bash
./gpu_insight_benchmark --device-info
./gpu_insight_benchmark --quick --repeat 10
./gpu_insight_benchmark --full --output results.json
./gpu_insight_benchmark --test gemm_tiled --size 1024 --block-size 32
```

Python works without the native binary (CPU baselines still run; CUDA kernel benchmarks
show as unavailable).

---

## Configuration

Default configuration is stored in `~/.gpu_insight_lab/config.json`:

```json
{
  "output_dir": "~/gpu_insight_lab_output",
  "db_path": "~/.gpu_insight_lab/gpu_insight.sqlite",
  "native_binary_path": null,
  "default_repeat": 10,
  "default_timeout": 120,
  "warmup_runs": 3,
  "save_sessions": true,
  "log_level": "INFO"
}
```

All settings can be overridden via CLI flags or environment variables:

| Environment Variable | Config Key | Default |
|---------------------|------------|---------|
| `GPU_INSIGHT_OUTPUT_DIR` | `output_dir` | `~/gpu_insight_lab_output` |
| `GPU_INSIGHT_DB_PATH` | `db_path` | `~/.gpu_insight_lab/gpu_insight.sqlite` |
| `GPU_INSIGHT_REPEAT` | `default_repeat` | `10` |
| `GPU_INSIGHT_TIMEOUT` | `default_timeout` | `120` |

---

## Development and Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov=collectors --cov=benchmarks \
    --cov=diagnosis --cov=storage --cov=reports -v

# Compile-check all Python modules
python -m compileall app collectors benchmarks diagnosis storage reports \
    profilers workloads tests -q

# CLI smoke tests
gpu-insight --version
gpu-insight system-info --json
gpu-insight quick-test
python -c "from app.gui.main_window import MainWindow; print('GUI import OK')"
```

### Test Coverage

| Test File | What It Tests |
|-----------|--------------|
| `tests/test_smoke.py` | Module imports, CLI `--help`, `--version`, JSON output schema |
| `tests/test_diagnosis.py` | Each diagnosis rule, evidence policy, rule returns None when data absent |
| `tests/test_storage.py` | Migration idempotency, save/load round-trip, session comparison |

---

## Architecture

```
app/           PySide6 GUI, argparse CLI, config, features, branding
collectors/    System, NVIDIA, CUDA, PCIe, tool, AMD data collection
benchmarks/    CPU baselines, native runner, session orchestrator, schemas
diagnosis/     Evidence-based rule engine, scoring
storage/       SQLite persistence with migrations
reports/       JSON, CSV, Markdown, HTML, Excel output
profilers/     Nsight Systems, Nsight Compute, nvidia-smi monitor, ROCm stub
workloads/     Image batch, media pipeline, LLM placeholder, custom command
native/        CUDA C++ benchmark kernels (separate binary)
tests/         pytest test suite
docs/          Architecture, methodology, rules, HIP guide, roadmap, interview guide
```

For the complete data flow diagram and design decisions, see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Interview Demo Positioning

GPU Insight Lab is designed as an interview-ready GPU engineering portfolio project. It
demonstrates not only CUDA kernel programming, but also PCIe transfer analysis, benchmark
methodology, profiling integration, correctness validation, evidence-based diagnosis,
regression comparison, and professional report generation.

Traditional CUDA samples show that a developer can write kernels. GPU Insight Lab shows
that the developer can build a reproducible GPU software engineering workflow.

**Target roles**: CUDA Performance Engineer · GPU Software Engineer · HPC Validation Engineer ·
GPU PCIe / System Validation · NVIDIA / AMD GPU Platform Software

See [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) for interview preparation, technical
claims with evidence, job-aligned portfolio projects, and a full CUDA question bank.

See [`docs/12_WEEK_CUDA_JOB_ROADMAP.md`](docs/12_WEEK_CUDA_JOB_ROADMAP.md) for the
12-week CUDA → NVIDIA/AMD job readiness roadmap.

---

## Limitations and Disclaimer

- **NVIDIA-primary**: Full support for NVIDIA GPUs via pynvml/nvidia-smi. AMD GPU results
  are marked `NOT_VALIDATED` throughout. Intel GPU support is not implemented.
- **Single GPU**: Only one GPU is benchmarked per session. Multi-GPU topologies (NVLink,
  SLI) are not covered.
- **FP32 only**: Native benchmarks use 32-bit float. Tensor Core (FP16/BF16/INT8) performance
  is not measured.
- **Microbenchmarks**: Results describe hardware capability under synthetic load, not
  application performance. Real ML training/inference throughput involves many additional
  factors.
- **Benchmarks affect system state**: Running GPU benchmarks at full load will increase
  GPU temperature and may cause thermal throttling. Do not run on production systems
  during live workloads.
- **Diagnosis is advisory**: Findings are based on heuristic rules applied to limited
  data. They are starting points for investigation, not definitive diagnoses.
- **Windows timing**: GPU kernel timing on Windows may have higher jitter than Linux due
  to WDDM scheduling. Results on Windows may show higher CV than equivalent Linux runs.

---

## License and Trademark Notice

Copyright (c) 2026 Rossi. Released under the [MIT License](LICENSE).

"GPU Insight Lab" is an unregistered trademark. Use of the name "GPU Insight Lab" to
identify derivative works, competing products, or services requires written permission.

This software is an independent project and is not affiliated with, endorsed by, or
sponsored by NVIDIA Corporation, Advanced Micro Devices (AMD), or any of their subsidiaries.
"CUDA" is a trademark of NVIDIA Corporation. "ROCm" and "HIP" are trademarks of Advanced
Micro Devices, Inc. All other trademarks are the property of their respective owners.
