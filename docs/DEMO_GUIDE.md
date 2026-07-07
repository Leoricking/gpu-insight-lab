# GPU Insight Lab — Interview Demo Guide

## Core Positioning

GPU Insight Lab is not a benchmark leaderboard. It is a reproducible GPU engineering workflow
that connects environment inspection, CUDA/HIP benchmark execution, PCIe transfer analysis,
diagnosis, regression comparison, and professional reporting.

GPU Insight Lab 不是跑分排行榜，而是一個可重複的 GPU 工程流程，將環境檢查、CUDA/HIP
benchmark、PCIe 傳輸分析、診斷、回歸比較與專業報告串成可驗證流程。

---

## 30-Second Verbal Introduction

"GPU Insight Lab is a Python-based GPU engineering workflow tool I built to demonstrate
end-to-end GPU performance diagnostics. It collects system and GPU telemetry, executes
CUDA and HIP benchmark kernels, analyzes PCIe transfer bandwidth, applies an evidence-based
diagnosis engine with 9 rules, computes a composite GPU Insight Score, and generates
professional reports in JSON, Markdown, HTML, and Excel formats — all from a single CLI
command. The project targets GPU software engineers who need reproducible, verifiable
performance data rather than one-shot benchmark numbers. Unlike a simple CUDA sample that
shows you can write a kernel, GPU Insight Lab shows you can build and maintain a complete
GPU software engineering workflow."

**Chinese version (30秒口頭介紹):**
「GPU Insight Lab 是我開發的 GPU 工程流程工具，展示端對端 GPU 效能診斷能力。
它收集系統與 GPU 遙測、執行 CUDA/HIP benchmark kernel、分析 PCIe 傳輸頻寬、套用
9 條證據式診斷規則、計算複合 GPU Insight Score，並輸出 JSON、Markdown、HTML、Excel
四種格式的專業報告。目標是讓 GPU 工程師獲得可重複、可驗證的效能數據，而非一次性的
benchmark 結果。」

---

## 3-Minute CLI Demo Flow

Run these commands in order. Each takes about 20-30 seconds.

### Step 1 — Show available commands
```bash
python -m app.cli --help
```
Expected output: lists all 9 subcommands (system-info, quick-test, full-test, benchmark,
diagnose, export, history, compare, demo-report), version string, description.

### Step 2 — Collect environment data
```bash
python -m app.cli system-info
```
Expected output: System section (hostname, OS, CPU, RAM), GPU section (name, driver,
CUDA version, VRAM, temperature, PCIe link gen/width), CUDA Toolchain section (nvcc
availability), Tools section (cmake, nvidia-smi, nsys, ncu status).
Without GPU: shows "No NVIDIA GPU detected" — this is correct graceful degradation.

### Step 3 — Run quick benchmark
```bash
python -m app.cli quick-test
```
Expected output: GPU Insight Score (0-100), benchmark results table, diagnosis summary.
Without CUDA binary: CPU baselines run; GPU benchmarks show SKIPPED/NOT_VALIDATED.
This is expected behavior, not an error.

### Step 4 — Run memory bandwidth benchmark
```bash
python -m app.cli benchmark --test memory
```
Expected output: H2D/D2H/D2D bandwidth in GB/s.
Without GPU: returns NOT_VALIDATED or SKIPPED with informative message.

### Step 5 — Run PCIe analysis
```bash
python -m app.cli benchmark --test pcie
```
Expected output: PCIe transfer bandwidth, comparison to theoretical maximum.
`pcie` is an alias for `memory_bandwidth` — explains how H2D/D2H maps to PCIe throughput.

### Step 6 — Run GEMM benchmark
```bash
python -m app.cli benchmark --test gemm
```
Expected output: GFLOP/s for naive vs. tiled GEMM, correctness check, speedup ratio.
Without GPU: returns SKIPPED — demonstrates graceful degradation.

### Step 7 — Show diagnosis
```bash
python -m app.cli diagnose --latest
```
Expected output: Evidence-based diagnosis findings, each with severity, category, evidence
string citing specific measured values, and recommendation.
Without prior sessions: "No sessions found" — run quick-test first.

### Step 8 — Generate demo report (no GPU required)
```bash
python -m app.cli demo-report
```
Expected output: Generates 4 sample report files in output/:
- output/gpu_insight_report_sample.json
- output/gpu_insight_report_sample.md
- output/gpu_insight_report_sample.html
- output/gpu_insight_report_sample.xlsx (if openpyxl installed)
Final line: "GPU Insight Lab Demo Report Generated"

---

## 5-Minute Complete Demo Flow

Extends the 3-minute flow with these additional steps:

### Step 9 — Run full test suite
```bash
python -m app.cli full-test
```
Expected output: All benchmark categories (CPU baselines + CUDA kernels if available),
complete score breakdown across 6 categories, session saved to SQLite.

### Step 10 — Export HTML report from latest session
```bash
python -m app.cli export --latest --format html
```
Expected output: Path to generated HTML report. Open in browser to show self-contained
report with inline CSS — works in air-gapped environments, no CDN.

### Step 11 — Export Excel report
```bash
python -m app.cli export --latest --format xlsx
```
Expected output: Path to .xlsx with 7 sheets: Summary, System, GPU, Benchmarks,
Diagnosis, Score, Raw. Requires openpyxl.

### Step 12 — Show session history
```bash
python -m app.cli history
```
Expected output: Table of stored sessions with ID, name, timestamp, GPU name, score, status.

### Step 13 — Compare sessions
```bash
python -m app.cli compare --session-a 1 --session-b 2
```
Expected output: Side-by-side comparison with delta % per benchmark. Useful for
regression detection after driver updates or configuration changes.

---

## Files to Show an Interviewer (Recommended Order)

1. **README.md** — Feature status table (Implemented / Partial / ROADMAP / NOT_VALIDATED)
2. **docs/CODE_COVERAGE_AUDIT.md** — Honest implemented/partial/roadmap breakdown
3. **app/cli.py** — 10-command CLI structure with argparse, exit codes, graceful degradation
4. **diagnosis/engine.py** — Evidence-based rules, every finding requires non-empty evidence
5. **benchmarks/schemas.py** — MemoryBenchmarkResult + KernelBenchmarkResult dataclass schemas
6. **native/cuda/gemm_tiled.cu** — CUDA kernel: shared-memory tiled GEMM, sm_75–sm_90
7. **native/hip/reduction_hip.cpp** — HIP portability demo: reduction ported from CUDA
8. **storage/database.py** — SQLite with WAL mode, schema migrations, session comparison
9. **output/gpu_insight_report_sample.html** — Open in browser: self-contained HTML report

---

## Demo Commands (Copy-Paste Ready)

```bash
python -m app.cli --help
python -m app.cli system-info
python -m app.cli quick-test
python -m app.cli benchmark --test memory
python -m app.cli benchmark --test pcie
python -m app.cli benchmark --test gemm
python -m app.cli diagnose --latest
python -m app.cli demo-report
python -m app.cli full-test
python -m app.cli export --latest --format html
python -m app.cli export --latest --format xlsx
python -m app.cli history
python -m app.cli compare --session-a 1 --session-b 2
```

PowerShell one-shot demo script:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_interview_demo.ps1
```

---

## Expected Output Per Command

| Command | Success looks like | SKIPPED/NOT_VALIDATED looks like |
|---------|-------------------|----------------------------------|
| `system-info` | GPU name, driver, PCIe gen/width, tools table | "No NVIDIA GPU detected" — acceptable |
| `quick-test` | Score printed, benchmark rows, diagnosis summary | CPU baselines pass; GPU rows show SKIPPED |
| `benchmark --test memory` | H2D/D2H/D2D bandwidth in GB/s | "Test: memory_bandwidth  Mean: N/A  Error: ..." |
| `benchmark --test pcie` | PCIe transfer GB/s | SKIPPED with note "native binary not built" |
| `benchmark --test gemm` | GFLOP/s + correctness_pass=True | SKIPPED — not a failure |
| `diagnose --latest` | Findings list with evidence strings | "No sessions found — run quick-test first" |
| `demo-report` | 4 file paths printed, final summary line | XLSX line shows FAILED if openpyxl missing |

**Key talking point**: SKIPPED and NOT_VALIDATED are honest engineering choices, not failures.
The tool is designed to work gracefully without hardware.

---

## How to Explain Each Component

### CUDA Benchmarks
"I wrote 7 native CUDA C++ kernels: vector add (memory-bound), reduction (parallel tree),
transpose (coalesced access), GEMM naive vs. tiled (shared memory optimization), memory
bandwidth (peak device BW), stream pipeline (async overlap), and image grayscale (2D tiling).
Each kernel measures timing with CUDA Events, computes bandwidth or GFLOP/s, checks
correctness against CPU reference, and reports coefficient of variation across repeats.
High CV triggers the HIGH_VARIANCE diagnosis rule."

### HIP Portability
"I ported three kernels from CUDA to HIP: vector_add_hip, reduction_hip, gemm_naive_hip.
HIP uses nearly identical syntax — __global__, threadIdx, blockDim — but compiles to
both NVIDIA (via NVCC) and AMD (via hipcc). The tool marks these results NOT_VALIDATED
on AMD hardware because I don't have an AMD GPU to validate against. This honesty about
validation status is intentional engineering practice."

### PCIe Analysis
"PCIe bandwidth is the first system bottleneck to check in GPU workflows. Before blaming
the kernel, you need to know if data transfers are the bottleneck. The tool measures H2D
(host-to-device), D2H (device-to-host), and D2D (device-to-device) bandwidth and compares
to theoretical PCIe maximum. PCIE_BOTTLENECK is triggered if link gen/width is below maximum."

### Diagnosis Engine
"Every diagnosis finding requires a non-empty evidence string citing specific measured
values. This is an explicit design constraint — no speculation allowed. Rules that can't
obtain sufficient data return None rather than a low-confidence finding. This mirrors
professional hardware validation practice where you don't file a bug without evidence."

### Report Studio
"The HTML report is fully self-contained: inline CSS, no external CDN dependencies.
This means it works in air-gapped lab environments where developers can't access the
internet. Excel reports use openpyxl with 7 sheets, freeze panes, auto-filter, and
severity color coding. No merged cells so the data is programmatically accessible."

### SQLite History
"Sessions are stored in SQLite with WAL mode for write concurrency and schema migrations
for forward compatibility. The compare command shows delta % between sessions — useful for
regression detection after driver updates, kernel changes, or hardware swaps. Schema
version is tracked in the database itself."

---

## If No CUDA Toolkit / GPU Available

- `demo-report` generates full sample reports with mock data — no GPU required
- `system-info` works fully without GPU (shows "No NVIDIA GPU detected")
- `diagnose --latest` works if any session exists in the database
- `history` and `export` work without GPU
- Benchmarks return `SKIPPED` or `NOT_VALIDATED` — these are informative, not crashes
- The GPU Insight Score still computes with reduced confidence (LOW or MEDIUM)

**State clearly in the interview**: "The tool is designed to work gracefully without hardware.
SKIPPED means the hardware or binary is unavailable, not that the code is broken."

---

## NVIDIA Interview Talking Points

- **CUDA kernel depth**: 7 native kernels covering memory-bound, compute-bound, and
  latency-hiding patterns — vector add, reduction, transpose, GEMM ×2, memory bandwidth,
  stream pipeline. Each with CUDA Events timing, correctness validation, CV statistics.
- **Nsight integration**: The profilers/ module includes NsightSystemsProfiler and
  NsightComputeProfiler stubs that launch `nsys` and `ncu` as subprocesses and parse
  their output. The tool_status collector detects nsys/ncu availability and reports it.
- **PCIe analysis**: H2D/D2H/D2D bandwidth measurement maps directly to PCIe Gen/width
  analysis. The PCIE_BOTTLENECK rule flags when the GPU is running at lower link than max.
- **sm_75 to sm_90 coverage**: Native kernels compiled for Turing, Ampere, Ada, and
  Hopper compute capabilities — demonstrates awareness of architecture-specific SM targets.

---

## AMD Interview Talking Points

- **HIP portability**: Three kernels ported from CUDA to HIP (vector_add, reduction,
  gemm_naive). Demonstrates understanding of HIP's CUDA-compatible syntax and the
  CUDA→HIP porting workflow.
- **Wavefront vs. warp awareness**: HIP kernels use wavefront-size-aware block dimensions.
  The HIP guide in docs/ explains warp (32 threads on NVIDIA) vs. wavefront (64 threads
  on AMD GCN/RDNA) and how this affects block sizing.
- **ROCm detection**: amd_collector.py detects rocminfo availability and AMD GPU presence.
  The AMD_NOT_VALIDATED diagnosis rule fires when an AMD GPU is detected, with an honest
  explanation that results are not validated on AMD hardware.
- **NOT_VALIDATED honesty**: Rather than claiming HIP results are validated on AMD, the
  tool explicitly marks them NOT_VALIDATED. This demonstrates engineering integrity —
  important for validation roles at AMD.

---

## GPU Validation Interview Talking Points

- **Evidence policy**: Every diagnosis finding requires a non-empty evidence string citing
  specific measured values. Rules that cannot obtain sufficient data return None. This is
  exactly the "no bug without repro" standard used in hardware validation.
- **Correctness checking**: Every CUDA benchmark compares GPU output to a CPU NumPy
  reference. CORRECTNESS_FAILURE diagnosis rule triggers on mismatch. This is analogous
  to functional verification in hardware validation.
- **Regression comparison**: The compare command shows delta % between any two sessions.
  Useful for catching regressions after driver updates, kernel changes, or hardware swaps.
- **Session history**: SQLite persistence with schema migrations means validation data is
  reproducible and auditable. Sessions include timestamp, GPU name, driver version, and
  full benchmark results — the minimum required for a validation report.

---

## Performance Engineering Interview Talking Points

- **Roofline model awareness**: Benchmarks classify workloads as memory-bound (vector add,
  transpose) or compute-bound (GEMM). The GPU Insight Score weights PCIe/memory and
  compute throughput separately, reflecting roofline model thinking.
- **Bandwidth analysis**: Memory bandwidth benchmark measures peak device bandwidth and
  PCIe H2D/D2H bandwidth separately. LOW_MEMORY_BANDWIDTH rule triggers when measured BW
  is significantly below the GPU model's expected value.
- **Bottleneck classification**: The diagnosis engine distinguishes PCIe bottleneck,
  thermal throttle, low memory bandwidth, low compute throughput, and high variance as
  separate categories — the standard bottleneck classification hierarchy for GPU performance.
- **Profiling workflow**: The profilers/ module integrates nvidia-smi power/temperature
  monitoring, Nsight Systems timeline profiling, and Nsight Compute kernel profiling.
  Even as stubs, they demonstrate the correct profiling pipeline: run → collect → parse → report.

---

## Known Limitations

- **No GPU = no real benchmark numbers**: Without an NVIDIA GPU and the native binary,
  CUDA kernel benchmarks return SKIPPED. CPU baselines still run via NumPy.
- **HIP not validated on AMD hardware**: HIP kernels compile but results are marked
  NOT_VALIDATED because testing was done without AMD GPU hardware.
- **Excel requires openpyxl**: `pip install openpyxl` — the tool gracefully skips Excel
  generation if not installed and continues with other formats.
- **HTML requires jinja2**: `pip install jinja2` — similarly skipped if not installed.
- **Single GPU only**: Multi-GPU topologies (NVLink, SLI) are not covered.
- **FP32 only**: Native benchmarks use 32-bit float. Tensor Core (FP16/BF16/INT8)
  performance is not measured.
- **Windows timing jitter**: GPU kernel timing on Windows may have higher CV than Linux
  due to WDDM scheduling.
- **GUI is partial**: PySide6 GUI exists but several pages are stubs requiring PySide6.

---

## Next Version Roadmap

From `docs/12_WEEK_CUDA_JOB_ROADMAP.md`:

- **Week 5-6**: Softmax, layer norm, GELU kernels with FP16/BF16 support
- **Week 7-8**: Flash Attention kernel skeleton, INT8 quantization benchmark
- **Week 9-10**: PyTorch custom extension (CUDA extension via torch.utils.cpp_extension)
- **Week 11-12**: TensorRT plugin skeleton, cuBLAS/cuFFT benchmarks
- Multi-machine session import (compare sessions from different machines)
- Batch execution mode (run benchmark suite across multiple configs unattended)
- Company-specific report templates (NVIDIA, AMD, HPC lab formats)
- Streamlit web dashboard for session browsing
