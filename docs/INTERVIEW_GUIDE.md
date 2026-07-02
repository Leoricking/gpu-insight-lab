# GPU Insight Lab — Interview Guide

## How to Present This Project

GPU Insight Lab demonstrates practical capabilities directly relevant to NVIDIA, AMD, and
GPU-adjacent software engineering roles. This guide explains what each part of the codebase
demonstrates, how to discuss it in interviews, and what follow-up questions to expect.

---

## Roles This Project Supports

- **GPU Software Engineer** (NVIDIA, AMD, Intel Arc teams)
- **CUDA Developer / GPU Performance Engineer**
- **Systems Software Engineer** (driver-adjacent, telemetry, benchmarking infrastructure)
- **ML Infrastructure Engineer** (GPU-side tooling, profiling, performance diagnostics)
- **Developer Tools Engineer** (profiler integration, developer experience)
- **Platform Software Engineer** (cross-vendor GPU support)

---

## Core Technical Claims and Supporting Evidence

### Claim 1: "I understand GPU memory hierarchy and bandwidth limits."

**Evidence in codebase**:
- `native/cuda/memory_bandwidth.cu`: streaming read/write kernel with correct stride
  patterns to avoid cache effects
- `native/cuda/transpose.cu`: demonstrates coalesced vs. non-coalesced access patterns
- `diagnosis/rules.py` → `LOW_MEMORY_BANDWIDTH` rule: expected bandwidth table, GB/s
  calculation from timing
- `docs/BENCHMARK_METHODOLOGY.md`: explanation of what "effective bandwidth" measures
  vs. theoretical peak

**How to discuss it**:
> "The memory bandwidth benchmark uses a streaming access pattern with intentionally large
> arrays to avoid L2 cache hits. The expected bandwidth table in the diagnosis engine lets
> me flag when a card is running significantly below its rated HBM2/GDDR6X peak — which
> is often the first sign of thermal throttling or ECC overhead."

**Follow-up questions to prepare for**:
- What is the Roofline model and how does it relate to your bandwidth benchmarks?
- How does ECC mode affect effective memory bandwidth?
- Why does your transpose kernel use shared memory tiling?

---

### Claim 2: "I can write CUDA kernels with shared memory optimization."

**Evidence in codebase**:
- `native/cuda/gemm_tiled.cu`: tiled matrix multiply with shared memory blocking
- `native/cuda/reduction.cu`: multi-pass reduction using shared memory + warp shuffles
- `native/include/cuda_check.cuh`: proper CUDA error checking macro used throughout

**How to discuss it**:
> "The tiled GEMM kernel loads tiles of A and B into shared memory to reuse each element
> across the tile dimension, reducing global memory accesses from O(M×N×K) loads to
> O(M×N×K / TILE_SIZE) per SM. The reduction kernel uses `__shfl_down_sync` for the final
> warp-level reduction to eliminate the last shared memory barrier."

**Follow-up questions to prepare for**:
- What is occupancy and when does increasing it not improve performance?
- How do shared memory bank conflicts affect your tiled GEMM?
- Why does your reduction use two-pass rather than a single atomic add?

---

### Claim 3: "I understand the CUDA profiling toolchain."

**Evidence in codebase**:
- `profilers/nsight_systems.py`: subprocess wrapper for `nsys profile`, parses SQLite
  output from Nsight Systems
- `profilers/nsight_compute.py`: subprocess wrapper for `ncu`, parses `--csv` output,
  handles `ncu --target-processes` options
- `profilers/nvidia_smi_monitor.py`: continuous polling of `nvidia-smi dmon` for power,
  temperature, SM utilization, memory utilization
- `benchmarks/native_runner.py`: JSON interface between Python orchestrator and native binary

**How to discuss it**:
> "I separated profiling concerns by tool. Nsight Systems gives you the timeline — where
> kernels overlap, where PCIe transfers happen, where the CPU is stalled. Nsight Compute
> gives you per-SM counters — achieved occupancy, memory throughput, pipe utilization.
> nvidia-smi dmon gives you board-level telemetry at low frequency. These answer different
> questions and I didn't conflate them."

**Follow-up questions to prepare for**:
- What is the difference between SM utilization (nvidia-smi) and achieved occupancy (ncu)?
- When would you use `--kernel-id` in Nsight Compute vs. profiling all kernels?
- How does CUPTI differ from Nsight Systems as a profiling interface?

---

### Claim 4: "I know how to handle cross-vendor GPU support responsibly."

**Evidence in codebase**:
- `collectors/amd_collector.py`: every AMDGPUInfo always sets `validation_status = "NOT_VALIDATED"`
- `diagnosis/rules.py` → `AMD_NOT_VALIDATED`: fires on any AMD detection as an INFO notice
- `docs/CUDA_VS_HIP.md`: warp vs. wavefront table, API mapping, portability implications
- `docs/DIAGNOSIS_RULES.md`: explains that AMD findings are explicitly not diagnosed

**How to discuss it**:
> "The AMD collector gathers what data it can from rocm-smi and rocminfo, but it stamps
> every result as NOT_VALIDATED and surfaces an INFO notice rather than applying NVIDIA
> performance baselines to AMD hardware. I think it's better engineering to say 'I don't
> have validated baselines for this hardware' than to generate numbers that look authoritative
> but aren't. The HIP portability guide in the docs shows I understand the differences —
> wavefront width, warp shuffle semantics, the GCN vs. RDNA architectural split."

---

### Claim 5: "I can build production-quality diagnostic tools, not just scripts."

**Evidence in codebase**:
- `storage/database.py` + `storage/migrations.py`: SQLite with versioned migrations, WAL
  mode, `PRAGMA foreign_keys = ON`, `compare_sessions()` for delta analysis
- `reports/excel_report.py`: 7 sheets, freeze_panes, auto_filter, auto-width columns,
  severity-colored cells, no merged cells (for programmatic access)
- `reports/templates/report.html.j2`: self-contained HTML (inline CSS), works without
  internet connection, Jinja2-rendered
- `app/config.py`: `AppConfig` dataclass with `load_config()`/`save_config()`, default
  in `~/.gpu_insight_lab/config.json`
- `tests/test_smoke.py`, `tests/test_diagnosis.py`, `tests/test_storage.py`: 35+ tests
  covering imports, CLI, diagnosis rules, storage round-trips

**How to discuss it**:
> "The difference between a benchmark script and a diagnostic tool is persistence,
> reproducibility, and interpretability. Sessions are stored in SQLite with schema
> versioning so I can compare GPU behavior across driver updates. The HTML report is
> self-contained — no CDN, no JavaScript framework — so it works in air-gapped
> environments. The Excel report avoids merged cells specifically so downstream
> data consumers (pandas, openpyxl readers) can process it without cell-parsing logic."

---

### Claim 6: "I can integrate Python tools with native CUDA binaries cleanly."

**Evidence in codebase**:
- `benchmarks/native_runner.py`: `find_executable()` searches multiple paths; all subprocess
  calls use list arguments (no `shell=True`); JSON stdout protocol; timeout handling
- `native/cuda/benchmark_main.cu`: `--device-info`, `--quick`, `--full`, `--test` modes;
  writes JSON to stdout or file; error output to stderr
- `native/CMakeLists.txt`: cmake 3.18+, CUDA architectures sm_75 through sm_90, links
  CUDA::cudart + CUDA::cuda_driver

**How to discuss it**:
> "The Python-native interface uses JSON over stdout. The Python side calls the binary
> with `subprocess.run([exe, '--quick', '--repeat', '10'], capture_output=True, timeout=120)`.
> stdout is JSON, stderr is diagnostic text. This means the native binary can be replaced,
> updated, or run standalone without touching Python. The subprocess call never uses
> `shell=True` to avoid injection vulnerabilities in paths with spaces."

---

## Design Decision Discussion Points

These are decisions you made explicitly and should be able to defend:

### "Why SQLite and not PostgreSQL?"

SQLite requires no server, no configuration, and no network. A benchmarking tool runs on
the user's local machine; adding a database server dependency would prevent installation
in restrictive environments (air-gapped HPC clusters, secure labs). SQLite with WAL mode
supports concurrent readers and is sufficient for the session-history use case.

### "Why PySide6 instead of tkinter or Electron?"

PySide6 (Qt6) provides native OS widgets (not drawn on a canvas), QThread for
non-blocking I/O, and a professional appearance. tkinter is too limited for the multi-page
sidebar navigation pattern. Electron ships a full Chromium instance (150+ MB) for a tool
that doesn't need a browser engine. PySide6 is the standard choice for Python-native
GPU tool GUIs in the NVIDIA ecosystem (e.g., NVIDIA NSight IDE plugins use Qt).

### "Why does every collector return partial data instead of raising?"

Diagnostic tools are expected to work on partially-configured systems. A tool that crashes
because pynvml isn't installed is useless as a diagnostic tool — the missing library might
be part of the problem. Graceful degradation means GPU Insight Lab can identify "CUDA toolkit
not found" and "driver missing" as diagnosis findings rather than as tool failures.

### "Why is the diagnosis engine rule-based instead of ML-based?"

Evidence-based rules are transparent and auditable. An ML model that says "this looks like
a throttling pattern" without explaining what data triggered it is not useful for an
engineer trying to reproduce or fix an issue. Every DiagnosisResult has a non-empty evidence
string that cites specific measured values. This is the same philosophy behind Nsight
Compute's guided analysis: rules that explain themselves.

---

## Numbers to Know

Be prepared to discuss these specific numbers from the codebase:

| Number | Context |
|--------|---------|
| 32 | NVIDIA warp size (all architectures) |
| 64 | AMD wavefront size (GCN/RDNA1/RDNA2) |
| 83°C | WARNING thermal threshold in diagnosis rules |
| 90°C | CRITICAL thermal threshold |
| 10% | High variance (CV) threshold for HIGH_VARIANCE rule |
| 3 | Warmup iterations before measurement |
| 10 | Default measurement repetitions |
| 6 | Number of score categories in GPU Insight Score |
| 100 | Maximum GPU Insight Score |
| 9 | Number of diagnosis rules in v0.1.0 |
| 7 | Number of native CUDA benchmark kernels |
| 7 | Number of sheets in Excel report |

---

## Common Interview Questions and Suggested Responses

**Q: "What is the difference between memory bandwidth and compute throughput as bottlenecks?"**

A: Memory bandwidth measures how fast the GPU can move data from GDDR6/HBM to the compute
units. Compute throughput measures how fast the GPU can execute floating-point operations.
A kernel is memory-bound if the ratio of FLOP/s to bytes accessed is below the
machine's arithmetic intensity threshold (the ridge point on the Roofline model). Most
real-world workloads (ML inference with large activation tensors) are memory-bound, not
compute-bound.

**Q: "Why does your benchmark use population standard deviation instead of sample?"**

A: Because I'm measuring the variance in the observed benchmark runs specifically, not
estimating the variance of some larger theoretical population. The 10 runs are the complete
dataset for the session. Using N-1 (Bessel's correction) would be appropriate if I were
trying to estimate what the standard deviation would be over thousands of runs; here I just
want to characterize what I actually observed.

**Q: "What would you add to make this production-ready for a data center?"**

A: Three things primarily: (1) Remote telemetry aggregation — sessions pushed to a central
time-series database (InfluxDB or Prometheus) so fleet-level trends are visible across
thousands of GPUs; (2) Continuous monitoring mode with alert thresholds, not just on-demand
benchmarks; (3) Integration with the GPU's SMBIOS/IPMI data for out-of-band health
monitoring that doesn't require a running driver.

---

## Portfolio Presentation Tips

1. **Lead with the architecture diagram** from `docs/ARCHITECTURE.md`. It shows you think
   in layers (collection → analysis → storage → reporting) rather than writing a monolithic
   script.

2. **Show the evidence policy** from `docs/DIAGNOSIS_RULES.md`. The rule "no finding without
   evidence, no diagnosis without data" is directly analogous to how NVIDIA's internal
   performance triage tools work.

3. **Run a live demo** — `python -m app.cli system-info --json` takes seconds and produces
   structured JSON that shows the graceful degradation pattern (partial data when hardware
   is absent, not a crash).

4. **Discuss the NOT_VALIDATED policy** for AMD. It shows intellectual honesty: you
   know the limits of your tool and you communicate them explicitly rather than generating
   misleading numbers.

5. **Reference specific CUDA architecture differences** — warp size, shared memory banking,
   the difference between SM utilization (a coarse metric) and achieved occupancy (a precise
   micro-architectural metric). These demonstrate depth beyond "I wrote some GPU code."
