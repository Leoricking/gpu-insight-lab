> **Live demo**: See [docs/DEMO_GUIDE.md](DEMO_GUIDE.md) and run `scripts/run_interview_demo.ps1` for a guided walkthrough.
> **現場展示**：請先看 [docs/DEMO_GUIDE.md](DEMO_GUIDE.md) 並執行 `scripts/run_interview_demo.ps1`。

# GPU Insight Lab — Interview Guide

## Project Positioning

**Target Role**: GPU Software / CUDA Performance / HPC Validation + Acceleration Engineer

This project is designed for an engineer with PCIe, HPC, driver, validation, and system
debugging background who is transitioning into CUDA/HIP GPU performance engineering.

**本專案不是把自己包裝成純 AI 工程師，而是定位成「懂硬體、PCIe、系統驗證與效能診斷的
CUDA / GPU Performance Engineer」。**

GPU Insight Lab 展示的不只是會寫 CUDA kernel，而是能把 PCIe 傳輸分析、benchmark 方法論、
profiling 整合、正確性驗證、證據式診斷、版本回歸比較與專業報告輸出整合成完整工程流程。

---

## Feature Status — What Is and Is Not Implemented

Use this table to answer "what does GPU Insight Lab actually do?" in interviews.
Never claim ROADMAP features as completed.

| Feature | Status | Requires |
|---------|--------|---------|
| System Inspector (CPU/GPU/PCIe/CUDA telemetry) | Implemented | None (Python only) |
| Memory Benchmark (H2D/D2H/D2D) | Implemented | CUDA GPU + native binary |
| Kernel Lab (vector_add, reduction, transpose, gemm×2, memory, streams) | Implemented | CUDA GPU + native binary |
| Evidence-based Diagnosis Engine (9 rules) | Implemented | None |
| GPU Insight Score (0-100, 6 categories) | Implemented | None |
| Multi-format Reports (JSON/CSV/MD/HTML/Excel) | Implemented | jinja2 + openpyxl for HTML/Excel |
| SQLite Session History + delta comparison | Implemented | None |
| PySide6 GUI | Partial — stub pages | PySide6 |
| CLI Automation (10 commands) | Implemented | None |
| CUDA to HIP Portability Demo | Partial — NOT_VALIDATED | AMD GPU + ROCm for real run |
| prefix_sum (Blelloch scan) | Skeleton / NOT_VALIDATED | |
| convolution_2d | Skeleton / NOT_VALIDATED | |
| softmax, layer_norm, GELU | **ROADMAP** | |
| Flash Attention | **ROADMAP** | |
| INT8 quantization | **ROADMAP** | |
| PyTorch extension / TensorRT plugin | **ROADMAP** | |
| cuFFT / cuBLAS full benchmark | **ROADMAP** | |
| Streamlit dashboard | **ROADMAP** | |
| Multi-machine import / batch execution | **ROADMAP** | |
| Company report templates | **ROADMAP** | |

---

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

---

## Job-Aligned Portfolio Projects

### 1. CUDA Performance Lab

A progression from CPU baseline → naive CUDA → optimized CUDA → Nsight profiling:

| Kernel | CPU Baseline | Naive CUDA | Optimized | Profiling |
|--------|-------------|------------|-----------|-----------|
| vector_add | NumPy | ✓ | bandwidth-bound | Nsight Compute |
| matrix_transpose | NumPy | non-coalesced | tiled + shared | Nsight Compute |
| reduction | NumPy sum | shared mem | warp shuffle | Nsight Compute |
| prefix_sum | NumPy cumsum | naive scan | Blelloch | Nsight Compute |
| matmul | BLAS | naive | tiled + register | Nsight Compute |
| conv2d | SciPy | direct | shared tiled | Nsight Compute |

Each benchmark includes: bottleneck analysis, Nsight Compute `.ncu-rep`, and measured
comparison table (GFLOP/s or GB/s).

**Interview talking point**: "This lab shows I can implement, benchmark, and optimize CUDA
kernels from scratch — and that I use the profiler to drive decisions, not guessing."

---

### 2. GPU PCIe Bandwidth Benchmark Tool

Measures H2D, D2H, D2D bandwidth across transfer sizes; compares pinned vs. pageable memory;
generates CSV / JSON / HTML / Excel report with PCIe bottleneck evidence.

| Measurement | Transfer Sizes | Output |
|-------------|---------------|--------|
| H2D bandwidth | 1 MB – 1 GB | GB/s vs. size chart |
| D2H bandwidth | 1 MB – 1 GB | GB/s vs. size chart |
| D2D bandwidth | 1 MB – 1 GB | GB/s vs. size chart |
| Pinned vs. pageable | 64 MB | speedup ratio |

**Interview talking point**: "PCIe bandwidth is the first thing I check when a GPU workflow
underperforms. This tool gives me evidence — not intuition — about where the bottleneck is."

GPU Insight Lab's `PCIE_BOTTLENECK` diagnosis rule and PCIe collector complement this tool.

---

### 3. CUDA → HIP Portability Demo

Ports reduction and tiled GEMM from CUDA to HIP with full API mapping, porting notes,
and architectural comparison.

| Item | CUDA | HIP |
|------|------|-----|
| Memory allocation | `cudaMalloc` | `hipMalloc` |
| Warp shuffle | `__shfl_down_sync` | `__shfl_down` |
| Warp size | 32 | 64 (GCN/RDNA) |
| Build | `nvcc` | `hipcc` |
| Profiler | Nsight Compute | ROCProfiler |

**Interview talking point**: "Porting these kernels forced me to confront the
warp-32 vs. wavefront-64 difference concretely, not just as a trivia fact."

GPU Insight Lab's `docs/CUDA_VS_HIP.md` and AMD collector are the reference for
cross-vendor detection and NOT_VALIDATED policy.

---

## AI Inference Kernel Roadmap (Kernel Lab Future Work)

| Kernel | Status | Notes |
|--------|--------|-------|
| softmax | Roadmap | numerically stable (subtract max before exp) |
| layer normalization | Roadmap | fused mean + variance |
| GELU | Roadmap | exact and approximate variants |
| 2D convolution | Roadmap | shared memory tiled |
| prefix sum / scan | Roadmap | Blelloch work-efficient |
| Flash Attention | Future | multi-head attention kernel |
| INT8 quantization | Future | inference optimization |
| PyTorch extension | Optional | `torch.utils.cpp_extension` |
| TensorRT plugin | Future | inference backend integration note |

---

## CUDA Interview Question Bank

### CUDA Fundamentals

**Q: What are grid, block, and thread?**

A: A grid is the collection of all thread blocks launched by a single kernel call. A block is
a group of threads that can cooperate via shared memory and synchronize with `__syncthreads()`.
A thread is the individual execution unit. The mapping is: GPU device executes a grid of blocks;
each SM executes one or more blocks; each block contains up to 1024 threads organized in warps
of 32.

**Q: What is a warp?**

A: A warp is a group of 32 threads that execute instructions in lockstep (SIMT — Single
Instruction Multiple Threads). All 32 threads in a warp execute the same instruction at the
same time. Warp divergence occurs when threads in the same warp take different execution paths
(different branches of an `if` statement), causing serialization of both paths.

**Q: What is global memory?**

A: Global memory is the GPU's main DRAM (GDDR6X or HBM2). It has the largest capacity
(e.g., 24 GB on RTX 3090), highest latency (~400–800 cycles), and highest bandwidth
(e.g., 936 GB/s on A100 HBM2e). All threads from all blocks can read and write global memory.
Coalesced access (contiguous aligned 128-byte transactions) is essential for performance.

**Q: What is shared memory?**

A: Shared memory is an on-chip, low-latency (≈ 4 cycles) memory space shared by all threads
within the same block. It is programmer-managed (not a cache). It is used to stage data that
will be reused multiple times (e.g., tiles in GEMM). Typical capacity: 48–96 KB per SM
on Ampere. Bank conflicts arise when multiple threads access the same bank simultaneously
(except broadcast).

**Q: What is memory coalescing?**

A: Coalescing is when threads in a warp access a contiguous, aligned 128-byte region of global
memory, allowing the hardware to service all 32 accesses in a single DRAM transaction.
Non-coalesced access (strided or random) requires multiple transactions, multiplying global
memory traffic and reducing effective bandwidth.

**Q: What is occupancy?**

A: Occupancy is the ratio of active warps per SM to the maximum possible warps per SM.
Higher occupancy enables the GPU to hide memory latency by switching to ready warps while
others wait for data. Occupancy is limited by register usage, shared memory usage, and
block size. Achieved occupancy (measured by Nsight Compute) ≠ theoretical occupancy
(calculated from resource limits).

**Q: What is warp divergence?**

A: Warp divergence occurs when threads within the same warp execute different code paths due
to a conditional branch. SIMT hardware must serialize both paths: threads taking the `if`
branch execute while `else` threads are masked, then vice versa. This halves throughput in
the worst case. Divergence can be reduced by restructuring data so threads in the same warp
take the same path.

---

### Performance Optimization

**Q: How do you optimize matrix transpose?**

A: Naive transpose is non-coalesced on writes (or reads). The fix is to load a tile into
shared memory (coalesced reads from global), then write the tile transposed to output
(coalesced writes). Add 1-element padding to the shared memory tile to avoid bank conflicts
on the diagonal. This typically achieves 80–90% of peak memory bandwidth.

**Q: How do you optimize reduction?**

A: Use shared memory for per-block partial sums. For the final warp, replace `__syncthreads()`
with `__shfl_down_sync` warp shuffle instructions (no synchronization needed within a warp).
Use a two-pass approach: first reduce to one value per block, then reduce block results. Avoid
atomic operations in the hot path; use them only for the final cross-block reduction.

**Q: When should shared memory be used?**

A: Use shared memory when data will be accessed multiple times by threads in the same block
(reuse justifies the staging cost). Examples: GEMM tiles (each element reused TILE_SIZE times),
convolution filters, reduction partial sums. Do not use shared memory for data accessed only
once — the overhead of loading into shared memory outweighs the benefit.

**Q: When should atomic operations be avoided?**

A: Avoid atomics in the hot path of a kernel. Multiple threads competing for the same atomic
location serialize, destroying parallelism. Instead, use a reduction pattern: compute per-thread
partial results in registers, reduce within warps using shuffles, reduce within blocks using
shared memory, then use one atomic per block to aggregate the final result.

**Q: How do you overlap data transfer and compute?**

A: Use CUDA streams. Issue `cudaMemcpyAsync` on stream A and a kernel on stream B simultaneously.
The hardware can overlap H2D/D2H transfers (via the copy engine) with kernel execution (on
the compute engine). Requires pinned (page-locked) host memory for async transfers. Verify
overlap with Nsight Systems timeline.

**Q: Why is pinned memory faster than pageable memory?**

A: Pageable host memory can be paged out by the OS. Before a DMA transfer, the CUDA driver
must lock the pages (pin them) to prevent paging during the transfer — adding latency.
Pinned memory (`cudaMallocHost`) is always page-locked, so the DMA transfer starts immediately
and the PCIe bus is used at full bandwidth. Also, pinned memory allows async (non-blocking)
memcpy.

**Q: How do you determine whether a workload is memory-bound or compute-bound?**

A: Use the Roofline model. Plot measured GFLOP/s against arithmetic intensity (FLOP / byte).
If the point falls on the memory bandwidth roof (left side), the workload is memory-bound.
If it falls on the compute roof (right side), it is compute-bound. In Nsight Compute, look at
the "Memory Throughput" and "SM Throughput" metrics — whichever is closer to 100% is the
bottleneck.

---

### Debug / Profiling

**Q: How do you debug CUDA kernels?**

A: First, use `cuda-memcheck` or `compute-sanitizer` to detect out-of-bounds accesses, race
conditions, and uninitialized memory. Add a `CUDA_CHECK` macro after every CUDA API call.
Use `printf` inside kernels (limited to Fermi and later) for small-scale debugging. For complex
bugs, use `cuda-gdb` (Linux) or Nsight Visual Studio (Windows). Compare GPU output to a CPU
reference for correctness validation.

**Q: What is the difference between cudaGetLastError and cudaDeviceSynchronize?**

A: `cudaGetLastError()` retrieves the last CUDA error on the calling thread's error stack
without blocking. `cudaDeviceSynchronize()` blocks the CPU until all previously issued CUDA
operations on the device complete — and returns any async error that occurred. Use
`cudaDeviceSynchronize()` after kernel launches during debugging to catch kernel-launch
errors that would otherwise be masked by async execution.

**Q: What is the difference between Nsight Systems and Nsight Compute?**

A: Nsight Systems gives a system-level timeline view: where kernels run, where PCIe transfers
happen, where the CPU is stalled, how streams overlap. It answers "what is running when."
Nsight Compute gives per-kernel SM-level counter data: achieved occupancy, memory throughput,
pipe utilization, warp stall reasons. It answers "why is this kernel slow." Use Nsight Systems
first to find the bottleneck kernel, then Nsight Compute to analyze it.

**Q: How do you interpret memory throughput, occupancy, and kernel duration?**

A: Memory throughput (GB/s) shows how much of the GPU's peak DRAM bandwidth is utilized.
A value near theoretical peak (e.g., 900 GB/s on A100) means the kernel is memory-bound.
Achieved occupancy (%) shows how many warps are active per SM vs. maximum; low occupancy
(< 50%) can indicate register or shared memory pressure. Kernel duration (ms) is the wall
clock time; compare across implementations to quantify speedup.

---

### NVIDIA / AMD Alignment

**Q: What is the difference between CUDA and HIP?**

A: CUDA is NVIDIA's proprietary GPU programming platform (nvcc compiler, CUDA runtime, PTX ISA).
HIP is AMD's open-source GPU programming API that mirrors CUDA syntax, allowing code to run
on both AMD (via ROCm) and NVIDIA GPUs. HIP code uses `hipMalloc`, `hipMemcpy`, `hipLaunchKernelGGL`.
The `hipify` tool automates CUDA → HIP source translation. Key difference: warp size is 32 on
NVIDIA; wavefront size is 64 on AMD GCN/RDNA.

**Q: What is ROCm?**

A: ROCm (Radeon Open Compute) is AMD's open-source GPU compute platform: drivers, runtime,
math libraries (rocBLAS, rocFFT, MIOpen), profiling tools (ROCProfiler, Omniperf), and the
HIP programming layer. ROCm is AMD's answer to CUDA — it enables high-performance GPU
computing on AMD Instinct / Radeon Pro hardware.

**Q: How would you port CUDA code to AMD HIP?**

A: Step 1: run `hipify-perl` or `hipify-clang` on the source to auto-translate CUDA APIs
to HIP equivalents. Step 2: manually review warp-level operations — `__shfl_down_sync`
(CUDA) → `__shfl_down` (HIP), noting wavefront-64 semantics. Step 3: update CMakeLists.txt
to use `hipcc`. Step 4: validate correctness by comparing output to CPU reference.
Step 5: profile with ROCProfiler and tune for AMD's GCN/RDNA memory model.

**Q: How would you design cross-vendor GPU code?**

A: Define a thin abstraction layer that maps vendor-specific calls to a common interface:
`gpu_malloc()`, `gpu_memcpy()`, `gpu_sync()`. Use conditional compilation (`#ifdef __HIP_PLATFORM_AMD__`)
for vendor-specific paths. Keep kernels portable by avoiding NVIDIA-only extensions.
Use standard warp size (32) as a safe default with runtime detection for wavefront-64.
Test on both backends with the same correctness harness.

**Q: What should be abstracted to support CUDA and HIP backends?**

A: Abstractions worth maintaining: memory management (`gpu_malloc`/`gpu_free`), host-device
transfers (`gpu_memcpy_h2d`/`d2h`), synchronization (`gpu_sync`), error handling macro,
and kernel launch syntax. Keep math library calls (cuBLAS → rocBLAS) behind a thin BLAS
wrapper. Avoid abstracting kernel internals — write separate CUDA and HIP kernel files when
warp/wavefront differences require different logic.

---

## Portfolio Presentation Order (Interview)

1. **GPU Insight Lab** — shows full engineering workflow (collection → analysis → storage → reporting)
2. **CUDA Performance Lab** — shows kernel-level depth and profiling methodology
3. **GPU PCIe Bandwidth Benchmark** — shows hardware/system-level awareness
4. **CUDA → HIP Portability Demo** — shows cross-vendor adaptability
5. **See also**: `docs/12_WEEK_CUDA_JOB_ROADMAP.md` for the full learning path behind this portfolio
