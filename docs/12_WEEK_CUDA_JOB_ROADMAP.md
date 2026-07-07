# 12-Week CUDA → NVIDIA / AMD Job Roadmap

**Target Roles**: CUDA Performance Engineer · GPU Software Engineer · HPC Software Engineer ·  
GPU Validation Engineer · CUDA Developer · AMD HIP / ROCm Engineer

**Profile**: Engineer with PCIe, HPC, driver, system validation, and system debugging background  
transitioning into CUDA / GPU Performance Engineering.

> This roadmap is not about becoming a "pure AI engineer." It is about positioning yourself  
> as an engineer who understands hardware, PCIe, system validation, and performance  
> diagnostics — and can now also write, profile, and optimize CUDA kernels.

---

## Overview

| Phase | Weeks | Focus |
|-------|-------|-------|
| Foundation | 1–3 | CUDA programming model, memory hierarchy, first kernels |
| Performance | 4–6 | Shared memory, occupancy, profiling with Nsight |
| Advanced Kernels | 7–9 | Reduction, prefix sum, matrix operations, AI inference kernels |
| Cross-Vendor | 10–11 | HIP / ROCm, portability, AMD architecture |
| Portfolio & Interview | 12 | End-to-end portfolio, demo polish, mock interviews |

---

## Week 1 — CUDA Programming Model Fundamentals

### Theme
Grid, block, thread hierarchy; CUDA memory model; first kernel launch.

### Deliverables
- `vector_add.cu`: host allocation, `cudaMalloc`, `cudaMemcpy`, kernel launch, `cudaFree`
- `device_query.cu`: enumerate device properties (SM count, warp size, clock, memory)
- Error checking macro `CUDA_CHECK(err)` used throughout

### GitHub Output
- Repo: `cuda-performance-lab/week01-fundamentals/`
- README with architecture diagram showing grid → block → thread mapping

### Interview Prep
- What is a grid? A block? A thread?
- How many threads per block is typical? Why is 256 common?
- What is the difference between `__global__`, `__device__`, and `__host__`?
- What happens if you launch more blocks than there are SMs?

### Role Alignment
- NVIDIA: demonstrates CUDA programming model knowledge
- GPU Validation: demonstrates ability to enumerate and record device properties

### GPU Insight Lab Connection
Compare your `vector_add` GB/s result against GPU Insight Lab's `vector_add` kernel output.
If they differ, use GPU Insight Lab's diagnosis engine to check for PCIe bottleneck or bandwidth headroom.

---

## Week 2 — Memory Hierarchy: Global, Shared, Registers

### Theme
GPU memory spaces, coalescing, shared memory tiling, bank conflicts.

### Deliverables
- `matrix_transpose_naive.cu`: non-coalesced baseline
- `matrix_transpose_tiled.cu`: 32×32 tile + shared memory, pad to avoid bank conflicts
- Measured GB/s comparison: naive vs. tiled
- CSV result file: `transpose_benchmark.csv`

### GitHub Output
- Repo: `cuda-performance-lab/week02-memory/`
- Roofline chart (matplotlib) showing naive vs. tiled vs. theoretical peak

### Interview Prep
- Why does non-coalesced access hurt performance?
- What is a shared memory bank conflict?
- How do you pad shared memory to avoid conflicts?
- What is the L2 cache hit rate and how do you check it in Nsight Compute?

### Role Alignment
- CUDA Performance: core memory optimization skill
- HPC Software: data layout optimization is foundational to HPC workloads

### GPU Insight Lab Connection
GPU Insight Lab's `transpose.cu` demonstrates the same pattern. Use it as reference implementation
and compare your measured GB/s to GPU Insight Lab's `LOW_MEMORY_BANDWIDTH` diagnosis threshold.

---

## Week 3 — PCIe Bandwidth & Host–Device Transfer

### Theme
`cudaMemcpy` bandwidth, pinned vs. pageable memory, PCIe transfer bottleneck evidence.

### Deliverables
- `pcie_bandwidth_benchmark.cu`:
  - H2D (host to device) bandwidth
  - D2H (device to host) bandwidth
  - D2D (device to device) bandwidth
  - Pinned vs. pageable memory comparison
  - Transfer sizes: 1 MB, 4 MB, 16 MB, 64 MB, 256 MB, 1 GB
- Output: CSV + JSON + HTML report
- Annotate: at what transfer size does H2D bandwidth plateau?

### GitHub Output
- Repo: `gpu-pcie-bandwidth-benchmark/`
- HTML report with bandwidth vs. transfer size chart
- README explaining PCIe Gen3 x16 theoretical peak (16 GB/s) vs. measured

### Interview Prep
- Why is pinned memory faster than pageable memory for H2D transfers?
- What is PCIe Gen3 x16 theoretical bidirectional bandwidth?
- How do you detect a PCIe bottleneck in a production system?
- What is the difference between H2D and D2D bandwidth on an A100?

### Role Alignment
- GPU Validation: PCIe bandwidth testing is a standard validation workload
- NVIDIA/AMD: PCIe health is a critical system-level metric
- HPC: PCIe bottleneck is a common root cause for slow MPI + GPU workflows

### GPU Insight Lab Connection
GPU Insight Lab's `PCIE_BOTTLENECK` diagnosis rule and PCIe collector align directly.
Map your measured H2D bandwidth against GPU Insight Lab's diagnosis evidence output.

---

## Week 4 — Reduction and Parallel Primitives

### Theme
Parallel reduction, warp shuffle instructions, multi-pass reduction.

### Deliverables
- `reduction_naive.cu`: per-block reduction with shared memory
- `reduction_warp_shuffle.cu`: `__shfl_down_sync` final warp reduction
- `reduction_cub.cu`: CUB `DeviceReduce::Sum` baseline
- Benchmarks: naive vs. warp shuffle vs. CUB — GB/s and ms

### GitHub Output
- Repo: `cuda-performance-lab/week04-reduction/`
- Performance comparison table in README
- Nsight Compute screenshot: achieved occupancy, memory throughput

### Interview Prep
- Why do you need two passes for a reduction (blocks then final)?
- What is `__shfl_down_sync` and why is it faster than shared memory for the final warp?
- What is warp divergence and where does it appear in a naive reduction?
- What does "achieved occupancy" mean in Nsight Compute?

### Role Alignment
- CUDA Performance: reduction is a foundational parallel primitive
- HPC Software: reduction is used in dot products, norms, collective ops

### GPU Insight Lab Connection
GPU Insight Lab's `reduction.cu` kernel can serve as your correctness reference.
Run your reduction and compare output to GPU Insight Lab's CPU baseline.

---

## Week 5 — GEMM: Naive → Tiled → cuBLAS

### Theme
Matrix multiply optimization: memory reuse, register blocking, cuBLAS as ceiling.

### Deliverables
- `gemm_naive.cu`: baseline O(N³) global memory access
- `gemm_tiled.cu`: shared memory tiles, 16×16 or 32×32
- `gemm_cublas.cu`: cuBLAS `cublasSgemm` as reference ceiling
- CPU baseline: NumPy `@` or BLAS `sgemm`
- Nsight Compute profile: memory throughput, pipe utilization for each variant

### GitHub Output
- Repo: `cuda-performance-lab/week05-gemm/`
- Nsight Compute `.ncu-rep` file checked in
- Table: naive / tiled / cuBLAS GFLOP/s

### Interview Prep
- What arithmetic intensity does a tiled GEMM achieve vs. naive?
- Why does the tiled kernel need `__syncthreads()` twice per tile iteration?
- What is the difference between TFLOP/s (theoretical) and GFLOP/s (measured)?
- How close to cuBLAS can a hand-written tiled kernel get on Ampere?

### Role Alignment
- CUDA Performance: GEMM is the core of neural network training
- NVIDIA: cuBLAS is the performance ceiling every GPU role must understand

### GPU Insight Lab Connection
GPU Insight Lab ships `gemm_naive` and `gemm_tiled` kernels. Use its `LOW_COMPUTE_THROUGHPUT`
rule to verify your kernel matches expected GFLOP/s for your GPU model.

---

## Week 6 — Profiling: Nsight Systems + Nsight Compute

### Theme
End-to-end profiling workflow: timeline (nsys) → kernel counters (ncu) → diagnosis.

### Deliverables
- Profile your Week 5 GEMM variants with Nsight Compute
- `nsight_report/`: save `.ncu-rep`, `.nsys-rep`, exported CSV
- Write a `PROFILING_ANALYSIS.md`:
  - For each kernel: achieved occupancy, memory throughput, SM utilization
  - Identify the bottleneck: memory-bound or compute-bound?
  - One optimization applied based on profiler guidance

### GitHub Output
- Repo: `cuda-performance-lab/week06-profiling/`
- `PROFILING_ANALYSIS.md` in repo

### Interview Prep
- What is the difference between Nsight Systems and Nsight Compute?
- What does SM utilization (nvidia-smi) measure vs. achieved occupancy (ncu)?
- What is the Roofline model? How do you read it in Nsight Compute?
- What counter tells you whether a kernel is memory-bound?

### Role Alignment
- CUDA Performance: profiling is the core engineering tool
- GPU Validation: Nsight is the standard diagnostic tool for NVIDIA GPUs
- NVIDIA DevTools: Nsight profiling is foundational to every performance role

### GPU Insight Lab Connection
GPU Insight Lab's `profilers/nsight_compute.py` and `profilers/nsight_systems.py` automate
this workflow. Demonstrate how you extended the lab to run your own kernels.

---

## Week 7 — Prefix Sum (Scan) and Advanced Primitives

### Theme
Inclusive/exclusive scan, Blelloch algorithm, CUB DeviceScan.

### Deliverables
- `scan_naive.cu`: sequential prefix sum as CPU reference
- `scan_blelloch.cu`: work-efficient parallel scan (up-sweep + down-sweep)
- `scan_cub.cu`: CUB `DeviceScan::ExclusiveSum`
- Correctness test: compare against CPU reference for 1M elements
- Benchmark: GB/s for each variant

### GitHub Output
- Repo: `cuda-performance-lab/week07-scan/`
- Correctness validation pass/fail badge in README

### Interview Prep
- What is the work complexity of a naive scan vs. Blelloch scan?
- What is the difference between inclusive and exclusive scan?
- Where does prefix sum appear in GPU computing? (histogram, compaction, sort)
- How does CUB's multi-block scan handle the cross-block dependency?

### Role Alignment
- CUDA Performance: scan is used in segmented operations, sparse kernels, stream compaction
- HPC Software: scan primitives appear in mesh operations, particle simulation

### GPU Insight Lab Connection
GPU Insight Lab's correctness validation pattern applies directly.
Use GPU Insight Lab's `CORRECTNESS_FAILURE` rule to auto-flag any scan result mismatches.

---

## Week 8 — 2D Convolution and AI Inference Kernels

### Theme
2D convolution, sliding window, im2col; AI inference kernel patterns.

### Deliverables
- `conv2d_naive.cu`: direct convolution (no optimization)
- `conv2d_shared.cu`: shared memory tile-based convolution
- `softmax_kernel.cu`: numerically stable softmax for inference
- `layer_norm.cu`: layer normalization kernel
- `gelu_kernel.cu`: GELU activation (exact and approximate)
- CPU reference for all: NumPy or SciPy

### GitHub Output
- Repo: `cuda-performance-lab/week08-ai-kernels/`
- Performance table: naive vs. optimized + cuDNN ceiling

### Interview Prep
- How do you handle the sliding window boundary in a 2D convolution kernel?
- Why is softmax numerically unstable in the naive form? How do you fix it?
- What is the difference between layer normalization and batch normalization at the kernel level?
- How does GELU differ from ReLU in terms of kernel implementation complexity?

### Role Alignment
- CUDA Performance: AI inference kernels are the growth area for GPU roles
- NVIDIA: cuDNN, TensorRT all rely on these primitives
- HPC Software: normalization and activation are universal in scientific ML

### GPU Insight Lab Connection
Add these kernels to GPU Insight Lab's future Kernel Lab roadmap.
Document them in GPU Insight Lab's `docs/COMMERCIAL_ROADMAP.md`.

---

## Week 9 — Streams, Overlap, and Concurrent Execution

### Theme
CUDA streams, async memcpy, kernel-transfer overlap, multi-stream concurrency.

### Deliverables
- `stream_overlap.cu`: overlap H2D transfer with kernel execution on separate streams
- `multi_stream.cu`: N independent tasks on N streams
- Nsight Systems timeline: visualize overlap
- Benchmark: serialized vs. pipelined throughput (GB/s + ms)

### GitHub Output
- Repo: `cuda-performance-lab/week09-streams/`
- Nsight Systems screenshot showing stream timeline overlap
- Measured speedup from pipelining

### Interview Prep
- How do you overlap data transfer and compute with CUDA streams?
- What is `cudaMemcpyAsync` and what does it require from host memory?
- What is a CUDA event and how do you use it to time across streams?
- What is the default stream and why does it serialize everything?

### Role Alignment
- CUDA Performance: overlapping compute and transfer is essential for production throughput
- HPC Software: pipeline overlap is fundamental to large-scale simulation
- NVIDIA: streaming patterns appear in TensorRT, NCCL, and inference backends

### GPU Insight Lab Connection
GPU Insight Lab's `stream_pipeline` kernel and `profilers/nsight_systems.py` directly cover
this topic. Use GPU Insight Lab to run your stream benchmark and compare to `stream_pipeline` results.

---

## Week 10 — HIP / ROCm and Cross-Vendor Portability

### Theme
CUDA → HIP porting, API mapping, build system differences, AMD architecture overview.

### Deliverables
- Port Week 4 reduction and Week 5 GEMM tiled from CUDA to HIP
- `hip_reduction.hip`: HIP version with `hipMalloc`, `hipMemcpy`, `__shfl_down`
- `hip_gemm_tiled.hip`: HIP tiled GEMM
- `PORTING_NOTES.md`:
  - API mapping table (CUDA → HIP)
  - Build differences (nvcc vs. hipcc)
  - Warp (32) vs. wavefront (64) differences
  - Performance considerations

### GitHub Output
- Repo: `cuda-to-hip-portability-demo/`
- Side-by-side diff: CUDA vs. HIP version
- `PORTING_NOTES.md` checked in

### Interview Prep
- What is ROCm? What is HIP?
- What is the difference between warp size (CUDA) and wavefront size (HIP/ROCm)?
- How does `hipify-perl` automate CUDA → HIP conversion?
- What CUDA APIs have no direct HIP equivalent?
- What should be abstracted to support both CUDA and HIP backends?

### Role Alignment
- AMD: directly demonstrates HIP/ROCm porting capability
- GPU Validation: cross-vendor testing is common in validation roles
- Platform SW: cross-vendor GPU support is a growing requirement

### GPU Insight Lab Connection
GPU Insight Lab's `docs/CUDA_VS_HIP.md`, `native/hip/README.md`, and AMD collector
are your reference. Demonstrate how GPU Insight Lab handles cross-vendor detection.

---

## Week 11 — System Validation: Stress Tests, Correctness, Regression

### Theme
GPU health validation workflow: stress tests, correctness checks, regression comparison.

### Deliverables
- `gpu_stress_test.cu`: sustained compute + memory stress with temperature monitoring
- Correctness validation harness: compare GPU outputs to CPU reference for 5 kernels
- Regression comparison: run same benchmark on two driver versions, diff results
- Use GPU Insight Lab to save both sessions and run `compare_sessions()`

### GitHub Output
- `cuda-performance-lab/week11-validation/`
- Regression comparison report (JSON + HTML)
- Evidence that GPU Insight Lab's `CORRECTNESS_FAILURE` rule fires on a deliberately broken kernel

### Interview Prep
- How do you validate that a GPU kernel is producing correct results?
- What is regression testing for GPU benchmarks?
- How do you compare benchmark results across driver versions?
- What is ECC and how does it affect memory bandwidth and correctness?

### Role Alignment
- GPU Validation: this is the core of validation engineering
- NVIDIA/AMD: regression tracking is mandatory for driver releases
- HPC: correctness validation is critical for scientific computing

### GPU Insight Lab Connection
This is where GPU Insight Lab shines as a portfolio piece.
GPU Insight Lab's SQLite session history, `compare_sessions()`, and diagnosis engine are exactly
the tools a GPU validation engineer would build. Demonstrate the full workflow live.

---

## Week 12 — Portfolio Polish, Demo Prep, Interview Simulation

### Theme
End-to-end portfolio assembly, interview demo script rehearsal, README and resume finalization.

### Deliverables
- All repos polished: README, LICENSE, CI badge (GitHub Actions), Nsight screenshots
- GPU Insight Lab: run full benchmark, generate HTML + Excel report, show session comparison
- Record a 3-minute screen demo of:
  1. `gpu-insight full-test` running
  2. HTML report opened in browser
  3. Session comparison showing delta %
  4. Nsight Compute screenshot from your GEMM kernel
- Updated resume with keywords (see below)
- 3 mock technical phone screens with a peer

### GitHub Output
- Pin the following 3 repos on your GitHub profile:
  1. `cuda-performance-lab` (Weeks 1–9)
  2. `gpu-pcie-bandwidth-benchmark` (Week 3)
  3. `cuda-to-hip-portability-demo` (Week 10)
- GPU Insight Lab as primary portfolio anchor

### Interview Prep
- Full mock interview: 45 minutes, CUDA fundamentals + profiling + system design
- Prepare "tell me about a project" narrative (use the demo script below)
- Prepare answers for all question categories in `docs/INTERVIEW_GUIDE.md`

---

## Portfolio: 3 Job-Aligned Projects

### 1. CUDA Performance Lab

**Repo**: `cuda-performance-lab`

| Benchmark | CPU Baseline | Naive CUDA | Optimized CUDA | Profiling |
|-----------|-------------|------------|----------------|-----------|
| vector_add | NumPy | ✓ | N/A (bandwidth-bound) | Nsight Compute |
| matrix_transpose | NumPy | non-coalesced | tiled + shared | Nsight Compute |
| reduction | NumPy sum | shared memory | warp shuffle | Nsight Compute |
| prefix_sum | NumPy cumsum | naive scan | Blelloch scan | Nsight Compute |
| matmul (GEMM) | BLAS | naive | tiled + register | Nsight Compute |
| conv2d | SciPy | direct | shared tiled | Nsight Compute |
| softmax | NumPy | naive | numerically stable | Nsight Compute |
| layer_norm | NumPy | naive | optimized | Nsight Compute |
| gelu | NumPy | exact | approximate | Nsight Compute |

Each benchmark includes: CPU baseline → naive CUDA → optimized CUDA → Nsight Compute profiling
report → bottleneck analysis.

**Interview talking point**:
> "This lab shows I can implement, benchmark, and optimize CUDA kernels from scratch — and that
> I use the profiler to drive optimization decisions rather than guessing."

---

### 2. GPU PCIe Bandwidth Benchmark Tool

**Repo**: `gpu-pcie-bandwidth-benchmark`

| Measurement | Transfer Sizes | Output |
|-------------|---------------|--------|
| H2D bandwidth | 1 MB – 1 GB | GB/s vs. size chart |
| D2H bandwidth | 1 MB – 1 GB | GB/s vs. size chart |
| D2D bandwidth | 1 MB – 1 GB | GB/s vs. size chart |
| Pinned vs. pageable | 64 MB | speedup ratio |
| PCIe bottleneck evidence | all sizes | diagnosis finding |

Output formats: CSV / JSON / HTML / Excel report.

**Interview talking point**:
> "PCIe bandwidth is the first thing I check when a GPU workflow underperforms. This tool
> gives me evidence — not just a gut feeling — about whether the bottleneck is the interconnect,
> the memory subsystem, or the kernel itself."

---

### 3. CUDA → HIP Portability Demo

**Repo**: `cuda-to-hip-portability-demo`

| Item | CUDA | HIP |
|------|------|-----|
| Memory allocation | `cudaMalloc` | `hipMalloc` |
| Kernel launch | `<<<grid, block>>>` | `hipLaunchKernelGGL` |
| Warp shuffle | `__shfl_down_sync` | `__shfl_down` |
| Warp size | 32 | 64 (GCN/RDNA) |
| Build | `nvcc` | `hipcc` |
| Profiler | Nsight Compute | ROCProfiler |

Includes: API mapping table, porting notes, build differences, performance considerations,
CUDA vs. HIP difference analysis.

**Interview talking point**:
> "Porting this code taught me the concrete differences between CUDA and HIP — not just the
> API surface, but the architectural implications of wavefront-64 vs. warp-32."

---

## AI Inference Kernel Roadmap

Add to `cuda-performance-lab` after Week 12 (future work):

| Kernel | Priority | Notes |
|--------|----------|-------|
| softmax | Done (Week 8) | numerically stable |
| layer normalization | Done (Week 8) | fused mean + variance |
| GELU | Done (Week 8) | exact + approximate |
| 2D convolution | Done (Week 8) | shared memory tiled |
| prefix sum / scan | Done (Week 7) | Blelloch algorithm |
| Flash Attention (simplified) | Future | multi-head attention kernel |
| INT8 quantization kernel | Future | inference optimization |
| PyTorch custom extension | Future | `torch.utils.cpp_extension` |
| TensorRT plugin | Future | inference backend integration |

---

## Final GitHub Portfolio Recommendations

### Profile Setup
1. Pin 4 repos: `gpu-insight-lab`, `cuda-performance-lab`, `gpu-pcie-bandwidth-benchmark`, `cuda-to-hip-portability-demo`
2. Profile bio: "GPU Software Engineer | CUDA · HIP · PCIe · System Validation · Performance Diagnostics"
3. Each repo must have: README with performance numbers, Nsight screenshots, license

### Repo Quality Checklist
- [ ] README with hardware requirements and build instructions
- [ ] Performance numbers table in README
- [ ] At least one Nsight Compute or Nsight Systems screenshot
- [ ] CMakeLists.txt with correct architecture targets (sm_75–sm_90)
- [ ] CUDA_CHECK error macro used consistently
- [ ] CPU reference baseline for correctness validation

### Presentation Order in Interview
1. Start with GPU Insight Lab — shows full engineering workflow
2. Pivot to CUDA Performance Lab — shows kernel-level depth
3. Reference PCIe Bandwidth Tool — shows hardware/system awareness
4. Mention HIP portability — shows cross-vendor adaptability

---

## Resume Keywords

### CUDA / GPU Programming
`CUDA` `CUDA C++` `CUDA kernels` `shared memory` `warp shuffle` `memory coalescing`
`occupancy optimization` `kernel fusion` `CUDA streams` `asynchronous execution`
`CUDA Events` `cudaMemcpyAsync` `pinned memory` `CUB` `Thrust`

### Profiling & Diagnostics
`Nsight Systems` `Nsight Compute` `nvidia-smi` `CUPTI` `Roofline model`
`memory throughput` `SM utilization` `achieved occupancy` `kernel profiling`
`performance regression` `bottleneck analysis`

### Hardware & System
`PCIe bandwidth` `H2D/D2H transfer` `PCIe bottleneck` `GPU memory hierarchy`
`HBM2` `GDDR6X` `thermal throttling` `ECC memory` `GPU validation`
`PCIe Gen4` `NVLink`

### Cross-Vendor / HPC
`HIP` `ROCm` `hipcc` `CUDA to HIP portability` `wavefront` `warp`
`cross-vendor GPU` `MPI+CUDA` `HPC workloads` `GPU cluster`

### Software Engineering
`CMake` `C++17` `Python` `pytest` `SQLite` `JSON` `subprocess` `pynvml`
`evidence-based diagnosis` `reproducible benchmarks` `session comparison`

---

## Interview Demo Script (3 Minutes)

Use this when an interviewer says: **"Walk me through one of your projects."**

---

**[30 seconds — Hook]**

> "I built GPU Insight Lab — a GPU performance diagnostics tool that I use as my portfolio
> anchor for CUDA and GPU validation engineering roles. It's not a CUDA sample. It's a
> complete engineering workflow: collect telemetry, run kernels, diagnose findings, store
> sessions, generate reports."

**[60 seconds — Technical Depth]**

> "The tool has a CUDA C++ binary with 7 benchmark kernels — vector add, reduction,
> matrix transpose, naive and tiled GEMM, memory bandwidth, and stream pipeline.
> Each kernel produces GB/s or GFLOP/s results. The Python layer orchestrates the binary
> via subprocess with a JSON stdout protocol, runs an evidence-based diagnosis engine
> with 9 rules, and saves sessions to SQLite for regression comparison.
>
> I can pull up two sessions from before and after a driver update and show you the
> delta percent. The diagnosis engine flags if bandwidth dropped, if PCIe is running
> below maximum link width, or if a kernel correctness check failed."

**[45 seconds — Hardware Awareness]**

> "What sets this apart from 'I took the CUDA samples tutorial' is the system-level
> thinking. I built a PCIe bandwidth benchmark that measures H2D, D2H, and D2D at
> multiple transfer sizes — because PCIe is often the first bottleneck and you need
> evidence, not intuition.
>
> I also ported two kernels from CUDA to HIP — reduction and tiled GEMM — which forced
> me to confront the warp-32 vs. wavefront-64 difference concretely, not just as a
> trivia fact."

**[45 seconds — Close]**

> "My background is in PCIe validation and system debugging, so I think about GPU
> performance the way a systems engineer does — what is the bottleneck, where is the
> evidence, can I reproduce it. GPU Insight Lab is how I translated that background
> into GPU software engineering.
>
> I'm happy to share the repo link, show you the HTML report it generates, or walk
> through any specific kernel or profiling result."

---

## Final Checklist (12-Week Acceptance)

| Item | Status |
|------|--------|
| 12-week CUDA job roadmap added to docs | ✓ |
| CUDA Performance Lab (9 kernels, CPU baseline, Nsight profiling) | Roadmap defined |
| GPU PCIe Bandwidth Benchmark Tool (H2D/D2H/D2D, CSV/JSON/HTML/Excel) | Roadmap defined |
| CUDA → HIP Portability Demo (API mapping, porting notes, warp/wavefront) | Roadmap defined |
| AI inference kernel roadmap (softmax, layer_norm, GELU, conv2d, scan) | Roadmap defined |
| Interview question bank added to INTERVIEW_GUIDE.md | ✓ |
| NVIDIA / AMD / GPU Validation / Performance Engineering alignment | ✓ |
| Resume keywords | ✓ |
| Interview demo script | ✓ |
| GitHub portfolio recommendations | ✓ |
