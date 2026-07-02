# GPU Insight Lab — Benchmark Methodology

## Overview

GPU Insight Lab benchmarks are designed for reproducibility, correctness verification, and
actionable interpretation. This document describes the statistical approach, warmup strategy,
correctness checks, and the specific limitations users must understand before drawing conclusions.

---

## Warmup Strategy

### Why Warmup Runs Are Required

GPU execution is subject to several transient effects that distort the first few measurements:

| Effect | Cause | Warmup Resolves? |
|--------|-------|-----------------|
| JIT compilation (CUDA) | First kernel launch triggers PTX → SASS JIT | Yes |
| Clock ramp-up | GPU boosts from idle frequency to boost frequency | Yes |
| TLB cold misses | Page table not yet populated | Yes |
| Memory prefetch | L2/L3 cache lines not yet populated | Partially |
| Driver overhead | First API call initializes context | Yes |

### Default Warmup Configuration

- **Native benchmarks**: 3 warmup iterations before any timing begins
- **CPU baselines**: 3 warmup iterations
- **Configurable via**: `--warmup N` (CLI / native binary), `AppConfig.warmup_runs`

Warmup results are discarded and never included in statistics.

---

## Measurement Repetitions

### Default Repeat Count

- **Default**: 10 measured iterations per benchmark
- **Rationale**: 10 iterations provide reasonable variance estimation without excessive runtime
- **Configurable via**: `--repeat N` (CLI), `AppConfig.repeat_count`

### Minimum Recommended Repeats

| Benchmark Type | Minimum | Recommended | Notes |
|----------------|---------|-------------|-------|
| Memory bandwidth | 5 | 10 | High variance from DRAM refresh |
| Compute kernels | 5 | 10 | Relatively stable |
| Host-device transfers | 10 | 20 | PCIe utilization varies with OS scheduler |
| CPU baselines | 10 | 20 | Subject to OS scheduling jitter |

---

## Statistics Computed

For every benchmark, the following statistics are computed over the `N` measured iterations:

| Statistic | Symbol | Computation |
|-----------|--------|-------------|
| Mean | μ | Σ(xᵢ) / N |
| Median | M | Middle value after sort |
| Minimum | min | min(xᵢ) |
| Maximum | max | max(xᵢ) |
| Standard deviation | σ | sqrt(Σ(xᵢ - μ)² / N) — **population** std dev |
| Coefficient of variation | CV | σ / μ × 100% |

### Why Population (Not Sample) Standard Deviation

The N measured runs are treated as the complete population of observations for that session.
The goal is to characterize observed variance, not estimate population variance from a sample.
Using N (not N-1) in the denominator is therefore correct for this use case.

### Interpretation of CV

| CV Range | Interpretation |
|----------|----------------|
| < 2% | Highly stable — results are reliable |
| 2–5% | Acceptable variance — typical for GPU benchmarks |
| 5–10% | Elevated variance — thermal throttling or power limits may be active |
| > 10% | High variance — results should be treated with caution; investigate cause |

---

## Timing Methodology

### Native CUDA Benchmarks (C++/CUDA)

Timing uses CUDA Events for kernel-only timing:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// kernel launch
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

**Measured**: Kernel execution time only — does NOT include:
- Host-to-device memory transfer
- Device-to-host memory transfer
- Memory allocation / deallocation

For benchmarks where transfer time is relevant (PCIe bandwidth), transfer time is measured
separately and reported as a distinct metric.

### CPU Python Baselines

Timing uses `time.perf_counter()`, which provides the highest-resolution monotonic clock
available on the platform:

```python
t0 = time.perf_counter()
# operation
elapsed = time.perf_counter() - t0
```

---

## Benchmark Catalogue

### Native CUDA Kernels

#### vector_add
- **What it measures**: Baseline memory-bandwidth-bound throughput; element-wise addition of two float arrays
- **Correctness check**: CPU reference computation; max absolute error must be < 1e-5
- **Performance metric**: GB/s (effective memory bandwidth)
- **Expected range**: 200–900 GB/s on modern NVIDIA GPUs

#### reduction
- **What it measures**: Parallel reduction; tests shared memory and warp-shuffle efficiency
- **Correctness check**: CPU sum reference; relative error < 0.01%
- **Performance metric**: GB/s input bandwidth
- **Notes**: Reduction efficiency reveals shared memory bank conflicts

#### transpose
- **What it measures**: Memory access pattern efficiency; coalesced vs. non-coalesced access
- **Correctness check**: CPU reference transpose; element-wise equality
- **Performance metric**: GB/s effective bandwidth
- **Notes**: Tiled transpose should approach peak bandwidth; naive transpose shows coalescing cost

#### gemm_naive
- **What it measures**: Baseline matrix multiply without optimization
- **Correctness check**: Compared against gemm_tiled; relative error < 0.01%
- **Performance metric**: GFLOP/s (2 × M × N × K floating point ops)

#### gemm_tiled
- **What it measures**: Tiled matrix multiply using shared memory tiling
- **Correctness check**: Compared against CPU numpy reference
- **Performance metric**: GFLOP/s
- **Notes**: Should be 4–10× faster than gemm_naive on the same hardware

#### memory_bandwidth
- **What it measures**: Raw device memory bandwidth (streaming read/write)
- **Correctness check**: None (pure bandwidth test; no output verification)
- **Performance metric**: GB/s
- **Expected range**: Should approach published peak bandwidth (e.g., 936 GB/s on A100)

#### stream_pipeline
- **What it measures**: Concurrent kernel execution and async H2D/D2H overlap via CUDA streams
- **Correctness check**: Final values validated against serial reference
- **Performance metric**: Throughput (GB/s) and pipeline latency (ms)
- **Notes**: Low overlap efficiency indicates driver or OS scheduling interference

#### image_grayscale
- **What it measures**: 2D texture/pixel processing; real-world image workload proxy
- **Correctness check**: Pixel values compared to CPU reference; max error < 2 (uint8 range)
- **Performance metric**: Megapixels/second

### CPU Python Baselines

CPU baselines provide a reference point for GPU speedup ratios and allow diagnosis even
when no GPU is present.

#### cpu_vector_add
- **Implementation**: NumPy element-wise addition
- **Purpose**: Reference for GPU vector_add speedup ratio

#### cpu_matrix_multiply
- **Implementation**: NumPy `@` operator (calls BLAS/LAPACK)
- **Purpose**: Reference for GPU GEMM speedup ratio

#### cpu_image_grayscale
- **Implementation**: Pillow or NumPy weighted sum
- **Purpose**: Reference for GPU image processing speedup

---

## Correctness Verification

Every benchmark with a deterministic expected output performs a correctness check before
reporting performance. If correctness fails:

- `BenchmarkResult.correctness_verified = False`
- `BenchmarkResult.correctness_error` is populated with the error magnitude
- Performance numbers are still reported but flagged with a warning
- The Diagnosis Engine will flag this as a CRITICAL finding

### Tolerance Values

| Data Type | Absolute Tolerance | Relative Tolerance |
|-----------|--------------------|-------------------|
| float32 | 1e-5 | 0.01% |
| float64 | 1e-9 | 0.001% |
| int/uint8 | 2 (absolute) | N/A |

Tolerances account for floating-point non-associativity in parallel reductions.

---

## Reproducibility

### Factors That Reduce Reproducibility

1. **Thermal throttling**: GPU may downclock mid-session if thermal headroom is exhausted
2. **Power limits**: Board partners set varying power limits; same GPU model may behave differently
3. **Driver version**: CUDA driver and toolkit versions can change kernel compilation
4. **System load**: Background processes compete for PCIe bandwidth and DRAM
5. **Memory state**: GDDR6X scrambling patterns vary with temperature

### Improving Reproducibility

- Run benchmarks on an idle system (no other GPU workloads)
- Allow GPU to reach thermal steady state before measuring (run warmup with high repeat count)
- Lock GPU clocks with `nvidia-smi -lgc <clock_mhz>` (requires sufficient privileges)
- Use `--repeat 20` or higher for publication-quality results
- Compare sessions only when system environment (driver, clock policy) is identical

### Session Metadata

Every `BenchmarkSession` records:
- Timestamp (UTC ISO 8601)
- Driver version
- CUDA toolkit version
- GPU model and UUID
- System memory and CPU model
- GPU clock state at time of benchmarking

This metadata makes session-to-session comparison traceable.

---

## Performance Metrics Reference

| Metric | Unit | What It Measures |
|--------|------|-----------------|
| Memory bandwidth | GB/s | Device memory read/write throughput |
| Compute throughput | GFLOP/s | Floating-point operations per second |
| PCIe throughput | GB/s | Host-device transfer bandwidth |
| Kernel latency | ms | End-to-end kernel execution time |
| Image throughput | Mpix/s | Processed pixels per second |
| Speedup ratio | × | GPU time / CPU baseline time |

---

## Limitations

- **Single-GPU only**: GPU Insight Lab tests one GPU at a time. Multi-GPU NVLink or SLI
  configurations are not covered.
- **No FP16/BF16/INT8**: All benchmarks use FP32. Tensor Core performance (FP16/BF16) is
  not tested in v0.1.0.
- **No cuBLAS/cuDNN**: GEMM benchmarks use hand-written CUDA kernels, not cuBLAS. cuBLAS
  would be significantly faster on Tensor Core hardware.
- **Microbenchmarks ≠ application performance**: Results describe hardware capability under
  synthetic load. Real workloads (training, inference, rendering) involve many additional
  bottlenecks.
- **Linux vs. Windows timing**: Windows GPU scheduling granularity may differ from Linux.
  Kernel launch latency is typically higher on Windows.
- **ROCm/HIP**: AMD GPU results carry `NOT_VALIDATED` status. Kernel correctness cannot be
  guaranteed without validated reference implementations on HIP.
- **CPU baselines use NumPy BLAS**: CPU matrix multiply throughput depends on which BLAS
  library NumPy was linked against (OpenBLAS, MKL, etc.). This is not controlled.
