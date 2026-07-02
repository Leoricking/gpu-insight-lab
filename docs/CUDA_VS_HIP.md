# GPU Insight Lab — CUDA vs. HIP Portability Guide

## Overview

GPU Insight Lab is designed primarily for NVIDIA CUDA but includes an AMD HIP compatibility
layer (stub) in `collectors/amd_collector.py` and `profilers/rocm_stub.py`. This document
covers the API mapping between CUDA and HIP, the critical architectural differences
(warp vs. wavefront), and the portability approach used in the codebase.

---

## Platform Status in v0.1.0

| Platform | Collectors | Benchmarks | Diagnosis | Profiling |
|----------|-----------|------------|-----------|-----------|
| NVIDIA CUDA | Full | Full | Full | Nsight Systems + Nsight Compute |
| AMD ROCm/HIP | Stub | Not validated | AMD_NOT_VALIDATED rule | rocm_stub.py (read-only) |
| Intel OneAPI | Not implemented | Not implemented | N/A | N/A |

AMD support is marked `NOT_VALIDATED` throughout. Every AMDGPUInfo result carries
`validation_status = "NOT_VALIDATED"` regardless of whether data collection succeeded.
This is a deliberate policy: GPU Insight Lab's correctness thresholds and performance
baselines were derived from NVIDIA hardware and cannot be applied to AMD without
validated AMD reference implementations.

---

## CUDA to HIP API Mapping

The table below covers APIs used in GPU Insight Lab's native CUDA kernels. HIP provides
header-level compatibility for most of these via `hip/hip_runtime.h`.

### Memory Management

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaMalloc` | `hipMalloc` | Identical semantics |
| `cudaFree` | `hipFree` | Identical semantics |
| `cudaMallocHost` | `hipHostMalloc` | Pinned host memory |
| `cudaFreeHost` | `hipHostFree` | |
| `cudaMemcpy` | `hipMemcpy` | |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | |
| `cudaMemset` | `hipMemset` | |
| `cudaMemGetInfo` | `hipMemGetInfo` | Returns free/total device memory |

### Execution Control

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | |
| `cudaStreamCreate` | `hipStreamCreate` | |
| `cudaStreamDestroy` | `hipStreamDestroy` | |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | |
| `cudaLaunchKernel` | `hipLaunchKernel` | Usually via `<<<>>>` syntax |
| `cudaFuncSetCacheConfig` | `hipFuncSetCacheConfig` | L1/shared split — AMD has unified cache |

### Events and Timing

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaEventCreate` | `hipEventCreate` | |
| `cudaEventRecord` | `hipEventRecord` | |
| `cudaEventSynchronize` | `hipEventSynchronize` | |
| `cudaEventElapsedTime` | `hipEventElapsedTime` | Returns float milliseconds |
| `cudaEventDestroy` | `hipEventDestroy` | |

### Device Queries

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaGetDeviceCount` | `hipGetDeviceCount` | |
| `cudaSetDevice` | `hipSetDevice` | |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | See struct differences below |
| `cudaDriverGetVersion` | `hipDriverGetVersion` | |
| `cudaRuntimeGetVersion` | `hipRuntimeGetVersion` | |

### Error Handling

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaError_t` | `hipError_t` | |
| `cudaSuccess` | `hipSuccess` | |
| `cudaGetLastError` | `hipGetLastError` | |
| `cudaGetErrorString` | `hipGetErrorString` | |

### Device Properties Structure Differences

`cudaDeviceProp` vs. `hipDeviceProp_t` are largely compatible but have some differences:

| Field | CUDA | HIP | Notes |
|-------|------|-----|-------|
| `warpSize` | 32 (all NVIDIA) | 64 (GCN/RDNA), 32 (RDNA3+) | **CRITICAL difference** |
| `maxThreadsPerBlock` | up to 1024 | up to 1024 | Same |
| `sharedMemPerBlock` | varies | varies | |
| `regsPerBlock` | varies | varies | |
| `multiProcessorCount` | SM count | CU count | Different compute units |
| `l2CacheSize` | varies | varies | |
| `totalGlobalMem` | bytes | bytes | Same |
| `gcnArchName` | N/A | present on HIP | AMD-specific |
| `gcnArch` | N/A | present on HIP | AMD GCN version |

---

## Warp vs. Wavefront

This is the most important architectural difference for kernel portability.

### NVIDIA: Warp = 32 Threads

On all NVIDIA GPUs (Kepler through Hopper), the warp size is **32 threads**.

- Threads in a warp execute in lockstep (SIMT)
- Divergent branches within a warp cause predication (serialization)
- Shared memory bank width (4 bytes) × 32 banks = 128-byte cache line
- Warp shuffle intrinsics: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`

### AMD: Wavefront = 64 Threads (historically) or 32 (RDNA3+)

On AMD GCN and RDNA1/RDNA2 GPUs, the wavefront size is **64 threads**.

- A kernel launched with block size 256 will have 256/32 = 8 warps on NVIDIA
- The same kernel on AMD GCN will have 256/64 = 4 wavefronts
- This halves the number of independent wavefronts available for latency hiding

On RDNA3 (RX 7000 series), AMD introduced optional 32-thread wave32 mode:
- `HIP_WAVE32_MODE` environment variable
- Kernel-level via `__attribute__((amdgpu_waves_per_eu(...)))`

### Portability Implications

#### Register Pressure

A kernel tuned for NVIDIA's 32-thread warp with 255 registers per thread will have
32 × 255 = 8,160 registers per warp. On AMD with 64-thread wavefronts, the same
per-thread register count doubles the register file pressure per wavefront, potentially
reducing occupancy further.

#### Warp Shuffle → Wavefront Shuffle

CUDA warp shuffle:
```cuda
float val = __shfl_down_sync(0xFFFFFFFF, src, delta, 32);
```

HIP wavefront shuffle (64-thread):
```hip
float val = __shfl_down(src, delta, 64);
// Note: mask parameter differs; HIP uses implicit full-wavefront mask
```

For wave32 mode on RDNA3:
```hip
float val = __shfl_down(src, delta, 32);
```

#### Shared Memory Reduction Example

A standard CUDA reduction written for warp=32:
```cuda
// Works on NVIDIA
if (tid < 32) {
    warpReduce(sdata, tid);  // unrolls for 32-thread warp
}
```

On AMD GCN (warp=64), the unrolled warp reduction must cover 64 threads:
```hip
// For AMD GCN wavefront=64
if (tid < 64) {
    wavefrontReduce(sdata, tid);
}
```

GPU Insight Lab's reduction kernel uses `warpSize` dynamically rather than hardcoding 32,
which is the correct approach for portability — but the HIP execution path is not
currently validated.

---

## Preprocessor Portability Pattern

The standard approach for CUDA/HIP dual compilation uses `hipify` or a compatibility header:

```cpp
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #define CUDA_CHECK(call) HIP_CHECK(call)
    // ... redefine other macros
#else
    #include <cuda_runtime.h>
    // CUDA-native path
#endif
```

GPU Insight Lab's native binary (`native/cuda/`) is CUDA-only. The portability shim is not
implemented in v0.1.0. `hipify-perl` (AMD's automatic translation tool) can produce a
HIP-compatible version of the kernels but correctness validation on AMD hardware has not
been performed.

---

## ROCm Toolchain vs. CUDA Toolchain

| Concept | NVIDIA CUDA | AMD ROCm |
|---------|-------------|----------|
| Compiler | `nvcc` | `hipcc` |
| Runtime library | `libcuda.so` / `cudart.dll` | `libamdhip64.so` / `amdhip64.dll` |
| Profiler (trace) | Nsight Systems (`nsys`) | ROCm Tracer (`rocprof --sys-trace`) |
| Profiler (kernel) | Nsight Compute (`ncu`) | ROCProfiler v2 (`rocprof`) |
| GPU query tool | `nvidia-smi` | `rocm-smi` |
| Architecture info | `nvidia-smi --query-gpu` | `rocminfo` |
| Compute capability | `sm_XY` (e.g., sm_86) | `gfxXXX` (e.g., gfx1100) |
| Profiling API | CUPTI | ROCTr / ATT |
| Streams | `cudaStream_t` | `hipStream_t` |
| Events | `cudaEvent_t` | `hipEvent_t` |
| Memory model | Unified Virtual Addressing (UVA) | Similar, platform-dependent |

---

## Performance Characteristics Comparison

The following differences affect how performance should be interpreted across platforms.
They are documented here for reference; GPU Insight Lab does not attempt cross-platform
performance normalization in v0.1.0.

| Characteristic | NVIDIA Ampere (A100) | AMD CDNA2 (MI250X) |
|----------------|---------------------|---------------------|
| Peak FP32 | 312 TFLOP/s (sparse) | 383 TFLOP/s |
| Peak FP16 (Tensor) | 624 TFLOP/s | 383 TFLOP/s (FP16 Matrix) |
| Memory bandwidth | 2,000 GB/s (HBM2e) | 3,276 GB/s (HBM2e) |
| Warp/wavefront size | 32 | 64 |
| Shared memory per CU/SM | 164 KB | 64 KB |
| L2 cache | 40 MB | 8 MB |
| NVLink vs Infinity Fabric | NVLink 3.0 | Infinity Fabric |

These figures are for illustration. Always refer to official datasheets for current
product specifications.

---

## Roadmap for AMD Support

The following work is planned for v0.2.0 to move AMD status from `NOT_VALIDATED` to
`VALIDATED`:

1. Port CUDA kernels to HIP using `hipify-perl` automatic translation
2. Validate correctness of all 7 kernel types on AMD RDNA3 and CDNA2 hardware
3. Update expected performance tables with AMD baseline values
4. Add wavefront-size-aware code paths (wave32 vs. wave64)
5. Validate `rocm_stub.py` profiling data collection with a real ROCm installation
6. Add AMD-specific diagnosis rules (wavefront occupancy, Infinity Fabric bandwidth)

Until this work is complete, AMD GPU users should treat all GPU Insight Lab performance
findings as `NOT_VALIDATED` and use AMD's native tools (`rocprof`, `rocm-bandwidth-test`)
for authoritative measurements.
