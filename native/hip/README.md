# HIP/ROCm Support — GPU Insight Lab

**Status: NOT_VALIDATED** — No AMD GPU was available during development of v0.1.0.
The HIP files in this directory are portability references, not validated benchmarks.

---

## NOT_VALIDATED Policy

All HIP/ROCm code in this directory carries status `NOT_VALIDATED`.
This means:
- The code compiles with `hipcc` (verified structurally).
- It has NOT been run on actual AMD GPU hardware.
- Performance numbers (if any) are NOT validated against AMD reference implementations.
- GPU Insight Lab's diagnosis thresholds and scoring were derived from NVIDIA hardware
  and do NOT apply to AMD GPUs without separate AMD-specific validation.

When an AMD GPU / ROCm environment is unavailable, all HIP benchmarks return:
```json
{"status": "NOT_VALIDATED", "policy": "NOT_VALIDATED when AMD GPU / ROCm unavailable"}
```

---

## Files

| File | Description | Status |
|------|-------------|--------|
| `vector_add_hip.cpp` | HIP vector add portability demo | NOT_VALIDATED |
| `reduction_hip.cpp` | HIP parallel reduction (Blelloch-style) | NOT_VALIDATED |
| `gemm_naive_hip.cpp` | HIP naive GEMM (C = A * B) | NOT_VALIDATED |

---

## Build Instructions

Requires ROCm 5.x or later: https://rocmdocs.amd.com

```bash
# Vector add
hipcc vector_add_hip.cpp -o vector_add_hip -D__HIP__
./vector_add_hip

# Reduction
hipcc reduction_hip.cpp -o reduction_hip -D__HIP__
./reduction_hip

# Naive GEMM
hipcc gemm_naive_hip.cpp -o gemm_naive_hip -D__HIP__
./gemm_naive_hip
```

Without `-D__HIP__`, each file compiles a stub that prints the NOT_VALIDATED JSON.

---

## CUDA to HIP API Mapping Table

The following table covers APIs used in GPU Insight Lab's kernels.

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

### Events and Timing

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaEvent_t` | `hipEvent_t` | Same usage |
| `cudaEventCreate` | `hipEventCreate` | |
| `cudaEventRecord` | `hipEventRecord` | |
| `cudaEventElapsedTime` | `hipEventElapsedTime` | Returns milliseconds |
| `cudaEventDestroy` | `hipEventDestroy` | |

### Error Handling

| CUDA API | HIP Equivalent | Notes |
|----------|---------------|-------|
| `cudaError_t` | `hipError_t` | Equivalent enum |
| `cudaGetErrorString` | `hipGetErrorString` | Same |
| `cudaSuccess` | `hipSuccess` | Same |

### Compile Toolchain

| CUDA Tool | HIP/AMD Equivalent |
|-----------|-------------------|
| `nvcc` | `hipcc` |
| Nsight Compute | ROCProfiler / Omniperf |
| Nsight Systems | ROCProfiler / ROCm Bandwidth Test |
| `nvprof` | `rocprof` |

### Thread Indexing

| CUDA | HIP | Notes |
|------|-----|-------|
| `blockIdx.x` | `hipBlockIdx_x` | HIP uses macros in device code |
| `threadIdx.x` | `hipThreadIdx_x` | |
| `blockDim.x` | `hipBlockDim_x` | |
| `gridDim.x` | `hipGridDim_x` | |
| `__shared__` | `__shared__` | Same keyword |
| `__syncthreads()` | `__syncthreads()` | Same |
| `atomicAdd` | `atomicAdd` | Same |

---

## Warp (CUDA) vs Wavefront (AMD GCN/RDNA)

| Property | CUDA (NVIDIA) | AMD GCN / RDNA |
|----------|--------------|----------------|
| Unit name | Warp | Wavefront |
| Size | **32 threads** | **64 threads** |
| `warpSize` constant | 32 | 64 |
| Warp shuffles | `__shfl_down_sync`, `__ballot_sync` | `__shfl_down`, HIP wrappers |
| Warp reduction | Needs mask `0xffffffff` | No mask needed in HIP |

**Critical portability rule:** Never hardcode `32` as the warp size.
Always use `warpSize` or `hipWarpSize` to ensure correct behavior on both platforms.
Algorithms like reduction and scan must handle both 32-thread (CUDA) and 64-thread (AMD) wavefronts.

Example:
```cpp
// Correct (portable):
for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);

// Wrong (CUDA-only, breaks on AMD):
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

---

## NOT_VALIDATED Policy (Detail)

GPU Insight Lab v0.1.0 was developed and tested on NVIDIA hardware only.
The HIP files compile with `hipcc` but have not been run on actual AMD hardware.
We include them to demonstrate awareness of cross-vendor portability, not to claim AMD support.

All AMD-related findings from `collectors/amd_collector.py` are tagged
`validation_status = "NOT_VALIDATED"` regardless of whether data collection succeeded.
