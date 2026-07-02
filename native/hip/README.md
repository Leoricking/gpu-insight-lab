# HIP/ROCm Support — GPU Insight Lab

**Status: NOT_VALIDATED** — No AMD GPU was available during development of v0.1.0.
The HIP vector_add stub is provided as a portability reference.

---

## Build Instructions

```bash
# Ensure ROCm is installed: https://rocmdocs.amd.com
hipcc vector_add_hip.cpp -o vector_add_hip -D__HIP__
./vector_add_hip
```

## CUDA vs HIP API Mapping

| CUDA API | HIP API | Notes |
|----------|---------|-------|
| `cudaMalloc` | `hipMalloc` | Same signature |
| `cudaFree` | `hipFree` | Same signature |
| `cudaMemcpy` | `hipMemcpy` | Same signature |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Same |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | Same enum value |
| `cudaError_t` | `hipError_t` | Equivalent enum |
| `cudaGetErrorString` | `hipGetErrorString` | Same |
| `cudaEvent_t` | `hipEvent_t` | Same usage |
| `cudaEventRecord` | `hipEventRecord` | Same |
| `cudaEventElapsedTime` | `hipEventElapsedTime` | Same (ms) |
| `cudaStream_t` | `hipStream_t` | Same |
| `cudaMemGetInfo` | `hipMemGetInfo` | Same |
| `__global__` | `__global__` | Same keyword |
| `blockIdx.x` | `hipBlockIdx_x` | HIP uses macros |
| `threadIdx.x` | `hipThreadIdx_x` | HIP uses macros |
| `blockDim.x` | `hipBlockDim_x` | HIP uses macros |
| `gridDim.x` | `hipGridDim_x` | HIP uses macros |
| `__shared__` | `__shared__` | Same |
| `__syncthreads()` | `__syncthreads()` | Same |
| `atomicAdd` | `atomicAdd` | Same |

## Warp vs Wavefront

| CUDA | AMD ROCm |
|------|----------|
| Warp: 32 threads | Wavefront: 64 threads (GCN/RDNA) |
| `warpSize` = 32 | `warpSize` = 64 |
| Warp-level intrinsics: `__shfl_*`, `__ballot_sync` | Wave-level: `__shfl_*` (HIP provides wrappers) |

> **Important:** Algorithms that hardcode warp size = 32 will produce incorrect results on AMD GPUs.
> Always use `blockDim.x` for reduction loops unless you explicitly handle both 32 and 64.

## Portability Notes

1. Use `HIP_CHECK` macros instead of `CUDA_CHECK` for AMD builds.
2. HIP supports `hipcc` as a drop-in replacement for `nvcc` for many workloads.
3. ROCm provides `hipBLAS`, `hipFFT`, `hipSolver` analogous to cuBLAS, cuFFT, cuSolver.
4. Memory model differences: ROCm may have different L2 behavior; profile with `rocprof` or Omniperf.
5. `hipMallocHost` (pinned memory) works the same as `cudaMallocHost`.

## Why This is NOT_VALIDATED

GPU Insight Lab v0.1.0 was developed and tested on NVIDIA hardware only.
The HIP stub compiles with `hipcc` but has not been run on actual AMD hardware.
We include it to demonstrate awareness of cross-vendor portability, not to claim AMD support.
