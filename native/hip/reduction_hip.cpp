// GPU Insight Lab - HIP Reduction (NOT_VALIDATED)
// AMD HIP port of the CUDA reduction kernel.
// Status: NOT_VALIDATED — no AMD GPU was available during development.
//
// CUDA to HIP portability notes:
//   cudaMalloc         -> hipMalloc
//   cudaMemcpy         -> hipMemcpy
//   cudaDeviceSynchronize -> hipDeviceSynchronize
//   cudaFree           -> hipFree
//   __shfl_down_sync   -> __shfl_down (HIP warp shuffle)
//   nvcc               -> hipcc
//   Nsight Compute     -> ROCProfiler / Omniperf
//
// Warp size difference:
//   CUDA warp = 32 threads
//   AMD GCN/RDNA wavefront = 64 threads
//   This implementation uses warpSize to adapt at compile time.
//   NEVER hardcode 32 if targeting both CUDA and AMD.
//
// Build (requires ROCm):
//   hipcc reduction_hip.cpp -o reduction_hip -D__HIP__
//
// See docs/CUDA_VS_HIP.md for full API mapping table.

#ifndef __HIP_PLATFORM_AMD__
// Compile note: This file is a HIP portability reference.
// Without AMD ROCm or hipcc, it will compile via the #else branch below.
#endif

#ifdef __HIP__
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <numeric>

#define HIP_CHECK(call) do {                                           \
    hipError_t _err = (call);                                          \
    if (_err != hipSuccess) {                                          \
        fprintf(stderr, "HIP error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, hipGetErrorString(_err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// Warp-level reduction using HIP shuffle.
// Note: AMD wavefront = 64, CUDA warp = 32.
// hipWarpSize resolves to the correct value at compile time.
__device__ float warp_reduce(float val) {
    for (int offset = hipWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void reduction_hip_kernel(const float* input, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and reduce per-thread
    float sum = 0.0f;
    while (gid < n) {
        sum += input[gid];
        gid += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Shared memory reduction
    for (int stride = blockDim.x / 2; stride > hipWarpSize; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Warp-level reduction for final warp
    if (tid < hipWarpSize) {
        sdata[tid] += sdata[tid + hipWarpSize];
        float val = sdata[tid];
        val = warp_reduce(val);
        if (tid == 0) partial_sums[blockIdx.x] = val;
    }
}

int main() {
    const int N = 1 << 20;
    std::vector<float> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 0.0f);

    float expected = (float)N * (N - 1) / 2.0f;

    float *d_input, *d_partial;
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(reduction_hip_kernel,
                       dim3(num_blocks), dim3(block_size),
                       block_size * sizeof(float), 0,
                       d_input, d_partial, N);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_partial(num_blocks);
    HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(float), hipMemcpyDeviceToHost));

    float total = 0.0f;
    for (float v : h_partial) total += v;

    float rel_err = fabsf(total - expected) / (fabsf(expected) + 1e-10f);
    printf("HIP Reduction: N=%d, result=%.0f, expected=%.0f, rel_err=%.2e, %s\n",
           N, total, expected, rel_err,
           rel_err < 1e-4f ? "PASS" : "FAIL");

    hipFree(d_input);
    hipFree(d_partial);
    return rel_err < 1e-4f ? 0 : 1;
}

#else

#include <cstdio>
int main() {
    printf("{\n");
    printf("  \"test\": \"reduction_hip\",\n");
    printf("  \"status\": \"NOT_VALIDATED\",\n");
    printf("  \"note\": \"HIP not available. Build with hipcc -D__HIP__ to enable AMD GPU support.\",\n");
    printf("  \"policy\": \"NOT_VALIDATED when AMD GPU / ROCm unavailable\"\n");
    printf("}\n");
    return 1;
}

#endif
