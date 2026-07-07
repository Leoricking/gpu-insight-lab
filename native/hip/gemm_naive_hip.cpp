// GPU Insight Lab - HIP Naive GEMM (NOT_VALIDATED)
// AMD HIP port of the CUDA naive GEMM kernel (C = A * B).
// Status: NOT_VALIDATED — no AMD GPU was available during development.
//
// CUDA to HIP portability notes:
//   cudaMalloc         -> hipMalloc
//   cudaMemcpy         -> hipMemcpy
//   cudaDeviceSynchronize -> hipDeviceSynchronize
//   cudaFree           -> hipFree
//   nvcc               -> hipcc
//   Nsight Compute     -> ROCProfiler / Omniperf
//   cuBLAS             -> hipBLAS (for production GEMM)
//
// Warp size difference:
//   CUDA warp = 32 threads
//   AMD GCN/RDNA wavefront = 64 threads
//   This naive GEMM does not use warp intrinsics, so it is portable.
//   Tiled GEMM must respect wavefront size = 64 on AMD.
//
// Build (requires ROCm):
//   hipcc gemm_naive_hip.cpp -o gemm_naive_hip -D__HIP__
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

#define HIP_CHECK(call) do {                                           \
    hipError_t _err = (call);                                          \
    if (_err != hipSuccess) {                                          \
        fprintf(stderr, "HIP error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, hipGetErrorString(_err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// Naive GEMM: each thread computes one element of C = A * B.
// This is memory-bound; use hipBLAS for production workloads.
__global__ void gemm_naive_hip_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

int main() {
    const int M = 256, K = 256, N = 256;
    std::vector<float> h_A(M * K, 1.0f), h_B(K * N, 1.0f), h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(gemm_naive_hip_kernel, grid, block, 0, 0,
                       d_A, d_B, d_C, M, K, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify: each element should equal K (sum of K ones * K ones)
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = fabsf(h_C[i] - (float)K);
        if (err > max_err) max_err = err;
    }

    printf("HIP Naive GEMM: M=%d K=%d N=%d, max_error=%.6f, %s\n",
           M, K, N, max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return max_err < 1e-3f ? 0 : 1;
}

#else

#include <cstdio>
int main() {
    printf("{\n");
    printf("  \"test\": \"gemm_naive_hip\",\n");
    printf("  \"status\": \"NOT_VALIDATED\",\n");
    printf("  \"note\": \"HIP not available. Build with hipcc -D__HIP__ to enable AMD GPU support.\",\n");
    printf("  \"policy\": \"NOT_VALIDATED when AMD GPU / ROCm unavailable\"\n");
    printf("}\n");
    return 1;
}

#endif
