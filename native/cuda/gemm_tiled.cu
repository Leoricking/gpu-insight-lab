// GPU Insight Lab - Tiled GEMM benchmark
// Shared-memory tiled GEMM. Supports non-tile-divisible sizes via boundary check.
// NOTE: Diagnostic tool only. Not compared to cuBLAS.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "../include/cuda_check.cuh"
#include "../include/statistics.hpp"
#include "../include/benchmark_result.hpp"

#define TILE_SIZE 16

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A
        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
        // Load tile of B
        if (col < N && (t * TILE_SIZE + ty) < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

BenchmarkResult run_gemm_tiled(int M, int K, int N, int repeat, int warmup) {
    BenchmarkResult result;
    result.test_name = "gemm_tiled";
    result.data_type = "float32";
    result.input_size = (long long)M * K + (long long)K * N;
    result.block_size = TILE_SIZE;
    result.measured_runs = repeat;
    result.warmup_runs = warmup;
    result.notes = "shared-memory tiled GEMM; NOT compared against cuBLAS";

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_ref(M * N, 0.0f);
    for (int i = 0; i < M * K; i++) h_A[i] = 0.01f * (i % 100);
    for (int i = 0; i < K * N; i++) h_B[i] = 0.01f * ((i + 1) % 100);

    int check_M = std::min(M, 16), check_N = std::min(N, 16);
    for (int r = 0; r < check_M; r++)
        for (int c = 0; c < check_N; c++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += h_A[r * K + k] * h_B[k * N + c];
            h_ref[r * N + c] = s;
        }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (!CUDA_CHECK_NF(cudaMalloc(&d_A, bytes_A)) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_B, bytes_B)) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_C, bytes_C))) {
        result.error = "cudaMalloc failed";
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        return result;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    result.grid_size = grid.x * grid.y;

    for (int i = 0; i < warmup; i++)
        gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<double> measurements(repeat);
    for (int iter = 0; iter < repeat; iter++) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        measurements[iter] = ms;
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    std::vector<float> h_C_gpu(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int r = 0; r < check_M; r++)
        for (int c = 0; c < check_N; c++) {
            float diff = fabsf(h_C_gpu[r * N + c] - h_ref[r * N + c]);
            if (diff > max_err) max_err = diff;
        }
    result.correctness_pass = (max_err < 0.01f * K) ? 1 : 0;
    result.max_error = max_err;

    Statistics stats = compute_statistics(measurements);
    result.mean = stats.mean; result.median = stats.median;
    result.min_val = stats.min_val; result.max_val = stats.max_val;
    result.standard_deviation = stats.std_dev;
    result.raw_measurements = measurements;
    result.gpu_time_ms = stats.mean;
    result.end_to_end_time_ms = stats.mean;

    double flops = 2.0 * M * K * N;
    result.throughput = (flops / (stats.mean / 1000.0)) / 1e9;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return result;
}
