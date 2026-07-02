// GPU Insight Lab - Matrix Transpose Benchmark
// Naive + tiled with padding. Correctness. Effective bandwidth.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "../include/cuda_check.cuh"
#include "../include/statistics.hpp"
#include "../include/benchmark_result.hpp"
#include "../include/timer.hpp"

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Naive transpose
__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < cols && y < rows) out[x * rows + y] = in[y * cols + x];
}

// Tiled transpose with padding to avoid shared memory bank conflicts
__global__ void transpose_tiled(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols)
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

BenchmarkResult run_transpose(int matrix_dim, int repeat, int warmup) {
    int rows = matrix_dim, cols = matrix_dim;
    size_t bytes = (size_t)rows * cols * sizeof(float);

    BenchmarkResult result;
    result.test_name = "transpose";
    result.data_type = "float32";
    result.input_size = (long long)rows * cols;
    result.block_size = TILE_DIM;
    result.measured_runs = repeat;
    result.warmup_runs = warmup;

    std::vector<float> h_in(rows * cols), h_out(rows * cols), h_ref(rows * cols);
    for (int i = 0; i < rows * cols; i++) h_in[i] = (float)i;

    // CPU reference
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            h_ref[c * rows + r] = h_in[r * cols + c];

    float *d_in = nullptr, *d_out = nullptr;
    if (!CUDA_CHECK_NF(cudaMalloc(&d_in, bytes)) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_out, bytes))) {
        result.error = "cudaMalloc failed";
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        return result;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    result.grid_size = grid.x * grid.y;

    // Warmup
    for (int i = 0; i < warmup; i++)
        transpose_tiled<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<double> measurements(repeat);
    for (int iter = 0; iter < repeat; iter++) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        transpose_tiled<<<grid, block>>>(d_in, d_out, rows, cols);
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        measurements[iter] = ms;
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Correctness
    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(h_out[i] - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }
    result.correctness_pass = (max_err < 1e-5f) ? 1 : 0;
    result.max_error = max_err;

    Statistics stats = compute_statistics(measurements);
    result.mean = stats.mean; result.median = stats.median;
    result.min_val = stats.min_val; result.max_val = stats.max_val;
    result.standard_deviation = stats.std_dev;
    result.raw_measurements = measurements;
    result.gpu_time_ms = stats.mean;
    result.end_to_end_time_ms = stats.mean;

    // Effective bandwidth: read + write
    result.bandwidth_gbps = (2.0 * bytes / (stats.mean / 1000.0)) / 1e9;
    result.notes = "tiled transpose with shared memory padding";

    cudaFree(d_in); cudaFree(d_out);
    return result;
}
