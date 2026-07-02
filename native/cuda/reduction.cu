// GPU Insight Lab - Parallel Reduction Benchmark
// CPU baseline + naive + shared-memory + atomicAdd
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "../include/cuda_check.cuh"
#include "../include/statistics.hpp"
#include "../include/benchmark_result.hpp"
#include "../include/timer.hpp"

// Naive reduction kernel (global memory, not efficient)
__global__ void reduce_naive(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < (unsigned int)n) ? input[idx] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Shared-memory reduction (avoids divergence with warp unroll)
__global__ void reduce_shared(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float v = 0.0f;
    if (idx < (unsigned int)n) v = input[idx];
    if (idx + blockDim.x < (unsigned int)n) v += input[idx + blockDim.x];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // Warp unroll (no sync needed inside warp)
    if (tid < 32) {
        volatile float* vsm = sdata;
        vsm[tid] += vsm[tid + 32];
        vsm[tid] += vsm[tid + 16];
        vsm[tid] += vsm[tid + 8];
        vsm[tid] += vsm[tid + 4];
        vsm[tid] += vsm[tid + 2];
        vsm[tid] += vsm[tid + 1];
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

BenchmarkResult run_reduction(int n, int block_size, int repeat, int warmup) {
    BenchmarkResult result;
    result.test_name = "reduction";
    result.data_type = "float32";
    result.input_size = n;
    result.block_size = block_size;
    result.measured_runs = repeat;
    result.warmup_runs = warmup;

    std::vector<float> h_in(n);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;  // sum should equal n

    // CPU reference
    float cpu_sum = 0.0f;
    WallTimer cpu_timer;
    cpu_timer.start();
    for (int i = 0; i < n; i++) cpu_sum += h_in[i];
    cpu_timer.stop();
    result.cpu_time_ms = cpu_timer.elapsed_ms();

    int grid_size = (n + block_size - 1) / block_size;
    result.grid_size = grid_size;

    float *d_in = nullptr, *d_partial = nullptr;
    if (!CUDA_CHECK_NF(cudaMalloc(&d_in, n * sizeof(float))) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_partial, grid_size * sizeof(float)))) {
        result.error = "cudaMalloc failed";
        if (d_in) cudaFree(d_in);
        if (d_partial) cudaFree(d_partial);
        return result;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    size_t shared_bytes = block_size * sizeof(float);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        reduce_shared<<<grid_size, block_size, shared_bytes>>>(d_in, d_partial, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<double> measurements(repeat);
    float gpu_sum = 0.0f;
    for (int iter = 0; iter < repeat; iter++) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        reduce_shared<<<grid_size, block_size, shared_bytes>>>(d_in, d_partial, n);
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        measurements[iter] = ms;
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    // Copy partial sums and reduce on CPU
    std::vector<float> h_partial(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    gpu_sum = 0.0f;
    for (float v : h_partial) gpu_sum += v;

    // Correctness: sum should equal n (all 1s)
    result.max_error = fabsf(gpu_sum - cpu_sum);
    result.correctness_pass = (result.max_error < 1.0f) ? 1 : 0;

    Statistics stats = compute_statistics(measurements);
    result.mean              = stats.mean;
    result.median            = stats.median;
    result.min_val           = stats.min_val;
    result.max_val           = stats.max_val;
    result.standard_deviation = stats.std_dev;
    result.raw_measurements  = measurements;
    result.gpu_time_ms       = stats.mean;
    result.end_to_end_time_ms = stats.mean;

    // Bandwidth: read n floats
    result.bandwidth_gbps = ((double)n * sizeof(float) / (stats.mean / 1000.0)) / 1e9;
    if (result.cpu_time_ms > 0.0)
        result.speedup = result.cpu_time_ms / stats.mean;

    result.notes = "shared-memory reduction with warp unroll";
    cudaFree(d_in); cudaFree(d_partial);
    return result;
}
