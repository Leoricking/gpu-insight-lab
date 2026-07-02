// GPU Insight Lab - Vector Add Benchmark
// CPU baseline + naive CUDA + grid-stride CUDA
// Block sizes: 128, 256, 512. Correctness check. 10 measured runs + warmup.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include "../include/cuda_check.cuh"
#include "../include/statistics.hpp"
#include "../include/benchmark_result.hpp"
#include "../include/timer.hpp"

// --- Kernels ---

__global__ void vector_add_naive(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void vector_add_grid_stride(const float* a, const float* b, float* c, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride)
        c[idx] = a[idx] + b[idx];
}

// CPU baseline
static void cpu_vector_add(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

BenchmarkResult run_vector_add(int n, int block_size, int repeat, int warmup) {
    BenchmarkResult result;
    result.test_name = "vector_add";
    result.data_type = "float32";
    result.input_size = n;
    result.block_size = block_size;
    result.measured_runs = repeat;
    result.warmup_runs = warmup;

    size_t bytes = n * sizeof(float);
    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)(i % 100) * 0.01f;
        h_b[i] = (float)((n - i) % 100) * 0.01f;
    }

    // CPU reference
    WallTimer cpu_timer;
    cpu_timer.start();
    cpu_vector_add(h_a.data(), h_b.data(), h_ref.data(), n);
    cpu_timer.stop();
    result.cpu_time_ms = cpu_timer.elapsed_ms();

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    if (!CUDA_CHECK_NF(cudaMalloc(&d_a, bytes)) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_b, bytes)) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_c, bytes))) {
        result.error = "cudaMalloc failed";
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        return result;
    }

    // Pageable H2D transfer timing
    WallTimer xfer_timer;
    xfer_timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    xfer_timer.stop();
    result.transfer_time_ms = xfer_timer.elapsed_ms();

    int grid_size = (n + block_size - 1) / block_size;
    result.grid_size = grid_size;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        vector_add_grid_stride<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measured runs (CUDA events for precise GPU timing)
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<double> measurements(repeat);
    for (int i = 0; i < repeat; i++) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        vector_add_grid_stride<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        measurements[i] = ms;
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    // D2H
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Correctness
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(h_c[i] - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }
    result.correctness_pass = (max_err < 1e-5f) ? 1 : 0;
    result.max_error = max_err;

    // Statistics
    Statistics stats = compute_statistics(measurements);
    result.mean              = stats.mean;
    result.median            = stats.median;
    result.min_val           = stats.min_val;
    result.max_val           = stats.max_val;
    result.standard_deviation = stats.std_dev;
    result.raw_measurements  = measurements;
    result.gpu_time_ms       = stats.mean;
    result.end_to_end_time_ms = result.transfer_time_ms + stats.mean;

    // Bandwidth: read 2, write 1 array
    double total_bytes = 3.0 * n * sizeof(float);
    result.bandwidth_gbps = (total_bytes / (stats.mean / 1000.0)) / 1e9;

    // Speedup
    if (result.cpu_time_ms > 0.0)
        result.speedup = result.cpu_time_ms / stats.mean;

    result.notes = "grid-stride kernel; correctness vs CPU baseline";

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return result;
}
