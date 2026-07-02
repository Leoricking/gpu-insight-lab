// GPU Insight Lab - Image Grayscale Benchmark
// CPU baseline + CUDA grayscale. Batch. Correctness. No OpenCV dependency.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "../include/cuda_check.cuh"
#include "../include/statistics.hpp"
#include "../include/benchmark_result.hpp"
#include "../include/timer.hpp"

// Y = 0.2989*R + 0.5870*G + 0.1140*B (BT.601 luma)
__global__ void grayscale_kernel(const unsigned char* rgb, unsigned char* gray,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int px = y * width + x;
    const unsigned char* p = rgb + px * 3;
    float luma = 0.2989f * p[0] + 0.5870f * p[1] + 0.1140f * p[2];
    gray[px] = (unsigned char)fminf(luma + 0.5f, 255.0f);
}

static void cpu_grayscale(const unsigned char* rgb, unsigned char* gray, int n_pixels) {
    for (int i = 0; i < n_pixels; i++) {
        const unsigned char* p = rgb + i * 3;
        float luma = 0.2989f * p[0] + 0.5870f * p[1] + 0.1140f * p[2];
        gray[i] = (unsigned char)(luma + 0.5f);
    }
}

BenchmarkResult run_image_grayscale(int width, int height, int batch, int repeat, int warmup) {
    BenchmarkResult result;
    result.test_name = "image_grayscale";
    result.data_type = "uint8";
    result.input_size = (long long)width * height * batch;
    result.measured_runs = repeat;
    result.warmup_runs = warmup;

    int n_pixels = width * height;
    size_t rgb_bytes = n_pixels * 3;
    size_t gray_bytes = n_pixels;

    std::vector<unsigned char> h_rgb(rgb_bytes * batch);
    std::vector<unsigned char> h_gray_cpu(n_pixels * batch);
    std::vector<unsigned char> h_gray_gpu(n_pixels);
    // Fill with synthetic data
    for (size_t i = 0; i < h_rgb.size(); i++) h_rgb[i] = (unsigned char)(i % 256);

    // CPU baseline (single image for timing reference)
    WallTimer cpu_timer;
    cpu_timer.start();
    for (int b = 0; b < batch; b++)
        cpu_grayscale(h_rgb.data() + b * rgb_bytes, h_gray_cpu.data() + b * n_pixels, n_pixels);
    cpu_timer.stop();
    result.cpu_time_ms = cpu_timer.elapsed_ms() / batch;

    unsigned char *d_rgb = nullptr, *d_gray = nullptr;
    if (!CUDA_CHECK_NF(cudaMalloc(&d_rgb, rgb_bytes)) ||
        !CUDA_CHECK_NF(cudaMalloc(&d_gray, gray_bytes))) {
        result.error = "cudaMalloc failed";
        if (d_rgb) cudaFree(d_rgb);
        if (d_gray) cudaFree(d_gray);
        return result;
    }
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), rgb_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    result.block_size = 16;
    result.grid_size = grid.x * grid.y;

    for (int i = 0; i < warmup; i++)
        grayscale_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    std::vector<double> measurements(repeat);
    for (int iter = 0; iter < repeat; iter++) {
        CUDA_CHECK(cudaEventRecord(ev0));
        grayscale_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        measurements[iter] = ms;
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    CUDA_CHECK(cudaMemcpy(h_gray_gpu.data(), d_gray, gray_bytes, cudaMemcpyDeviceToHost));

    // Correctness: compare GPU output vs CPU reference for first image (allow rounding ±1)
    int max_diff = 0;
    for (int i = 0; i < n_pixels; i++) {
        int diff = abs((int)h_gray_gpu[i] - (int)h_gray_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    result.correctness_pass = (max_diff <= 1) ? 1 : 0;
    result.max_error = (double)max_diff;

    Statistics stats = compute_statistics(measurements);
    result.mean = stats.mean; result.median = stats.median;
    result.min_val = stats.min_val; result.max_val = stats.max_val;
    result.standard_deviation = stats.std_dev;
    result.raw_measurements = measurements;
    result.gpu_time_ms = stats.mean;
    result.end_to_end_time_ms = stats.mean;

    // Bandwidth: read 3 channels, write 1
    double total_bytes = (double)n_pixels * (3 + 1);
    result.bandwidth_gbps = (total_bytes / (stats.mean / 1000.0)) / 1e9;
    if (result.cpu_time_ms > 0.0)
        result.speedup = result.cpu_time_ms / stats.mean;

    result.notes = "CUDA grayscale BT.601 luma, 16x16 block";
    cudaFree(d_rgb); cudaFree(d_gray);
    return result;
}
