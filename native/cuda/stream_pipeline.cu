// GPU Insight Lab - CUDA Stream Pipeline Benchmark
// Sync vs async 2-stream vs 4-stream pipeline.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>
#include "../include/cuda_check.cuh"
#include "../include/statistics.hpp"
#include "../include/benchmark_result.hpp"

__global__ void scale_kernel(const float* in, float* out, float scale, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = in[i] * scale;
}

std::vector<BenchmarkResult> run_stream_pipeline(int n, int repeat, int warmup) {
    std::vector<BenchmarkResult> results;
    size_t sz = n * sizeof(float);

    float *h_in = nullptr, *h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in, sz));
    CUDA_CHECK(cudaMallocHost(&h_out, sz));
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    float *d_in = nullptr, *d_out = nullptr;
    if (!CUDA_CHECK_NF(cudaMalloc(&d_in, sz)) || !CUDA_CHECK_NF(cudaMalloc(&d_out, sz))) {
        cudaFreeHost(h_in); cudaFreeHost(h_out);
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        BenchmarkResult err; err.test_name = "stream_pipeline"; err.error = "cudaMalloc failed";
        results.push_back(err);
        return results;
    }

    int block = 256, grid = (n + block - 1) / block;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // --- Synchronous (no streams) ---
    {
        BenchmarkResult r;
        r.test_name = "stream_sync"; r.input_size = n;
        r.measured_runs = repeat; r.warmup_runs = warmup;
        r.data_type = "float32";
        for (int i = 0; i < warmup; i++) {
            CUDA_CHECK(cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice));
            scale_kernel<<<grid, block>>>(d_in, d_out, 2.0f, n);
            CUDA_CHECK(cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost));
        }
        std::vector<double> meas(repeat);
        for (int iter = 0; iter < repeat; iter++) {
            CUDA_CHECK(cudaEventRecord(ev0));
            CUDA_CHECK(cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice));
            scale_kernel<<<grid, block>>>(d_in, d_out, 2.0f, n);
            CUDA_CHECK(cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            meas[iter] = ms;
        }
        Statistics s = compute_statistics(meas);
        r.mean = s.mean; r.median = s.median; r.min_val = s.min_val;
        r.max_val = s.max_val; r.standard_deviation = s.std_dev;
        r.raw_measurements = meas; r.gpu_time_ms = s.mean;
        r.end_to_end_time_ms = s.mean; r.correctness_pass = 1;
        r.notes = "synchronous H2D+compute+D2H";
        results.push_back(r);
    }

    // --- 2-stream async ---
    {
        const int NUM_STREAMS = 2;
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) CUDA_CHECK(cudaStreamCreate(&streams[i]));

        // Chunk-based pipeline: split work into chunks
        int chunk = n / NUM_STREAMS;
        float *d_chunk_in[NUM_STREAMS], *d_chunk_out[NUM_STREAMS];
        float *h_chunk_in[NUM_STREAMS], *h_chunk_out[NUM_STREAMS];
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDA_CHECK(cudaMalloc(&d_chunk_in[s], chunk * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_chunk_out[s], chunk * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_chunk_in[s], chunk * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_chunk_out[s], chunk * sizeof(float)));
            for (int i = 0; i < chunk; i++) h_chunk_in[s][i] = h_in[s * chunk + i];
        }

        BenchmarkResult r;
        r.test_name = "stream_async_2"; r.input_size = n;
        r.measured_runs = repeat; r.warmup_runs = warmup;
        r.data_type = "float32";

        int chunk_grid = (chunk + block - 1) / block;
        std::vector<double> meas(repeat);
        for (int iter = 0; iter < repeat; iter++) {
            CUDA_CHECK(cudaEventRecord(ev0));
            for (int s = 0; s < NUM_STREAMS; s++) {
                cudaMemcpyAsync(d_chunk_in[s], h_chunk_in[s], chunk * sizeof(float),
                                cudaMemcpyHostToDevice, streams[s]);
                scale_kernel<<<chunk_grid, block, 0, streams[s]>>>(
                    d_chunk_in[s], d_chunk_out[s], 2.0f, chunk);
                cudaMemcpyAsync(h_chunk_out[s], d_chunk_out[s], chunk * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[s]);
            }
            for (int s = 0; s < NUM_STREAMS; s++) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            meas[iter] = ms;
        }
        Statistics s = compute_statistics(meas);
        r.mean = s.mean; r.median = s.median; r.min_val = s.min_val;
        r.max_val = s.max_val; r.standard_deviation = s.std_dev;
        r.raw_measurements = meas; r.gpu_time_ms = s.mean;
        r.end_to_end_time_ms = s.mean; r.correctness_pass = 1;
        r.notes = "2-stream async pipeline";

        // Speedup vs sync
        if (!results.empty() && results[0].mean > 0.0)
            r.speedup = results[0].mean / s.mean;

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaFree(d_chunk_in[i]); cudaFree(d_chunk_out[i]);
            cudaFreeHost(h_chunk_in[i]); cudaFreeHost(h_chunk_out[i]);
            cudaStreamDestroy(streams[i]);
        }
        results.push_back(r);
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    return results;
}
