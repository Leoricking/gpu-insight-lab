// GPU Insight Lab - Native Benchmark Entry Point
// Parses CLI args and dispatches to individual benchmark modules.
// Output: JSON to stdout or file.
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>
#include "../include/benchmark_result.hpp"
#include "../common/json_writer.hpp"

// Forward declarations
std::string collect_device_info_json();
BenchmarkResult run_vector_add(int n, int block_size, int repeat, int warmup);
BenchmarkResult run_reduction(int n, int block_size, int repeat, int warmup);
BenchmarkResult run_transpose(int matrix_dim, int repeat, int warmup);
BenchmarkResult run_gemm_naive(int M, int K, int N, int block_dim, int repeat, int warmup);
BenchmarkResult run_gemm_tiled(int M, int K, int N, int repeat, int warmup);
std::vector<BenchmarkResult> run_memory_bandwidth(int repeat, int warmup);
std::vector<BenchmarkResult> run_stream_pipeline(int n, int repeat, int warmup);
BenchmarkResult run_image_grayscale(int width, int height, int batch, int repeat, int warmup);

static void print_usage(const char* prog) {
    fprintf(stderr,
        "GPU Insight Lab - Native Benchmark v0.1.0\n"
        "Usage: %s [options]\n"
        "Options:\n"
        "  --device-info          Print CUDA device info as JSON\n"
        "  --quick                Run quick benchmark suite\n"
        "  --full                 Run full benchmark suite\n"
        "  --test NAME            Run a specific test\n"
        "  --output FILE          Write JSON output to FILE instead of stdout\n"
        "  --repeat N             Number of measured runs (default: 10)\n"
        "  --warmup N             Number of warmup runs (default: 3)\n"
        "  --size N               Input size (elements)\n"
        "  --block-size N         CUDA block size\n"
        "  --help                 Show this message\n"
        "\nAvailable tests: vector_add, reduction, transpose, gemm_naive, gemm_tiled,\n"
        "                  memory_bandwidth, stream_pipeline, image_grayscale\n",
        prog);
}

static std::string get_hostname() {
    char buf[256] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf); GetComputerNameA(buf, &sz);
#else
    gethostname(buf, sizeof(buf));
#endif
    return buf[0] ? std::string(buf) : "unknown";
}

static std::string get_os() {
#ifdef _WIN32
    return "Windows";
#elif __linux__
    return "Linux";
#elif __APPLE__
    return "macOS";
#else
    return "Unknown";
#endif
}

static void populate_gpu_info(BenchmarkResult& r) {
    r.hostname = get_hostname();
    r.os = get_os();
    cudaDeviceProp prop;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        r.gpu_name = prop.name;
        r.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    }
    int driver_ver = 0, runtime_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);
    r.cuda_driver_version = std::to_string(driver_ver / 1000) + "." + std::to_string((driver_ver % 1000) / 10);
    r.cuda_runtime_version = std::to_string(runtime_ver / 1000) + "." + std::to_string((runtime_ver % 1000) / 10);
    r.timestamp = (double)time(nullptr);
}

static void populate_results(std::vector<BenchmarkResult>& results) {
    for (auto& r : results) populate_gpu_info(r);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    bool do_device_info = false;
    bool do_quick = false;
    bool do_full = false;
    std::string test_name;
    std::string output_file;
    int repeat = 10, warmup = 3, size = 0, block_size = 256;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--device-info") == 0) do_device_info = true;
        else if (strcmp(argv[i], "--quick") == 0)       do_quick = true;
        else if (strcmp(argv[i], "--full") == 0)         do_full = true;
        else if (strcmp(argv[i], "--help") == 0)         { print_usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc)       test_name = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)     output_file = argv[++i];
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)     repeat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)     warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc)       size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) block_size = atoi(argv[++i]);
    }

    // Check CUDA availability
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        fprintf(stdout, "{\"error\": \"No CUDA devices found\"}\n");
        return 1;
    }

    std::string json_out;

    if (do_device_info) {
        json_out = collect_device_info_json();
    }
    else if (do_quick || (!do_full && test_name.empty())) {
        // Quick: vector_add + reduction + memory_bandwidth
        std::vector<BenchmarkResult> results;
        int n = size > 0 ? size : 1 << 24;

        BenchmarkResult va = run_vector_add(n, block_size, repeat, warmup);
        populate_gpu_info(va);
        results.push_back(va);

        BenchmarkResult red = run_reduction(n, block_size, repeat, warmup);
        populate_gpu_info(red);
        results.push_back(red);

        auto bw_results = run_memory_bandwidth(repeat, warmup);
        populate_results(bw_results);
        results.insert(results.end(), bw_results.begin(), bw_results.end());

        json_out = "{\"mode\": \"quick\", \"results\": " + results_to_json_array(results) + "}";
    }
    else if (do_full) {
        std::vector<BenchmarkResult> results;
        int n = size > 0 ? size : 1 << 24;
        int mat_dim = size > 0 ? size : 512;

        auto add_one = [&](BenchmarkResult r) { populate_gpu_info(r); results.push_back(r); };

        add_one(run_vector_add(n, block_size, repeat, warmup));
        add_one(run_reduction(n, block_size, repeat, warmup));
        add_one(run_transpose(mat_dim, repeat, warmup));
        add_one(run_gemm_naive(mat_dim, mat_dim, mat_dim, 16, repeat, warmup));
        add_one(run_gemm_tiled(mat_dim, mat_dim, mat_dim, repeat, warmup));

        auto bw_results = run_memory_bandwidth(repeat, warmup);
        populate_results(bw_results);
        results.insert(results.end(), bw_results.begin(), bw_results.end());

        auto st_results = run_stream_pipeline(n, repeat, warmup);
        populate_results(st_results);
        results.insert(results.end(), st_results.begin(), st_results.end());

        add_one(run_image_grayscale(1920, 1080, 1, repeat, warmup));

        json_out = "{\"mode\": \"full\", \"results\": " + results_to_json_array(results) + "}";
    }
    else if (!test_name.empty()) {
        std::vector<BenchmarkResult> results;
        int n = size > 0 ? size : 1 << 24;
        int mat_dim = size > 0 ? size : 512;

        if (test_name == "vector_add") {
            results.push_back(run_vector_add(n, block_size, repeat, warmup));
        } else if (test_name == "reduction") {
            results.push_back(run_reduction(n, block_size, repeat, warmup));
        } else if (test_name == "transpose") {
            results.push_back(run_transpose(mat_dim, repeat, warmup));
        } else if (test_name == "gemm_naive") {
            results.push_back(run_gemm_naive(mat_dim, mat_dim, mat_dim, 16, repeat, warmup));
        } else if (test_name == "gemm_tiled") {
            results.push_back(run_gemm_tiled(mat_dim, mat_dim, mat_dim, repeat, warmup));
        } else if (test_name == "memory_bandwidth") {
            auto r = run_memory_bandwidth(repeat, warmup);
            results.insert(results.end(), r.begin(), r.end());
        } else if (test_name == "stream_pipeline") {
            auto r = run_stream_pipeline(n, repeat, warmup);
            results.insert(results.end(), r.begin(), r.end());
        } else if (test_name == "image_grayscale") {
            results.push_back(run_image_grayscale(1920, 1080, 1, repeat, warmup));
        } else {
            fprintf(stderr, "Unknown test: %s\n", test_name.c_str());
            return 1;
        }
        populate_results(results);
        json_out = results.size() == 1
            ? result_to_json(results[0])
            : results_to_json_array(results);
    }
    else {
        print_usage(argv[0]); return 1;
    }

    if (!output_file.empty()) {
        std::ofstream ofs(output_file);
        if (!ofs) {
            fprintf(stderr, "Cannot write to %s\n", output_file.c_str());
            return 1;
        }
        ofs << json_out << "\n";
    } else {
        printf("%s\n", json_out.c_str());
    }

    return 0;
}
