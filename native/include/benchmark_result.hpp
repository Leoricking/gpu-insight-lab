// GPU Insight Lab - BenchmarkResult struct
#pragma once
#include <string>
#include <vector>
#include <ctime>

struct BenchmarkResult {
    // Schema and identity
    std::string schema_version     = "1.0";
    double      timestamp          = 0.0;
    std::string hostname;
    std::string os;

    // GPU info
    std::string gpu_name;
    std::string gpu_uuid;
    std::string driver_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    std::string compute_capability;

    // Test configuration
    std::string test_name;
    std::string data_type          = "float32";
    long long   input_size         = 0;
    int         block_size         = 0;
    int         grid_size          = 0;
    int         warmup_runs        = 0;
    int         measured_runs      = 0;

    // Timing (ms)
    double cpu_time_ms             = -1.0;
    double gpu_time_ms             = -1.0;
    double transfer_time_ms        = -1.0;
    double end_to_end_time_ms      = -1.0;

    // Performance
    double throughput              = -1.0;  // GFLOPS or elements/s
    double bandwidth_gbps          = -1.0;
    double speedup                 = -1.0;

    // Correctness
    int    correctness_pass        = -1;   // -1=not checked, 0=fail, 1=pass
    double max_error               = -1.0;

    // Statistics
    double mean                    = -1.0;
    double median                  = -1.0;
    double min_val                 = -1.0;
    double max_val                 = -1.0;
    double standard_deviation      = -1.0;
    std::vector<double> raw_measurements;

    std::string notes;
    std::string error;
};
