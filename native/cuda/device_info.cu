// GPU Insight Lab - Device Info Collection
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <iomanip>

std::string collect_device_info_json() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return "{\"error\": \"No CUDA devices found\", \"device_count\": 0}";
    }

    std::ostringstream j;
    j << "{\n";
    j << "  \"device_count\": " << device_count << ",\n";
    j << "  \"devices\": [\n";

    for (int d = 0; d < device_count; ++d) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);

        // CUDA driver and runtime versions
        int driver_ver = 0, runtime_ver = 0;
        cudaDriverGetVersion(&driver_ver);
        cudaRuntimeGetVersion(&runtime_ver);

        if (d > 0) j << ",\n";
        j << "    {\n";
        j << "      \"device_index\": " << d << ",\n";
        j << "      \"name\": \"" << prop.name << "\",\n";
        j << "      \"uuid\": \"GPU-" << d << "\",\n";
        j << "      \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
        j << "      \"total_global_memory_mb\": " << std::fixed << std::setprecision(1)
          << (prop.totalGlobalMem / (1024.0 * 1024.0)) << ",\n";
        j << "      \"sm_count\": " << prop.multiProcessorCount << ",\n";
        j << "      \"max_threads_per_sm\": " << prop.maxThreadsPerMultiProcessor << ",\n";
        j << "      \"max_threads_per_block\": " << prop.maxThreadsPerBlock << ",\n";
        j << "      \"warp_size\": " << prop.warpSize << ",\n";
        j << "      \"shared_memory_per_block_kb\": "
          << (prop.sharedMemPerBlock / 1024.0) << ",\n";
        j << "      \"l2_cache_size_mb\": "
          << (prop.l2CacheSize / (1024.0 * 1024.0)) << ",\n";
        j << "      \"memory_clock_mhz\": " << (prop.memoryClockRate / 1000) << ",\n";
        j << "      \"memory_bus_width_bits\": " << prop.memoryBusWidth << ",\n";
        j << "      \"peak_memory_bandwidth_gbps\": "
          << std::setprecision(2)
          << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << ",\n";
        j << "      \"clock_rate_mhz\": " << (prop.clockRate / 1000) << ",\n";
        j << "      \"ecc_enabled\": " << (prop.ECCEnabled ? "true" : "false") << ",\n";
        j << "      \"unified_addressing\": " << (prop.unifiedAddressing ? "true" : "false") << ",\n";
        j << "      \"concurrent_kernels\": " << (prop.concurrentKernels ? "true" : "false") << ",\n";
        j << "      \"cuda_driver_version\": \"" << (driver_ver / 1000) << "." << ((driver_ver % 1000) / 10) << "\",\n";
        j << "      \"cuda_runtime_version\": \"" << (runtime_ver / 1000) << "." << ((runtime_ver % 1000) / 10) << "\"\n";
        j << "    }";
    }
    j << "\n  ]\n}";
    return j.str();
}
