// GPU Insight Lab - JSON serialization for BenchmarkResult
#include "json_writer.hpp"
#include <sstream>
#include <iomanip>
#include <ctime>

static std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string dbl(double v, int precision = 4) {
    if (v < -0.5e15) return "null";  // sentinel -1.0 → null
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

static std::string dbl_nullable(double v, int precision = 4) {
    // Values < 0 were used as "not set" sentinels
    if (v < 0.0) return "null";
    return dbl(v, precision);
}

std::string result_to_json(const BenchmarkResult& r, bool pretty) {
    const std::string nl = pretty ? "\n" : "";
    const std::string ind = pretty ? "  " : "";
    std::ostringstream j;
    j << "{" << nl;

    auto kv_str = [&](const std::string& key, const std::string& val) {
        j << ind << "\"" << key << "\": \"" << escape_json_string(val) << "\"," << nl;
    };
    auto kv_num = [&](const std::string& key, const std::string& val) {
        j << ind << "\"" << key << "\": " << val << "," << nl;
    };
    auto kv_bool = [&](const std::string& key, bool val) {
        j << ind << "\"" << key << "\": " << (val ? "true" : "false") << "," << nl;
    };

    kv_str("schema_version", r.schema_version);
    kv_num("timestamp", dbl(r.timestamp, 3));
    kv_str("hostname", r.hostname);
    kv_str("os", r.os);
    kv_str("gpu_name", r.gpu_name);
    kv_str("gpu_uuid", r.gpu_uuid);
    kv_str("driver_version", r.driver_version);
    kv_str("cuda_runtime_version", r.cuda_runtime_version);
    kv_str("cuda_driver_version", r.cuda_driver_version);
    kv_str("compute_capability", r.compute_capability);
    kv_str("test_name", r.test_name);
    kv_str("data_type", r.data_type);
    kv_num("input_size", std::to_string(r.input_size));
    kv_num("block_size", std::to_string(r.block_size));
    kv_num("grid_size", std::to_string(r.grid_size));
    kv_num("warmup_runs", std::to_string(r.warmup_runs));
    kv_num("measured_runs", std::to_string(r.measured_runs));
    kv_num("cpu_time_ms", dbl_nullable(r.cpu_time_ms));
    kv_num("gpu_time_ms", dbl_nullable(r.gpu_time_ms));
    kv_num("transfer_time_ms", dbl_nullable(r.transfer_time_ms));
    kv_num("end_to_end_time_ms", dbl_nullable(r.end_to_end_time_ms));
    kv_num("throughput", dbl_nullable(r.throughput));
    kv_num("bandwidth_gbps", dbl_nullable(r.bandwidth_gbps));
    kv_num("speedup", dbl_nullable(r.speedup));

    // Correctness
    if (r.correctness_pass < 0) {
        j << ind << "\"correctness_pass\": null," << nl;
    } else {
        kv_bool("correctness_pass", r.correctness_pass != 0);
    }
    kv_num("max_error", dbl_nullable(r.max_error));
    kv_num("mean", dbl_nullable(r.mean));
    kv_num("median", dbl_nullable(r.median));
    kv_num("min_val", dbl_nullable(r.min_val));
    kv_num("max_val", dbl_nullable(r.max_val));
    kv_num("standard_deviation", dbl_nullable(r.standard_deviation));

    // raw_measurements array
    j << ind << "\"raw_measurements\": [";
    for (size_t i = 0; i < r.raw_measurements.size(); ++i) {
        if (i > 0) j << ", ";
        j << dbl(r.raw_measurements[i]);
    }
    j << "]," << nl;

    kv_str("notes", r.notes);

    // error (last, no trailing comma)
    j << ind << "\"error\": ";
    if (r.error.empty()) j << "null";
    else j << "\"" << escape_json_string(r.error) << "\"";
    j << nl;

    j << "}";
    return j.str();
}

std::string results_to_json_array(const std::vector<BenchmarkResult>& results, bool pretty) {
    std::ostringstream j;
    j << "[";
    if (pretty) j << "\n";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) { j << ","; if (pretty) j << "\n"; }
        j << result_to_json(results[i], pretty);
    }
    if (pretty) j << "\n";
    j << "]";
    return j.str();
}
