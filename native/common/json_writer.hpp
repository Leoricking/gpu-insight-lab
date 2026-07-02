// GPU Insight Lab - JSON writer declarations
#pragma once
#include "../include/benchmark_result.hpp"
#include <vector>
#include <string>

std::string result_to_json(const BenchmarkResult& r, bool pretty = true);
std::string results_to_json_array(const std::vector<BenchmarkResult>& results, bool pretty = true);
