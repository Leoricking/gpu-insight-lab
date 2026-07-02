// GPU Insight Lab - Statistics utilities
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

struct Statistics {
    double mean     = 0.0;
    double median   = 0.0;
    double min_val  = 0.0;
    double max_val  = 0.0;
    double std_dev  = 0.0;
    std::vector<double> raw;
};

inline Statistics compute_statistics(const std::vector<double>& data) {
    Statistics s;
    if (data.empty()) return s;

    s.raw = data;
    size_t n = data.size();

    // Mean
    double sum = 0.0;
    for (double v : data) sum += v;
    s.mean = sum / n;

    // Variance (population)
    double var_sum = 0.0;
    for (double v : data) var_sum += (v - s.mean) * (v - s.mean);
    s.std_dev = std::sqrt(var_sum / n);

    // Sorted copy for min/max/median
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    s.min_val = sorted_data.front();
    s.max_val = sorted_data.back();

    if (n % 2 == 1) {
        s.median = sorted_data[n / 2];
    } else {
        s.median = (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0;
    }

    return s;
}
