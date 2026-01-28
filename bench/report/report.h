#pragma once

#include "bench/train/train_loop.h"

#include <cstddef>
#include <string>
#include <vector>

namespace bench::report {

struct RobustnessPoint {
    double epsilon = 0.0;
    double metric = 0.0;
};

struct ReportData {
    std::string task;
    std::string model;
    size_t seed = 0;
    size_t parameters = 0;
    double flops = 0.0;
    size_t grid_size = 0;
    bench::train::TrainResult results;
    std::vector<RobustnessPoint> robustness;
};

std::string to_json(const ReportData& report);
void write_report(const ReportData& report, const std::string& path);

}  // namespace bench::report
