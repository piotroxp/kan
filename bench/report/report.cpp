#include "bench/report/report.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace bench::report {

namespace {

void append_metric(std::ostringstream& oss, const char* name, double value, bool last) {
    oss << "\"" << name << "\":" << std::fixed << std::setprecision(6) << value;
    if (!last) {
        oss << ",";
    }
}

void append_eval(std::ostringstream& oss, const bench::train::EvalResult& eval) {
    oss << "{";
    append_metric(oss, "mse", eval.mse, false);
    append_metric(oss, "mae", eval.mae, false);
    append_metric(oss, "accuracy", eval.accuracy, false);
    append_metric(oss, "cross_entropy", eval.cross_entropy, true);
    oss << "}";
}

}  // namespace

std::string to_json(const ReportData& report) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"task\":\"" << report.task << "\",";
    oss << "\"model\":\"" << report.model << "\",";
    oss << "\"seed\":" << report.seed << ",";
    oss << "\"parameters\":" << report.parameters << ",";
    oss << "\"flops\":" << std::fixed << std::setprecision(2) << report.flops << ",";
    oss << "\"grid_size\":" << report.grid_size << ",";

    oss << "\"train\":";
    append_eval(oss, report.results.train_metrics);
    oss << ",\"val\":";
    append_eval(oss, report.results.val_metrics);
    oss << ",\"test\":";
    append_eval(oss, report.results.test_metrics);

    oss << ",\"robustness\":[";
    for (size_t i = 0; i < report.robustness.size(); ++i) {
        const auto& point = report.robustness[i];
        oss << "{\"epsilon\":" << point.epsilon << ",\"metric\":" << point.metric << "}";
        if (i + 1 < report.robustness.size()) {
            oss << ",";
        }
    }
    oss << "]}";
    return oss.str();
}

void write_report(const ReportData& report, const std::string& path) {
    std::ofstream out(path);
    out << to_json(report) << "\n";
}

}  // namespace bench::report
