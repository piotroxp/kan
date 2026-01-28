#include "bench/metrics/metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace bench::metrics {

std::vector<double> softmax(const std::vector<double>& logits) {
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> exps(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_val);
        sum += exps[i];
    }
    for (double& val : exps) {
        val /= sum;
    }
    return exps;
}

double mean_squared_error(const std::vector<std::vector<double>>& predictions,
                           const std::vector<std::vector<double>>& targets) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            double diff = predictions[i][j] - targets[i][j];
            sum += diff * diff;
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double mean_absolute_error(const std::vector<std::vector<double>>& predictions,
                            const std::vector<std::vector<double>>& targets) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            sum += std::abs(predictions[i][j] - targets[i][j]);
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double cross_entropy(const std::vector<std::vector<double>>& logits,
                     const std::vector<std::vector<double>>& targets) {
    double loss = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        auto probs = softmax(logits[i]);
        for (size_t j = 0; j < probs.size(); ++j) {
            double target = targets[i][j];
            double prob = std::max(1e-12, std::min(1.0 - 1e-12, probs[j]));
            loss += -target * std::log(prob);
        }
    }
    return logits.empty() ? 0.0 : loss / static_cast<double>(logits.size());
}

double accuracy(const std::vector<std::vector<double>>& logits,
                const std::vector<std::vector<double>>& targets) {
    size_t correct = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        size_t pred = static_cast<size_t>(std::distance(logits[i].begin(),
                                                        std::max_element(logits[i].begin(), logits[i].end())));
        size_t target = static_cast<size_t>(std::distance(targets[i].begin(),
                                                          std::max_element(targets[i].begin(), targets[i].end())));
        if (pred == target) {
            ++correct;
        }
    }
    return logits.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(logits.size());
}

}  // namespace bench::metrics
