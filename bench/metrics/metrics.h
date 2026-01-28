#pragma once

#include <vector>

namespace bench::metrics {

double mean_squared_error(const std::vector<std::vector<double>>& predictions,
                           const std::vector<std::vector<double>>& targets);

double mean_absolute_error(const std::vector<std::vector<double>>& predictions,
                            const std::vector<std::vector<double>>& targets);

double cross_entropy(const std::vector<std::vector<double>>& logits,
                     const std::vector<std::vector<double>>& targets);

double accuracy(const std::vector<std::vector<double>>& logits,
                const std::vector<std::vector<double>>& targets);

std::vector<double> softmax(const std::vector<double>& logits);

}  // namespace bench::metrics
