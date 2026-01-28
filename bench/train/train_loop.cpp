#include "bench/train/train_loop.h"

#include "bench/metrics/metrics.h"
#include "src/training/optimizer.hpp"

#include <algorithm>
#include <numeric>

namespace bench::train {

namespace {

std::vector<double> compute_grad_output(const std::vector<double>& prediction,
                                        const std::vector<double>& target,
                                        bool classification) {
    if (classification) {
        auto probs = bench::metrics::softmax(prediction);
        std::vector<double> grad(prediction.size(), 0.0);
        for (size_t i = 0; i < prediction.size(); ++i) {
            grad[i] = probs[i] - target[i];
        }
        return grad;
    }

    std::vector<double> grad(prediction.size(), 0.0);
    for (size_t i = 0; i < prediction.size(); ++i) {
        grad[i] = 2.0 * (prediction[i] - target[i]) / static_cast<double>(prediction.size());
    }
    return grad;
}

}  // namespace

TrainResult train_model(bench::models::Model& model,
                        const bench::datasets::DatasetSplit& dataset,
                        const TrainConfig& config,
                        std::mt19937& rng) {
    AdamW optimizer(config.learning_rate);
    auto params = model.parameters();
    optimizer.initialize(params);

    std::vector<size_t> indices(dataset.train.features.size());
    std::iota(indices.begin(), indices.end(), 0U);

    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t start = 0; start < indices.size(); start += config.batch_size) {
            size_t end = std::min(indices.size(), start + config.batch_size);
            std::vector<std::vector<double>> grads(params.size());
            for (size_t g = 0; g < grads.size(); ++g) {
                grads[g].assign(params[g].size(), 0.0);
            }

            for (size_t idx = start; idx < end; ++idx) {
                size_t sample_idx = indices[idx];
                const auto& x = dataset.train.features[sample_idx];
                const auto& y = dataset.train.targets[sample_idx];
                auto prediction = model.forward(x);
                auto grad_output = compute_grad_output(prediction, y, dataset.train.classification);
                auto sample_grads = model.backward(x, grad_output);
                for (size_t g = 0; g < grads.size(); ++g) {
                    for (size_t j = 0; j < grads[g].size(); ++j) {
                        grads[g][j] += sample_grads[g][j];
                    }
                }
            }

            double scale = 1.0 / static_cast<double>(end - start);
            for (size_t g = 0; g < grads.size(); ++g) {
                for (double& val : grads[g]) {
                    val *= scale;
                }
            }

            optimizer.step(params, grads);
            model.set_parameters(params);
        }
    }

    TrainResult result;
    result.train_metrics = evaluate_model(model, dataset.train);
    result.val_metrics = evaluate_model(model, dataset.val);
    result.test_metrics = evaluate_model(model, dataset.test);
    return result;
}

EvalResult evaluate_model(bench::models::Model& model,
                          const bench::datasets::Dataset& dataset) {
    std::vector<std::vector<double>> predictions;
    predictions.reserve(dataset.features.size());
    for (const auto& sample : dataset.features) {
        predictions.push_back(model.forward(sample));
    }

    EvalResult result;
    if (dataset.classification) {
        result.cross_entropy = bench::metrics::cross_entropy(predictions, dataset.targets);
        result.accuracy = bench::metrics::accuracy(predictions, dataset.targets);
    } else {
        result.mse = bench::metrics::mean_squared_error(predictions, dataset.targets);
        result.mae = bench::metrics::mean_absolute_error(predictions, dataset.targets);
    }

    return result;
}

}  // namespace bench::train
