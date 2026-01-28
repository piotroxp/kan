#pragma once

#include "bench/datasets/dataset.h"
#include "bench/models/model.h"

#include <random>

namespace bench::train {

struct TrainConfig {
    size_t epochs = 200;
    size_t batch_size = 64;
    double learning_rate = 1e-3;
    bool classification = false;
};

struct EvalResult {
    double mse = 0.0;
    double mae = 0.0;
    double accuracy = 0.0;
    double cross_entropy = 0.0;
};

struct TrainResult {
    EvalResult train_metrics;
    EvalResult val_metrics;
    EvalResult test_metrics;
};

TrainResult train_model(bench::models::Model& model,
                        const bench::datasets::DatasetSplit& dataset,
                        const TrainConfig& config,
                        std::mt19937& rng);

EvalResult evaluate_model(bench::models::Model& model,
                          const bench::datasets::Dataset& dataset);

}  // namespace bench::train
