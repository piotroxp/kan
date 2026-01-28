#pragma once

#include <vector>
#include <random>
#include <cstddef>

namespace bench::datasets {

struct Dataset {
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> targets;
    size_t input_dim = 0;
    size_t output_dim = 0;
    bool classification = false;
};

struct DatasetSplit {
    Dataset train;
    Dataset val;
    Dataset test;
};

DatasetSplit split_dataset(const Dataset& dataset,
                           double train_ratio,
                           double val_ratio,
                           std::mt19937& rng);

}  // namespace bench::datasets
