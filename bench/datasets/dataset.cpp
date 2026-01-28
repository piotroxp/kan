#include "bench/datasets/dataset.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace bench::datasets {

DatasetSplit split_dataset(const Dataset& dataset,
                           double train_ratio,
                           double val_ratio,
                           std::mt19937& rng) {
    if (dataset.features.size() != dataset.targets.size()) {
        throw std::runtime_error("Feature/target size mismatch");
    }
    if (train_ratio <= 0.0 || val_ratio <= 0.0 || train_ratio + val_ratio >= 1.0) {
        throw std::runtime_error("Invalid split ratios");
    }

    std::vector<size_t> indices(dataset.features.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t total = indices.size();
    size_t train_count = static_cast<size_t>(total * train_ratio);
    size_t val_count = static_cast<size_t>(total * val_ratio);

    DatasetSplit split;
    split.train.input_dim = dataset.input_dim;
    split.train.output_dim = dataset.output_dim;
    split.train.classification = dataset.classification;
    split.val = split.train;
    split.test = split.train;

    auto assign_sample = [&](Dataset& target, size_t idx) {
        target.features.push_back(dataset.features[idx]);
        target.targets.push_back(dataset.targets[idx]);
    };

    for (size_t i = 0; i < total; ++i) {
        size_t idx = indices[i];
        if (i < train_count) {
            assign_sample(split.train, idx);
        } else if (i < train_count + val_count) {
            assign_sample(split.val, idx);
        } else {
            assign_sample(split.test, idx);
        }
    }

    return split;
}

}  // namespace bench::datasets
