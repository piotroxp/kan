#include "bench/datasets/discontinuous.h"

#include <algorithm>
#include <stdexcept>

namespace bench::datasets {

namespace {

double evaluate_discontinuous(DiscontinuousType type, double x) {
    switch (type) {
        case DiscontinuousType::Step:
            return x > 0.0 ? 1.0 : 0.0;
        case DiscontinuousType::SignPlateau:
            if (x < -0.2) {
                return -1.0;
            }
            if (x <= 0.2) {
                return 0.0;
            }
            return 1.0;
        case DiscontinuousType::PiecewiseSlope:
            return x < 0.0 ? x + 1.0 : 2.0 * x - 1.0;
    }
    return 0.0;
}

}  // namespace

DatasetSplit make_discontinuous_dataset(DiscontinuousType type,
                                        size_t total_samples,
                                        std::mt19937& rng) {
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);

    Dataset dataset;
    dataset.input_dim = 1;
    dataset.output_dim = 1;
    dataset.classification = false;
    dataset.features.reserve(total_samples);
    dataset.targets.reserve(total_samples);

    for (size_t i = 0; i < total_samples; ++i) {
        double x = uniform(rng);
        double y = evaluate_discontinuous(type, x);
        dataset.features.push_back({x});
        dataset.targets.push_back({y});
    }

    return split_dataset(dataset, 0.7, 0.15, rng);
}

}  // namespace bench::datasets
