#pragma once

#include "bench/datasets/dataset.h"

#include <random>

namespace bench::datasets {

enum class DiscontinuousType {
    Step,
    SignPlateau,
    PiecewiseSlope
};

DatasetSplit make_discontinuous_dataset(DiscontinuousType type,
                                        size_t total_samples,
                                        std::mt19937& rng);

}  // namespace bench::datasets
