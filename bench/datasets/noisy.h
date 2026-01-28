#pragma once

#include "bench/datasets/dataset.h"

#include <random>

namespace bench::datasets {

enum class NoiseType {
    Gaussian,
    StudentT
};

DatasetSplit make_noisy_regression_dataset(NoiseType type,
                                           double sigma,
                                           size_t total_samples,
                                           std::mt19937& rng);

DatasetSplit make_noisy_two_moons(double noise_std,
                                  double label_noise,
                                  size_t total_samples,
                                  std::mt19937& rng);

}  // namespace bench::datasets
