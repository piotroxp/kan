#include "bench/datasets/discontinuous.h"
#include "bench/datasets/noisy.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

using Catch::Approx;

TEST_CASE("Discontinuous dataset splits and targets") {
    std::mt19937 rng(42);
    auto split = bench::datasets::make_discontinuous_dataset(
        bench::datasets::DiscontinuousType::Step, 1000, rng);

    REQUIRE(split.train.features.size() + split.val.features.size() + split.test.features.size() == 1000);
    REQUIRE(split.train.features.size() > split.val.features.size());
    REQUIRE(split.val.features.size() == split.test.features.size());

    for (size_t i = 0; i < split.train.features.size(); ++i) {
        double x = split.train.features[i][0];
        double y = split.train.targets[i][0];
        double expected = x > 0.0 ? 1.0 : 0.0;
        REQUIRE(y == Approx(expected));
    }
}

TEST_CASE("Noisy regression noise variance") {
    std::mt19937 rng(123);
    double sigma = 0.5;
    auto split = bench::datasets::make_noisy_regression_dataset(
        bench::datasets::NoiseType::Gaussian, sigma, 2000, rng);

    double sum = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;
    for (const auto& sample : split.train.features) {
        double x = sample[0];
        double base = std::sin(3.0 * x) + 0.3 * std::cos(9.0 * x);
        double noise = split.train.targets[count][0] - base;
        sum += noise;
        sum_sq += noise * noise;
        ++count;
    }
    double mean = sum / static_cast<double>(count);
    double var = sum_sq / static_cast<double>(count) - mean * mean;
    REQUIRE(std::abs(mean) < 0.05);
    REQUIRE(var == Approx(sigma * sigma).margin(0.05));
}

TEST_CASE("Two moons classification labels are one-hot") {
    std::mt19937 rng(7);
    auto split = bench::datasets::make_noisy_two_moons(0.1, 0.0, 200, rng);
    for (const auto& target : split.train.targets) {
        REQUIRE(target.size() == 2);
        double sum = target[0] + target[1];
        REQUIRE(sum == Approx(1.0));
    }
}
