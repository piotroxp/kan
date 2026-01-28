#include "bench/datasets/noisy.h"

#include <cmath>
#include <algorithm>

namespace bench::datasets {

namespace {

double base_function(double x) {
    return std::sin(3.0 * x) + 0.3 * std::cos(9.0 * x);
}

}  // namespace

DatasetSplit make_noisy_regression_dataset(NoiseType type,
                                           double sigma,
                                           size_t total_samples,
                                           std::mt19937& rng) {
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::normal_distribution<double> gaussian(0.0, sigma);
    std::student_t_distribution<double> student_t(3.0);

    Dataset dataset;
    dataset.input_dim = 1;
    dataset.output_dim = 1;
    dataset.classification = false;
    dataset.features.reserve(total_samples);
    dataset.targets.reserve(total_samples);

    for (size_t i = 0; i < total_samples; ++i) {
        double x = uniform(rng);
        double noise = 0.0;
        if (type == NoiseType::Gaussian) {
            noise = gaussian(rng);
        } else {
            double raw = student_t(rng);
            noise = raw * sigma / std::sqrt(3.0);
        }
        double y = base_function(x) + noise;
        dataset.features.push_back({x});
        dataset.targets.push_back({y});
    }

    return split_dataset(dataset, 0.7, 0.15, rng);
}

DatasetSplit make_noisy_two_moons(double noise_std,
                                  double label_noise,
                                  size_t total_samples,
                                  std::mt19937& rng) {
    constexpr double kPi = 3.14159265358979323846;
    std::uniform_real_distribution<double> angle_dist(0.0, kPi);
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    std::uniform_real_distribution<double> flip_dist(0.0, 1.0);

    Dataset dataset;
    dataset.input_dim = 2;
    dataset.output_dim = 2;
    dataset.classification = true;
    dataset.features.reserve(total_samples);
    dataset.targets.reserve(total_samples);

    size_t half = total_samples / 2;
    for (size_t i = 0; i < total_samples; ++i) {
        bool upper = i < half;
        double angle = angle_dist(rng);
        double x = std::cos(angle);
        double y = std::sin(angle);
        if (!upper) {
            x += 1.0;
            y = -y - 0.2;
        }
        x += noise_dist(rng);
        y += noise_dist(rng);

        int label = upper ? 0 : 1;
        if (flip_dist(rng) < label_noise) {
            label = 1 - label;
        }

        dataset.features.push_back({x, y});
        dataset.targets.push_back(label == 0 ? std::vector<double>{1.0, 0.0}
                                            : std::vector<double>{0.0, 1.0});
    }

    return split_dataset(dataset, 0.7, 0.15, rng);
}

}  // namespace bench::datasets
