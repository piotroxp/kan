#include "bench/datasets/categorical.h"
#include "bench/datasets/discontinuous.h"
#include "bench/datasets/noisy.h"
#include "bench/metrics/metrics.h"
#include "bench/models/embedding_kan.h"
#include "bench/models/embedding_mlp.h"
#include "bench/models/kan_wrapper.h"
#include "bench/models/mlp.h"
#include "bench/report/report.h"
#include "bench/train/train_loop.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace {

struct Options {
    std::string task = "discontinuous_step";
    std::string model = "kan";
    size_t seed = 123;
    size_t epochs = 200;
    size_t batch_size = 64;
    double learning_rate = 1e-3;
    size_t grid_size = 64;
    bool sweep = true;
    double sigma = 0.3;
    double label_noise = 0.1;
    double noise_std = 0.1;
    size_t samples = 4096;
    std::string report_path;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--task") {
            opts.task = next();
        } else if (arg == "--model") {
            opts.model = next();
        } else if (arg == "--seed") {
            opts.seed = static_cast<size_t>(std::stoul(next()));
        } else if (arg == "--epochs") {
            opts.epochs = static_cast<size_t>(std::stoul(next()));
        } else if (arg == "--batch") {
            opts.batch_size = static_cast<size_t>(std::stoul(next()));
        } else if (arg == "--lr") {
            opts.learning_rate = std::stod(next());
        } else if (arg == "--grid_size") {
            opts.grid_size = static_cast<size_t>(std::stoul(next()));
        } else if (arg == "--sweep") {
            opts.sweep = next() == "true";
        } else if (arg == "--sigma") {
            opts.sigma = std::stod(next());
        } else if (arg == "--label_noise") {
            opts.label_noise = std::stod(next());
        } else if (arg == "--noise_std") {
            opts.noise_std = std::stod(next());
        } else if (arg == "--samples") {
            opts.samples = static_cast<size_t>(std::stoul(next()));
        } else if (arg == "--report") {
            opts.report_path = next();
        }
    }
    return opts;
}

size_t estimate_flops(const bench::models::Model& model) {
    return model.parameter_count() * 2;
}

bench::train::TrainConfig make_train_config(const Options& opts, bool classification) {
    bench::train::TrainConfig config;
    config.epochs = opts.epochs;
    config.batch_size = opts.batch_size;
    config.learning_rate = opts.learning_rate;
    config.classification = classification;
    return config;
}

std::vector<bench::report::RobustnessPoint> random_perturbation_curve(
    bench::models::Model& model,
    const bench::datasets::Dataset& dataset,
    const std::vector<double>& epsilons,
    std::mt19937& rng) {
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<bench::report::RobustnessPoint> curve;
    for (double eps : epsilons) {
        bench::datasets::Dataset perturbed = dataset;
        for (auto& sample : perturbed.features) {
            for (double& val : sample) {
                val += eps * noise(rng);
            }
        }
        auto metrics = bench::train::evaluate_model(model, perturbed);
        curve.push_back({eps, dataset.classification ? metrics.accuracy : metrics.mse});
    }
    return curve;
}

std::vector<double> finite_difference_grad(bench::models::Model& model,
                                           const std::vector<double>& input,
                                           const std::vector<double>& target,
                                           bool classification) {
    const double delta = 1e-3;
    std::vector<double> grad(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        std::vector<double> plus = input;
        std::vector<double> minus = input;
        plus[i] += delta;
        minus[i] -= delta;
        auto pred_plus = model.forward(plus);
        auto pred_minus = model.forward(minus);
        double loss_plus = 0.0;
        double loss_minus = 0.0;
        if (classification) {
            auto probs_plus = bench::metrics::softmax(pred_plus);
            auto probs_minus = bench::metrics::softmax(pred_minus);
            for (size_t j = 0; j < target.size(); ++j) {
                loss_plus += -target[j] * std::log(std::max(1e-12, probs_plus[j]));
                loss_minus += -target[j] * std::log(std::max(1e-12, probs_minus[j]));
            }
        } else {
            for (size_t j = 0; j < target.size(); ++j) {
                double diff_plus = pred_plus[j] - target[j];
                double diff_minus = pred_minus[j] - target[j];
                loss_plus += diff_plus * diff_plus;
                loss_minus += diff_minus * diff_minus;
            }
        }
        grad[i] = (loss_plus - loss_minus) / (2.0 * delta);
    }
    return grad;
}

std::vector<bench::report::RobustnessPoint> fgsm_curve(
    bench::models::Model& model,
    const bench::datasets::Dataset& dataset,
    const std::vector<double>& epsilons) {
    std::vector<bench::report::RobustnessPoint> curve;
    for (double eps : epsilons) {
        bench::datasets::Dataset perturbed = dataset;
        for (size_t i = 0; i < dataset.features.size(); ++i) {
            auto grad = finite_difference_grad(model, dataset.features[i], dataset.targets[i], dataset.classification);
            for (size_t j = 0; j < grad.size(); ++j) {
                perturbed.features[i][j] += eps * (grad[j] >= 0.0 ? 1.0 : -1.0);
            }
        }
        auto metrics = bench::train::evaluate_model(model, perturbed);
        curve.push_back({eps, dataset.classification ? metrics.accuracy : metrics.mse});
    }
    return curve;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        std::mt19937 rng(opts.seed);

        bench::datasets::DatasetSplit dataset;
        bool is_classification = false;
        size_t input_dim = 1;
        size_t output_dim = 1;

        if (opts.task == "discontinuous_step") {
            dataset = bench::datasets::make_discontinuous_dataset(
                bench::datasets::DiscontinuousType::Step, opts.samples, rng);
        } else if (opts.task == "discontinuous_sign") {
            dataset = bench::datasets::make_discontinuous_dataset(
                bench::datasets::DiscontinuousType::SignPlateau, opts.samples, rng);
        } else if (opts.task == "discontinuous_piecewise") {
            dataset = bench::datasets::make_discontinuous_dataset(
                bench::datasets::DiscontinuousType::PiecewiseSlope, opts.samples, rng);
        } else if (opts.task == "noisy_regression_gaussian") {
            dataset = bench::datasets::make_noisy_regression_dataset(
                bench::datasets::NoiseType::Gaussian, opts.sigma, opts.samples, rng);
        } else if (opts.task == "noisy_regression_student") {
            dataset = bench::datasets::make_noisy_regression_dataset(
                bench::datasets::NoiseType::StudentT, opts.sigma, opts.samples, rng);
        } else if (opts.task == "noisy_moons") {
            dataset = bench::datasets::make_noisy_two_moons(opts.noise_std, opts.label_noise, opts.samples, rng);
        } else if (opts.task == "adversarial_moons") {
            dataset = bench::datasets::make_noisy_two_moons(opts.noise_std, 0.0, opts.samples, rng);
        } else if (opts.task == "adversarial_regression") {
            dataset = bench::datasets::make_discontinuous_dataset(
                bench::datasets::DiscontinuousType::PiecewiseSlope, opts.samples, rng);
        } else if (opts.task == "categorical_bag") {
            dataset = bench::datasets::make_bag_of_categories(50000, 10, opts.samples, rng);
        } else if (opts.task == "categorical_sequence") {
            dataset = bench::datasets::make_sequence_pattern(10000, 32, opts.samples, rng);
        } else {
            throw std::runtime_error("Unknown task: " + opts.task);
        }

        is_classification = dataset.train.classification;
        input_dim = dataset.train.input_dim;
        output_dim = dataset.train.output_dim;

        std::vector<size_t> grid_sizes = {opts.grid_size};
        if (opts.sweep && opts.task.rfind("discontinuous", 0) == 0 && opts.model == "kan") {
            grid_sizes = {16, 32, 64, 128};
        }

        std::vector<bench::report::ReportData> reports;
        for (size_t grid_size : grid_sizes) {
            std::unique_ptr<bench::models::Model> model;
            if (opts.task.rfind("categorical", 0) == 0) {
                size_t embedding_dim = 32;
                if (opts.model == "kan") {
                    model = std::make_unique<bench::models::EmbeddingKAN>(
                        opts.task == "categorical_bag" ? 50000 : 10000,
                        embedding_dim,
                        output_dim,
                        grid_size,
                        rng);
                } else {
                    size_t target_params = embedding_dim * (opts.task == "categorical_bag" ? 50000 : 10000)
                        + embedding_dim * output_dim * grid_size;
                    size_t hidden = bench::models::estimate_mlp_hidden(embedding_dim, output_dim, target_params);
                    model = std::make_unique<bench::models::EmbeddingMLP>(
                        opts.task == "categorical_bag" ? 50000 : 10000,
                        embedding_dim,
                        output_dim,
                        std::vector<size_t>{hidden},
                        bench::models::PoolingType::Mean,
                        rng);
                }
            } else {
                if (opts.model == "kan") {
                    model = std::make_unique<bench::models::KANWrapper>(input_dim, output_dim, grid_size);
                } else {
                    size_t target_params = input_dim * output_dim * grid_size;
                    size_t hidden = bench::models::estimate_mlp_hidden(input_dim, output_dim, target_params);
                    model = std::make_unique<bench::models::MLP>(
                        input_dim,
                        output_dim,
                        std::vector<size_t>{hidden},
                        bench::models::ActivationType::Relu,
                        rng);
                }
            }

            auto config = make_train_config(opts, is_classification);
            auto train_result = bench::train::train_model(*model, dataset, config, rng);

            bench::report::ReportData report;
            report.task = opts.task;
            report.model = model->name();
            report.seed = opts.seed;
            report.parameters = model->parameter_count();
            report.flops = static_cast<double>(estimate_flops(*model));
            report.grid_size = grid_size;
            report.results = train_result;

            if (opts.task == "noisy_regression_gaussian" || opts.task == "noisy_regression_student") {
                std::vector<double> sigmas{0.1, 0.3, 0.5, 1.0};
                report.robustness.clear();
                for (double sigma : sigmas) {
                    std::mt19937 eval_rng(opts.seed + static_cast<size_t>(sigma * 100));
                    auto eval_split = bench::datasets::make_noisy_regression_dataset(
                        opts.task == "noisy_regression_gaussian" ? bench::datasets::NoiseType::Gaussian
                                                                : bench::datasets::NoiseType::StudentT,
                        sigma,
                        opts.samples,
                        eval_rng);
                    auto metrics = bench::train::evaluate_model(*model, eval_split.test);
                    report.robustness.push_back({sigma, metrics.mse});
                }
            } else if (opts.task == "noisy_moons") {
                std::vector<double> label_rates{0.0, 0.1, 0.2, 0.4};
                report.robustness.clear();
                for (double rate : label_rates) {
                    std::mt19937 eval_rng(opts.seed + static_cast<size_t>(rate * 100));
                    auto eval_split = bench::datasets::make_noisy_two_moons(
                        opts.noise_std, rate, opts.samples, eval_rng);
                    auto metrics = bench::train::evaluate_model(*model, eval_split.test);
                    report.robustness.push_back({rate, metrics.accuracy});
                }
            }

            if (opts.task == "adversarial_moons") {
                std::vector<double> epsilons{0.0, 0.05, 0.1, 0.2, 0.3};
                report.task = opts.task + "_random";
                report.robustness = random_perturbation_curve(*model, dataset.test, epsilons, rng);
                reports.push_back(report);

                bench::report::ReportData fgsm_report = report;
                fgsm_report.task = opts.task + "_fgsm";
                fgsm_report.robustness = fgsm_curve(*model, dataset.test, epsilons);
                reports.push_back(fgsm_report);
            } else if (opts.task == "adversarial_regression") {
                std::vector<double> epsilons{0.0, 0.02, 0.05, 0.1, 0.2};
                report.robustness.clear();
                for (double eps : epsilons) {
                    double total = 0.0;
                    for (const auto& sample : dataset.test.features) {
                        double x = sample[0];
                        auto pred_center = model->forward(sample);
                        double target_center = x < 0.0 ? x + 1.0 : 2.0 * x - 1.0;
                        double worst = std::abs(pred_center[0] - target_center);
                        for (double delta : {-eps, eps}) {
                            double xp = x + delta;
                            double target = xp < 0.0 ? xp + 1.0 : 2.0 * xp - 1.0;
                            auto pred = model->forward({xp});
                            worst = std::max(worst, std::abs(pred[0] - target));
                        }
                        total += worst;
                    }
                    double mean_worst = total / static_cast<double>(dataset.test.features.size());
                    report.robustness.push_back({eps, mean_worst});
                }
                reports.push_back(report);
            } else {
                reports.push_back(report);
            }

            std::cout << "Task " << opts.task << " model " << report.model
                      << " grid " << grid_size
                      << " test metric "
                      << (is_classification ? report.results.test_metrics.accuracy
                                             : report.results.test_metrics.mse)
                      << "\n";
        }

        if (!opts.report_path.empty()) {
            if (reports.size() == 1) {
                bench::report::write_report(reports.front(), opts.report_path);
            } else {
                std::ostringstream oss;
                oss << "[";
                for (size_t i = 0; i < reports.size(); ++i) {
                    oss << bench::report::to_json(reports[i]);
                    if (i + 1 < reports.size()) {
                        oss << ",";
                    }
                }
                oss << "]\n";
                std::ofstream out(opts.report_path);
                out << oss.str();
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
