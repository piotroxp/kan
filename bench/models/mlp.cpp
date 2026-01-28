#include "bench/models/mlp.h"

#include <algorithm>
#include <cmath>

namespace bench::models {

namespace {

double silu(double x) {
    return x / (1.0 + std::exp(-x));
}

double silu_derivative(double x) {
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    return sigmoid + x * sigmoid * (1.0 - sigmoid);
}

}  // namespace

MLP::MLP(size_t input_dim,
         size_t output_dim,
         const std::vector<size_t>& hidden_layers,
         ActivationType activation,
         std::mt19937& rng)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      activation_(activation) {
    layer_dims_.clear();
    layer_dims_.push_back(input_dim);
    layer_dims_.insert(layer_dims_.end(), hidden_layers.begin(), hidden_layers.end());
    layer_dims_.push_back(output_dim);

    std::normal_distribution<double> init_dist(0.0, 0.1);

    size_t num_layers = layer_dims_.size() - 1;
    weights_.resize(num_layers);
    biases_.resize(num_layers);

    for (size_t layer = 0; layer < num_layers; ++layer) {
        size_t in_dim = layer_dims_[layer];
        size_t out_dim = layer_dims_[layer + 1];
        weights_[layer].resize(in_dim * out_dim);
        biases_[layer].resize(out_dim, 0.0);
        for (double& w : weights_[layer]) {
            w = init_dist(rng);
        }
    }
}

std::vector<double> MLP::forward(const std::vector<double>& input) {
    cache_.activations.clear();
    cache_.pre_activations.clear();
    cache_.activations.push_back(input);

    std::vector<double> current = input;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        size_t out_dim = layer_dims_[layer + 1];
        std::vector<double> pre = matvec(weights_[layer], current, out_dim);
        for (size_t j = 0; j < out_dim; ++j) {
            pre[j] += biases_[layer][j];
        }
        cache_.pre_activations.push_back(pre);

        std::vector<double> activated = pre;
        if (layer + 1 < weights_.size()) {
            for (double& val : activated) {
                val = activate(val);
            }
        }
        cache_.activations.push_back(activated);
        current = activated;
    }

    return current;
}

std::vector<std::vector<double>> MLP::parameters() const {
    std::vector<std::vector<double>> params;
    params.reserve(weights_.size() * 2);
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        params.push_back(weights_[layer]);
        params.push_back(biases_[layer]);
    }
    return params;
}

void MLP::set_parameters(const std::vector<std::vector<double>>& params) {
    size_t index = 0;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        weights_[layer] = params.at(index++);
        biases_[layer] = params.at(index++);
    }
}

std::vector<std::vector<double>> MLP::backward(const std::vector<double>&,
                                               const std::vector<double>& grad_output) {
    std::vector<std::vector<double>> grads;
    grads.reserve(weights_.size() * 2);

    std::vector<double> delta = grad_output;
    for (size_t layer_idx = weights_.size(); layer_idx-- > 0;) {
        const std::vector<double>& pre = cache_.pre_activations[layer_idx];
        const std::vector<double>& act_prev = cache_.activations[layer_idx];
        size_t in_dim = layer_dims_[layer_idx];
        size_t out_dim = layer_dims_[layer_idx + 1];

        if (layer_idx + 1 < weights_.size()) {
            for (size_t j = 0; j < out_dim; ++j) {
                delta[j] *= activate_derivative(pre[j]);
            }
        }

        std::vector<double> weight_grad(in_dim * out_dim, 0.0);
        std::vector<double> bias_grad(out_dim, 0.0);

        for (size_t j = 0; j < out_dim; ++j) {
            bias_grad[j] = delta[j];
            for (size_t i = 0; i < in_dim; ++i) {
                weight_grad[j * in_dim + i] = delta[j] * act_prev[i];
            }
        }

        grads.insert(grads.begin(), bias_grad);
        grads.insert(grads.begin(), weight_grad);

        if (layer_idx > 0) {
            std::vector<double> next_delta(in_dim, 0.0);
            for (size_t i = 0; i < in_dim; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < out_dim; ++j) {
                    sum += weights_[layer_idx][j * in_dim + i] * delta[j];
                }
                next_delta[i] = sum;
            }
            delta = next_delta;
        }
    }

    return grads;
}

size_t MLP::parameter_count() const {
    size_t count = 0;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        count += weights_[layer].size();
        count += biases_[layer].size();
    }
    return count;
}

double MLP::activate(double x) const {
    if (activation_ == ActivationType::SiLU) {
        return silu(x);
    }
    return std::max(0.0, x);
}

double MLP::activate_derivative(double x) const {
    if (activation_ == ActivationType::SiLU) {
        return silu_derivative(x);
    }
    return x > 0.0 ? 1.0 : 0.0;
}

std::vector<double> MLP::matvec(const std::vector<double>& weights,
                                const std::vector<double>& input,
                                size_t out_dim) const {
    size_t in_dim = input.size();
    std::vector<double> output(out_dim, 0.0);
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) {
            sum += weights[j * in_dim + i] * input[i];
        }
        output[j] = sum;
    }
    return output;
}

size_t estimate_mlp_hidden(size_t input_dim,
                           size_t output_dim,
                           size_t target_params) {
    if (input_dim == 0 || output_dim == 0) {
        return 1;
    }
    size_t best = 1;
    size_t best_diff = target_params;
    for (size_t hidden = 1; hidden < 2048; ++hidden) {
        size_t params = input_dim * hidden + hidden + hidden * output_dim + output_dim;
        size_t diff = params > target_params ? params - target_params : target_params - params;
        if (diff < best_diff) {
            best_diff = diff;
            best = hidden;
        }
        if (params > target_params && diff > best_diff) {
            break;
        }
    }
    return best;
}

}  // namespace bench::models
