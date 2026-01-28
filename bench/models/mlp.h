#pragma once

#include "bench/models/model.h"

#include <random>

namespace bench::models {

enum class ActivationType {
    Relu,
    SiLU
};

class MLP : public Model {
public:
    MLP(size_t input_dim,
        size_t output_dim,
        const std::vector<size_t>& hidden_layers,
        ActivationType activation,
        std::mt19937& rng);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<std::vector<double>> parameters() const override;
    void set_parameters(const std::vector<std::vector<double>>& params) override;
    std::vector<std::vector<double>> backward(const std::vector<double>& input,
                                              const std::vector<double>& grad_output) override;
    size_t parameter_count() const override;
    size_t input_dim() const override { return input_dim_; }
    size_t output_dim() const override { return output_dim_; }
    std::string name() const override { return "MLP"; }

private:
    struct Cache {
        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> pre_activations;
    };

    size_t input_dim_;
    size_t output_dim_;
    ActivationType activation_;
    std::vector<size_t> layer_dims_;
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> biases_;
    Cache cache_;

    double activate(double x) const;
    double activate_derivative(double x) const;
    std::vector<double> matvec(const std::vector<double>& weights,
                               const std::vector<double>& input,
                               size_t out_dim) const;
};

size_t estimate_mlp_hidden(size_t input_dim,
                           size_t output_dim,
                           size_t target_params);

}  // namespace bench::models
