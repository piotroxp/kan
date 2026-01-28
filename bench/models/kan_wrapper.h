#pragma once

#include "bench/models/model.h"
#include "src/core/bspline_kan.hpp"
#include "src/training/gradients.hpp"

#include <memory>

namespace bench::models {

class KANWrapper : public Model {
public:
    KANWrapper(size_t input_dim,
               size_t output_dim,
               size_t grid_size);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<std::vector<double>> parameters() const override;
    void set_parameters(const std::vector<std::vector<double>>& params) override;
    std::vector<std::vector<double>> backward(const std::vector<double>& input,
                                              const std::vector<double>& grad_output) override;
    size_t parameter_count() const override;
    size_t input_dim() const override { return input_dim_; }
    size_t output_dim() const override { return output_dim_; }
    std::string name() const override { return "KAN"; }

    size_t grid_size() const { return grid_size_; }

private:
    size_t input_dim_;
    size_t output_dim_;
    size_t grid_size_;
    BSplineKANLayer layer_;
};

}  // namespace bench::models
