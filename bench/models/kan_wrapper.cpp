#include "bench/models/kan_wrapper.h"

#include "src/core/tensor.hpp"

#include <stdexcept>

namespace bench::models {

KANWrapper::KANWrapper(size_t input_dim,
                       size_t output_dim,
                       size_t grid_size)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      grid_size_(grid_size),
      layer_(static_cast<int>(input_dim), static_cast<int>(output_dim), static_cast<int>(grid_size)) {
    if (input_dim == 0 || output_dim == 0) {
        throw std::runtime_error("Invalid KAN dimensions");
    }
}

std::vector<double> KANWrapper::forward(const std::vector<double>& input) {
    Tensor input_tensor({input_dim_});
    for (size_t i = 0; i < input_dim_; ++i) {
        input_tensor[i] = input[i];
    }

    Tensor output = layer_.forward(input_tensor);
    std::vector<double> result(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        result[i] = output[i];
    }
    return result;
}

std::vector<std::vector<double>> KANWrapper::parameters() const {
    return {layer_.parameters()};
}

void KANWrapper::set_parameters(const std::vector<std::vector<double>>& params) {
    if (params.size() != 1) {
        throw std::runtime_error("KAN parameter size mismatch");
    }
    layer_.parameters() = params[0];
}

std::vector<std::vector<double>> KANWrapper::backward(const std::vector<double>& input,
                                                      const std::vector<double>& grad_output) {
    Tensor input_tensor({input_dim_});
    Tensor grad_tensor({output_dim_});
    for (size_t i = 0; i < input_dim_; ++i) {
        input_tensor[i] = input[i];
    }
    for (size_t i = 0; i < output_dim_; ++i) {
        grad_tensor[i] = grad_output[i];
    }

    auto grads = KANGradient::backward(grad_tensor, input_tensor, layer_);
    return {grads.parameter_grad};
}

size_t KANWrapper::parameter_count() const {
    return layer_.parameters().size();
}

}  // namespace bench::models
