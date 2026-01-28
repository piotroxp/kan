#pragma once

#include <string>
#include <vector>

namespace bench::models {

class Model {
public:
    virtual ~Model() = default;

    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<std::vector<double>> parameters() const = 0;
    virtual void set_parameters(const std::vector<std::vector<double>>& params) = 0;
    virtual std::vector<std::vector<double>> backward(const std::vector<double>& input,
                                                      const std::vector<double>& grad_output) = 0;
    virtual size_t parameter_count() const = 0;
    virtual size_t input_dim() const = 0;
    virtual size_t output_dim() const = 0;
    virtual std::string name() const = 0;
};

}  // namespace bench::models
