#pragma once

#include "bench/models/model.h"

#include <random>

namespace bench::models {

enum class PoolingType {
    Mean,
    Sum
};

class EmbeddingMLP : public Model {
public:
    EmbeddingMLP(size_t vocab_size,
                 size_t embedding_dim,
                 size_t output_dim,
                 const std::vector<size_t>& hidden_layers,
                 PoolingType pooling,
                 std::mt19937& rng);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<std::vector<double>> parameters() const override;
    void set_parameters(const std::vector<std::vector<double>>& params) override;
    std::vector<std::vector<double>> backward(const std::vector<double>& input,
                                              const std::vector<double>& grad_output) override;
    size_t parameter_count() const override;
    size_t input_dim() const override { return input_dim_; }
    size_t output_dim() const override { return output_dim_; }
    std::string name() const override { return "EmbeddingMLP"; }

private:
    size_t vocab_size_;
    size_t embedding_dim_;
    size_t output_dim_;
    size_t input_dim_;
    PoolingType pooling_;
    std::vector<double> embeddings_;

    std::vector<size_t> mlp_layers_;
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> biases_;

    std::vector<double> pooled_cache_;
    std::vector<std::vector<double>> activations_cache_;
    std::vector<std::vector<double>> pre_activations_cache_;

    size_t embedding_index(size_t token, size_t dim) const;
    std::vector<double> embed_and_pool(const std::vector<double>& input);
};

}  // namespace bench::models
