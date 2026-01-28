#pragma once

#include "bench/models/kan_wrapper.h"

#include <random>

namespace bench::models {

class EmbeddingKAN : public Model {
public:
    EmbeddingKAN(size_t vocab_size,
                 size_t embedding_dim,
                 size_t output_dim,
                 size_t grid_size,
                 std::mt19937& rng);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<std::vector<double>> parameters() const override;
    void set_parameters(const std::vector<std::vector<double>>& params) override;
    std::vector<std::vector<double>> backward(const std::vector<double>& input,
                                              const std::vector<double>& grad_output) override;
    size_t parameter_count() const override;
    size_t input_dim() const override { return embedding_dim_; }
    size_t output_dim() const override { return output_dim_; }
    std::string name() const override { return "EmbeddingKAN"; }

private:
    size_t vocab_size_;
    size_t embedding_dim_;
    size_t output_dim_;
    std::vector<double> embeddings_;
    std::vector<double> pooled_cache_;
    KANWrapper kan_;

    size_t embedding_index(size_t token, size_t dim) const;
    std::vector<double> embed_and_pool(const std::vector<double>& input);
};

}  // namespace bench::models
