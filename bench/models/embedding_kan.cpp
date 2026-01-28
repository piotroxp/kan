#include "bench/models/embedding_kan.h"

#include <cmath>

namespace bench::models {

EmbeddingKAN::EmbeddingKAN(size_t vocab_size,
                           size_t embedding_dim,
                           size_t output_dim,
                           size_t grid_size,
                           std::mt19937& rng)
    : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      output_dim_(output_dim),
      embeddings_(vocab_size * embedding_dim),
      kan_(embedding_dim, output_dim, grid_size) {
    std::normal_distribution<double> init_dist(0.0, 0.1);
    for (double& val : embeddings_) {
        val = init_dist(rng);
    }
}

std::vector<double> EmbeddingKAN::forward(const std::vector<double>& input) {
    pooled_cache_ = embed_and_pool(input);
    return kan_.forward(pooled_cache_);
}

std::vector<std::vector<double>> EmbeddingKAN::parameters() const {
    std::vector<std::vector<double>> params;
    params.push_back(embeddings_);
    auto kan_params = kan_.parameters();
    params.insert(params.end(), kan_params.begin(), kan_params.end());
    return params;
}

void EmbeddingKAN::set_parameters(const std::vector<std::vector<double>>& params) {
    embeddings_ = params.at(0);
    std::vector<std::vector<double>> kan_params(params.begin() + 1, params.end());
    kan_.set_parameters(kan_params);
}

std::vector<std::vector<double>> EmbeddingKAN::backward(const std::vector<double>& input,
                                                        const std::vector<double>& grad_output) {
    std::vector<std::vector<double>> grads;
    std::vector<double> pooled = embed_and_pool(input);
    auto kan_grads = kan_.backward(pooled, grad_output);

    const double delta = 1e-3;
    std::vector<double> input_grad(embedding_dim_, 0.0);
    for (size_t dim = 0; dim < embedding_dim_; ++dim) {
        std::vector<double> plus = pooled;
        std::vector<double> minus = pooled;
        plus[dim] += delta;
        minus[dim] -= delta;
        auto out_plus = kan_.forward(plus);
        auto out_minus = kan_.forward(minus);
        double dot = 0.0;
        for (size_t j = 0; j < grad_output.size(); ++j) {
            dot += (out_plus[j] - out_minus[j]) * grad_output[j];
        }
        input_grad[dim] = dot / (2.0 * delta);
    }

    std::vector<double> embedding_grad(embeddings_.size(), 0.0);
    std::vector<size_t> tokens;
    tokens.reserve(input.size());
    for (double val : input) {
        size_t token = static_cast<size_t>(val);
        if (token < vocab_size_) {
            tokens.push_back(token);
        }
    }
    if (!tokens.empty()) {
        for (size_t token : tokens) {
            for (size_t dim = 0; dim < embedding_dim_; ++dim) {
                embedding_grad[embedding_index(token, dim)] += input_grad[dim] /
                    static_cast<double>(tokens.size());
            }
        }
    }

    grads.push_back(embedding_grad);
    grads.insert(grads.end(), kan_grads.begin(), kan_grads.end());
    return grads;
}

size_t EmbeddingKAN::parameter_count() const {
    size_t count = embeddings_.size();
    auto kan_params = kan_.parameters();
    for (const auto& param : kan_params) {
        count += param.size();
    }
    return count;
}

size_t EmbeddingKAN::embedding_index(size_t token, size_t dim) const {
    return token * embedding_dim_ + dim;
}

std::vector<double> EmbeddingKAN::embed_and_pool(const std::vector<double>& input) {
    std::vector<size_t> tokens;
    tokens.reserve(input.size());
    for (double val : input) {
        size_t token = static_cast<size_t>(val);
        if (token < vocab_size_) {
            tokens.push_back(token);
        }
    }

    std::vector<double> pooled(embedding_dim_, 0.0);
    if (tokens.empty()) {
        return pooled;
    }

    for (size_t token : tokens) {
        for (size_t dim = 0; dim < embedding_dim_; ++dim) {
            pooled[dim] += embeddings_[embedding_index(token, dim)];
        }
    }

    for (double& val : pooled) {
        val /= static_cast<double>(tokens.size());
    }

    return pooled;
}

}  // namespace bench::models
