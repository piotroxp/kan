#include "bench/models/embedding_mlp.h"

#include <algorithm>
#include <cmath>

namespace bench::models {

EmbeddingMLP::EmbeddingMLP(size_t vocab_size,
                           size_t embedding_dim,
                           size_t output_dim,
                           const std::vector<size_t>& hidden_layers,
                           PoolingType pooling,
                           std::mt19937& rng)
    : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      output_dim_(output_dim),
      input_dim_(embedding_dim),
      pooling_(pooling) {
    std::normal_distribution<double> init_dist(0.0, 0.1);
    embeddings_.resize(vocab_size_ * embedding_dim_);
    for (double& val : embeddings_) {
        val = init_dist(rng);
    }

    mlp_layers_.clear();
    mlp_layers_.push_back(embedding_dim_);
    mlp_layers_.insert(mlp_layers_.end(), hidden_layers.begin(), hidden_layers.end());
    mlp_layers_.push_back(output_dim_);

    size_t num_layers = mlp_layers_.size() - 1;
    weights_.resize(num_layers);
    biases_.resize(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        size_t in_dim = mlp_layers_[layer];
        size_t out_dim = mlp_layers_[layer + 1];
        weights_[layer].resize(in_dim * out_dim);
        biases_[layer].resize(out_dim, 0.0);
        for (double& w : weights_[layer]) {
            w = init_dist(rng);
        }
    }
}

std::vector<double> EmbeddingMLP::forward(const std::vector<double>& input) {
    pooled_cache_ = embed_and_pool(input);
    activations_cache_.clear();
    pre_activations_cache_.clear();
    activations_cache_.push_back(pooled_cache_);

    std::vector<double> current = pooled_cache_;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        size_t out_dim = mlp_layers_[layer + 1];
        std::vector<double> pre(out_dim, 0.0);
        size_t in_dim = mlp_layers_[layer];
        for (size_t j = 0; j < out_dim; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < in_dim; ++i) {
                sum += weights_[layer][j * in_dim + i] * current[i];
            }
            pre[j] = sum + biases_[layer][j];
        }
        pre_activations_cache_.push_back(pre);
        std::vector<double> activated = pre;
        if (layer + 1 < weights_.size()) {
            for (double& val : activated) {
                val = std::max(0.0, val);
            }
        }
        activations_cache_.push_back(activated);
        current = activated;
    }

    return current;
}

std::vector<std::vector<double>> EmbeddingMLP::parameters() const {
    std::vector<std::vector<double>> params;
    params.reserve(weights_.size() * 2 + 1);
    params.push_back(embeddings_);
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        params.push_back(weights_[layer]);
        params.push_back(biases_[layer]);
    }
    return params;
}

void EmbeddingMLP::set_parameters(const std::vector<std::vector<double>>& params) {
    size_t index = 0;
    embeddings_ = params.at(index++);
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        weights_[layer] = params.at(index++);
        biases_[layer] = params.at(index++);
    }
}

std::vector<std::vector<double>> EmbeddingMLP::backward(const std::vector<double>& input,
                                                        const std::vector<double>& grad_output) {
    std::vector<std::vector<double>> grads;
    grads.reserve(weights_.size() * 2 + 1);

    std::vector<std::vector<double>> weight_grads;
    std::vector<std::vector<double>> bias_grads;
    weight_grads.resize(weights_.size());
    bias_grads.resize(biases_.size());

    std::vector<double> delta = grad_output;
    std::vector<double> pooled_grad(embedding_dim_, 0.0);
    for (size_t layer_idx = weights_.size(); layer_idx-- > 0;) {
        size_t in_dim = mlp_layers_[layer_idx];
        size_t out_dim = mlp_layers_[layer_idx + 1];
        const std::vector<double>& pre = pre_activations_cache_[layer_idx];
        const std::vector<double>& prev_act = activations_cache_[layer_idx];

        if (layer_idx + 1 < weights_.size()) {
            for (size_t j = 0; j < out_dim; ++j) {
                delta[j] *= pre[j] > 0.0 ? 1.0 : 0.0;
            }
        }

        weight_grads[layer_idx].assign(in_dim * out_dim, 0.0);
        bias_grads[layer_idx].assign(out_dim, 0.0);
        for (size_t j = 0; j < out_dim; ++j) {
            bias_grads[layer_idx][j] = delta[j];
            for (size_t i = 0; i < in_dim; ++i) {
                weight_grads[layer_idx][j * in_dim + i] = delta[j] * prev_act[i];
            }
        }

        std::vector<double> next_delta(in_dim, 0.0);
        for (size_t i = 0; i < in_dim; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < out_dim; ++j) {
                sum += weights_[layer_idx][j * in_dim + i] * delta[j];
            }
            next_delta[i] = sum;
        }

        if (layer_idx == 0) {
            pooled_grad = next_delta;
        } else {
            delta = next_delta;
        }
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

    if (pooling_ == PoolingType::Mean && !tokens.empty()) {
        for (double& val : pooled_grad) {
            val /= static_cast<double>(tokens.size());
        }
    }

    for (size_t token : tokens) {
        for (size_t dim = 0; dim < embedding_dim_; ++dim) {
            embedding_grad[embedding_index(token, dim)] += pooled_grad[dim];
        }
    }

    grads.push_back(embedding_grad);
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        grads.push_back(weight_grads[layer]);
        grads.push_back(bias_grads[layer]);
    }

    return grads;
}

size_t EmbeddingMLP::parameter_count() const {
    size_t count = embeddings_.size();
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        count += weights_[layer].size();
        count += biases_[layer].size();
    }
    return count;
}

size_t EmbeddingMLP::embedding_index(size_t token, size_t dim) const {
    return token * embedding_dim_ + dim;
}

std::vector<double> EmbeddingMLP::embed_and_pool(const std::vector<double>& input) {
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

    if (pooling_ == PoolingType::Mean) {
        for (double& val : pooled) {
            val /= static_cast<double>(tokens.size());
        }
    }

    return pooled;
}

}  // namespace bench::models
