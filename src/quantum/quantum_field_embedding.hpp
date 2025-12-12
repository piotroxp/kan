#pragma once

#include "wavefunction.hpp"
#include "../core/chebyshev_kan.hpp"
#include "../core/tensor.hpp"
#include <vector>
#include <memory>

class QuantumFieldEmbeddingCore {
public:
    QuantumFieldEmbeddingCore(
        int input_dim,
        int embedding_dim,
        int grid_size = 1024,
        double L = 12.0,
        double sigma = 1.5
    ) : input_dim_(input_dim),
        embedding_dim_(embedding_dim),
        grid_size_(grid_size),
        L_(L),
        sigma_(sigma),
        alpha_extractor_(input_dim, 1, 10, 8),  // Chebyshev KAN: input_dim -> 1, grid=10, order=8
        beta_extractor_(input_dim, 1, 10, 8),
        gamma_extractor_(input_dim, 1, 10, 8) {
    }
    
    // Encode audio features to quantum embeddings
    std::vector<Wavefunction> encode(const Tensor& audio_features) {
        // Flatten audio features if needed
        Tensor flat_features = audio_features;
        if (audio_features.ndim() > 1) {
            size_t total_size = audio_features.size();
            flat_features = audio_features.reshape({total_size});
        }
        
        // Extract quantum parameters using Chebyshev KAN
        Tensor alpha_tensor = alpha_extractor_.forward(flat_features);
        Tensor beta_tensor = beta_extractor_.forward(flat_features);
        Tensor gamma_tensor = gamma_extractor_.forward(flat_features);
        
        // Extract scalar values (assuming single output)
        double alpha = alpha_tensor[0];
        double beta = beta_tensor[0];
        double gamma = gamma_tensor[0];
        
        // Create wavefunction
        Wavefunction psi(grid_size_, L_);
        psi.compute_squeezed_coherent(alpha, beta, gamma, sigma_);
        
        return {psi};
    }
    
    // Encode batch of features
    std::vector<std::vector<Wavefunction>> encode_batch(
        const std::vector<Tensor>& features_batch) {
        std::vector<std::vector<Wavefunction>> result;
        for (const auto& features : features_batch) {
            result.push_back(encode(features));
        }
        return result;
    }
    
    // Compute similarity between two embeddings (Born-rule fidelity)
    double similarity(const Wavefunction& psi1, const Wavefunction& psi2) const {
        return psi1.fidelity(psi2);
    }
    
    // Get parameters for training
    std::vector<double>& get_alpha_params() { 
        return alpha_extractor_.parameters(); 
    }
    std::vector<double>& get_beta_params() { 
        return beta_extractor_.parameters(); 
    }
    std::vector<double>& get_gamma_params() { 
        return gamma_extractor_.parameters(); 
    }
    
private:
    int input_dim_;
    int embedding_dim_;
    int grid_size_;
    double L_;
    double sigma_;
    
    // Chebyshev KAN layers for extracting quantum parameters
    ChebyshevKANLayer alpha_extractor_;
    ChebyshevKANLayer beta_extractor_;
    ChebyshevKANLayer gamma_extractor_;
};



