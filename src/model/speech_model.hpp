#pragma once

#include "../audio/feature_extraction.hpp"
#include "../quantum/quantum_field_embedding.hpp"
#include "../core/bspline_kan.hpp"
#include "../core/tensor.hpp"
#include "../quantum/wavefunction.hpp"
#include <vector>
#include <memory>

// Unified speech model integrating all components
class SpeechModel {
public:
    struct ModelOutput {
        Tensor audio_features;              // Mel-spectrogram
        std::vector<Wavefunction> quantum_embeddings;  // Quantum embeddings
        Tensor semantic_representation;     // Semantic features
        Tensor classification_logits;        // FSD50K class predictions
    };
    
    SpeechModel(
        int n_mels = 80,
        int embedding_dim = 256,
        int num_classes = 200,
        int quantum_grid_size = 1024
    ) : feature_extractor_(n_mels, 2048, 512, 2048, 0.0, 22050.0, 44100),
        quantum_embedding_(n_mels, embedding_dim, quantum_grid_size, 12.0, 1.5),
        semantic_layer_(embedding_dim, embedding_dim, 5, 3),  // B-spline KAN
        classification_head_(embedding_dim, num_classes, 5, 3),  // B-spline KAN
        num_classes_(num_classes) {
    }
    
    // Forward pass: Audio -> Features -> Quantum -> Semantic -> Classification
    ModelOutput forward(const AudioBuffer& audio) {
        ModelOutput output;
        
        // Step 1: Extract mel-spectrogram features
        output.audio_features = feature_extractor_.process(audio);
        
        // Step 2: Pool over time (mean pooling)
        Tensor pooled_features = pool_temporal(output.audio_features);
        
        // Step 3: Encode to quantum embeddings
        output.quantum_embeddings = quantum_embedding_.encode(pooled_features);
        
        // Step 4: Extract semantic representation from quantum embedding
        // Convert wavefunction to real vector (use real/imag parts or magnitude)
        Tensor quantum_vector = wavefunction_to_tensor(output.quantum_embeddings[0]);
        output.semantic_representation = semantic_layer_.forward(quantum_vector);
        
        // Step 5: Classification
        output.classification_logits = classification_head_.forward(output.semantic_representation);
        
        return output;
    }
    
    // Batch forward pass
    std::vector<ModelOutput> forward_batch(const std::vector<AudioBuffer>& audio_batch) {
        std::vector<ModelOutput> outputs;
        for (const auto& audio : audio_batch) {
            outputs.push_back(forward(audio));
        }
        return outputs;
    }
    
    // Get model parameters for training
    std::vector<std::vector<double>> get_parameters() {
        std::vector<std::vector<double>> params;
        
        // Semantic layer parameters
        params.push_back(semantic_layer_.parameters());
        
        // Classification head parameters
        params.push_back(classification_head_.parameters());
        
        // Quantum embedding parameters
        params.push_back(quantum_embedding_.get_alpha_params());
        params.push_back(quantum_embedding_.get_beta_params());
        params.push_back(quantum_embedding_.get_gamma_params());
        
        return params;
    }
    
    // Set parameters (for loading checkpoints)
    void set_parameters(const std::vector<std::vector<double>>& params) {
        // In production, implement proper parameter loading
        // For now, placeholder
    }
    
    int num_classes() const { return num_classes_; }
    
private:
    SincKANMelSpectrogram feature_extractor_;
    QuantumFieldEmbeddingCore quantum_embedding_;
    BSplineKANLayer semantic_layer_;
    BSplineKANLayer classification_head_;
    int num_classes_;
    
    // Pool temporal dimension (mean pooling)
    Tensor pool_temporal(const Tensor& mel_spec) {
        size_t num_frames = mel_spec.shape()[0];
        size_t num_mels = mel_spec.shape()[1];
        
        Tensor pooled({num_mels});
        pooled.fill(0.0);
        
        for (size_t t = 0; t < num_frames; ++t) {
            for (size_t m = 0; m < num_mels; ++m) {
                pooled[m] += mel_spec.at({t, m});
            }
        }
        
        for (size_t m = 0; m < num_mels; ++m) {
            pooled[m] /= num_frames;
        }
        
        return pooled;
    }
    
    // Convert wavefunction to tensor (use magnitude, downsample for semantic layer)
    Tensor wavefunction_to_tensor(const Wavefunction& psi) {
        // Downsample from 1024 to embedding_dim for semantic layer input
        int target_size = semantic_layer_.n_in();
        Tensor result({static_cast<size_t>(target_size)});
        
        int step = psi.size() / target_size;
        for (int i = 0; i < target_size; ++i) {
            int idx = i * step;
            if (idx < psi.size()) {
                const auto& val = psi.values()[idx];
                double magnitude = std::sqrt(val.real() * val.real() + val.imag() * val.imag());
                result[i] = magnitude;
            } else {
                result[i] = 0.0;
            }
        }
        
        return result;
    }
};

