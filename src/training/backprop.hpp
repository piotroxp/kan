#pragma once

#include "../core/tensor.hpp"
#include "../core/bspline_kan.hpp"
#include "../core/chebyshev_kan.hpp"
#include "../quantum/wavefunction.hpp"
#include <vector>
#include <memory>

// Forward declare SpeechModel to avoid circular dependency
class SpeechModel;

// Model output structure (matches SpeechModel::ModelOutput)
struct ModelOutput {
    Tensor audio_features;
    std::vector<Wavefunction> quantum_embeddings;
    Tensor semantic_representation;
    Tensor classification_logits;
};

// Backpropagation through the full model
class ModelBackward {
public:
    struct Gradients {
        std::vector<Tensor> layer_grads;           // Gradients for each layer
        std::vector<std::vector<double>> param_grads;  // Parameter gradients
    };
    
    // Backward pass through SpeechModel
    static Gradients backward(
        const ModelOutput& output,
        const Tensor& loss_grad,
        SpeechModel& model
    );
    
private:
    // Backward through classification head
    static Tensor backward_classification_head(
        const Tensor& logits,
        const Tensor& semantic_input,
        const Tensor& loss_grad,
        SpeechModel& model
    );
    
    // Backward through semantic layer
    static Tensor backward_semantic_layer(
        const Tensor& semantic_output,
        const Wavefunction& quantum_input,
        const Tensor& grad_output,
        SpeechModel& model
    );
    
    // Backward through quantum embedding
    static Tensor backward_quantum_embedding(
        const Wavefunction& quantum_output,
        const Tensor& grad_output,
        SpeechModel& model
    );
    
    // Compute parameter gradients
    static std::vector<std::vector<double>> compute_parameter_gradients(
        const ModelOutput& output,
        const Tensor& grad_semantic,
        const Tensor& grad_quantum,
        SpeechModel& model
    );
    
    // Helper: convert wavefunction to tensor
    static Tensor wavefunction_to_tensor(const Wavefunction& psi);
};
