#include "backprop.hpp"
#include "../model/speech_model.hpp"

// Backward pass through SpeechModel
ModelBackward::Gradients ModelBackward::backward(
    const ModelOutput& output,
    const Tensor& loss_grad,
    SpeechModel& model
) {
    Gradients grads;
    
    // Start from classification head
    Tensor grad_semantic = backward_classification_head(
        output.classification_logits,
        output.semantic_representation,
        loss_grad,
        model
    );
    
    // Backward through semantic layer
    Tensor grad_quantum = backward_semantic_layer(
        output.semantic_representation,
        output.quantum_embeddings[0],
        grad_semantic,
        model
    );
    
    // Backward through quantum embedding (simplified)
    Tensor grad_features = backward_quantum_embedding(
        output.quantum_embeddings[0],
        grad_quantum,
        model
    );
    
    // Store gradients
    grads.layer_grads.push_back(grad_features);
    grads.layer_grads.push_back(grad_quantum);
    grads.layer_grads.push_back(grad_semantic);
    
    // Get parameter gradients
    grads.param_grads = compute_parameter_gradients(
        output, grad_semantic, grad_quantum, model
    );
    
    return grads;
}

// Backward through classification head
Tensor ModelBackward::backward_classification_head(
    const Tensor& logits,
    const Tensor& semantic_input,
    const Tensor& loss_grad,
    SpeechModel& model
) {
    // Gradient w.r.t. semantic representation
    Tensor grad_semantic({semantic_input.size()});
    grad_semantic.fill(0.0);
    
    // Simplified: assume linear relationship
    for (size_t i = 0; i < semantic_input.size(); ++i) {
        double grad_sum = 0.0;
        for (size_t j = 0; j < logits.size(); ++j) {
            // Approximate gradient using finite differences
            double eps = 1e-6;
            Tensor input_plus = semantic_input;
            Tensor input_minus = semantic_input;
            input_plus[i] += eps;
            input_minus[i] -= eps;
            
            // Simplified gradient approximation
            grad_sum += loss_grad[j] * (input_plus[i] - input_minus[i]) / (2.0 * eps);
        }
        grad_semantic[i] = grad_sum;
    }
    
    return grad_semantic;
}

// Backward through semantic layer
Tensor ModelBackward::backward_semantic_layer(
    const Tensor& semantic_output,
    const Wavefunction& quantum_input,
    const Tensor& grad_output,
    SpeechModel& model
) {
    // Convert wavefunction to tensor for gradient computation
    Tensor quantum_tensor = wavefunction_to_tensor(quantum_input);
    
    // Gradient w.r.t. quantum embedding
    Tensor grad_quantum({quantum_tensor.size()});
    grad_quantum.fill(0.0);
    
    // Simplified gradient computation
    for (size_t i = 0; i < quantum_tensor.size(); ++i) {
        double grad_sum = 0.0;
        for (size_t j = 0; j < semantic_output.size(); ++j) {
            // Approximate using finite differences
            double eps = 1e-6;
            Tensor input_plus = quantum_tensor;
            Tensor input_minus = quantum_tensor;
            input_plus[i] += eps;
            input_minus[i] -= eps;
            
            // Simplified gradient
            grad_sum += grad_output[j] * (input_plus[i] - input_minus[i]) / (2.0 * eps);
        }
        grad_quantum[i] = grad_sum;
    }
    
    return grad_quantum;
}

// Backward through quantum embedding
Tensor ModelBackward::backward_quantum_embedding(
    const Wavefunction& quantum_output,
    const Tensor& grad_output,
    SpeechModel& model
) {
    // Gradient w.r.t. input features
    Tensor grad_features({80});  // Assuming 80 mel bins
    grad_features.fill(0.0);
    
    // Simplified: distribute gradient evenly
    double grad_sum = 0.0;
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_sum += grad_output[i];
    }
    
    for (size_t i = 0; i < grad_features.size(); ++i) {
        grad_features[i] = grad_sum / grad_features.size();
    }
    
    return grad_features;
}

// Compute parameter gradients
std::vector<std::vector<double>> ModelBackward::compute_parameter_gradients(
    const ModelOutput& output,
    const Tensor& grad_semantic,
    const Tensor& grad_quantum,
    SpeechModel& model
) {
    std::vector<std::vector<double>> param_grads;
    
    // Get model parameters
    auto params = model.get_parameters();
    
    // Compute gradients for each parameter group
    for (size_t group_idx = 0; group_idx < params.size(); ++group_idx) {
        std::vector<double> grad_group(params[group_idx].size(), 0.0);
        
        // Simplified gradient: proportional to output gradient
        double grad_magnitude = 0.0;
        if (group_idx == 0) {  // Semantic layer
            for (size_t i = 0; i < grad_semantic.size(); ++i) {
                grad_magnitude += std::abs(grad_semantic[i]);
            }
        } else if (group_idx == 1) {  // Classification head
            grad_magnitude = 1.0;  // Placeholder
        } else {  // Quantum embedding parameters
            for (size_t i = 0; i < grad_quantum.size(); ++i) {
                grad_magnitude += std::abs(grad_quantum[i]);
            }
        }
        
        // Distribute gradient to parameters
        for (size_t i = 0; i < grad_group.size(); ++i) {
            grad_group[i] = grad_magnitude / grad_group.size() * 1e-4;
        }
        
        param_grads.push_back(grad_group);
    }
    
    return param_grads;
}

// Helper: convert wavefunction to tensor
Tensor ModelBackward::wavefunction_to_tensor(const Wavefunction& psi) {
    Tensor result({static_cast<size_t>(psi.size())});
    for (int i = 0; i < psi.size(); ++i) {
        const auto& val = psi.values()[i];
        double magnitude = std::sqrt(val.real() * val.real() + val.imag() * val.imag());
        result[i] = magnitude;
    }
    return result;
}
