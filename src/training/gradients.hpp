#pragma once

#include "../core/kan_layer.hpp"
#include "../core/tensor.hpp"
#include <vector>

// Gradient computation for KAN layers
class KANGradient {
public:
    // Compute gradients for a KAN layer
    // Returns: gradients w.r.t. input and gradients w.r.t. parameters
    struct Gradients {
        Tensor input_grad;                    // Gradient w.r.t. input
        std::vector<double> parameter_grad;   // Gradient w.r.t. parameters
    };
    
    // Backward pass through KAN layer
    // grad_output: gradient from next layer
    // x: input to forward pass
    // layer: the KAN layer
    template<typename KANLayerType>
    static Gradients backward(
        const Tensor& grad_output,
        const Tensor& x,
        KANLayerType& layer
    ) {
        Gradients grads;
        
        int n_in = layer.n_in();
        int n_out = layer.n_out();
        int grid_size = layer.grid_size();
        
        // Initialize input gradient
        grads.input_grad = Tensor({static_cast<size_t>(n_in)});
        grads.input_grad.fill(0.0);
        
        // Initialize parameter gradient
        size_t num_params = n_out * n_in * grid_size;
        grads.parameter_grad.resize(num_params, 0.0);
        
        // Compute gradients
        for (int j = 0; j < n_out; ++j) {
            double grad_out_j = grad_output[j];
            
            for (int i = 0; i < n_in; ++i) {
                double x_i = x[i];
                
                // Gradient w.r.t. input: sum over outputs
                // ∂L/∂x_i = Σ_j (∂L/∂y_j) * (∂φ_{j,i}/∂x_i)
                // Approximate using finite differences
                double eps = 1e-6;
                double phi_plus = evaluate_phi_approx(layer, x_i + eps, i, j);
                double phi_minus = evaluate_phi_approx(layer, x_i - eps, i, j);
                double dphi_dx = (phi_plus - phi_minus) / (2.0 * eps);
                
                grads.input_grad[i] += grad_out_j * dphi_dx;
                
                // Gradient w.r.t. parameters
                // ∂L/∂c_{j,i,k} = (∂L/∂y_j) * (∂φ_{j,i}/∂c_{j,i,k})
                // For basis functions, this is the basis function value
                for (int k = 0; k < grid_size; ++k) {
                    size_t param_idx = layer.param_index(i, j, k);
                    double basis_val = evaluate_basis_derivative(layer, x_i, i, j, k);
                    grads.parameter_grad[param_idx] += grad_out_j * basis_val;
                }
            }
        }
        
        return grads;
    }
    
private:
    // Approximate phi evaluation (for gradient computation)
    template<typename KANLayerType>
    static double evaluate_phi_approx(KANLayerType& layer, double x, int i, int j) {
        // Create temporary input tensor
        Tensor temp_input({static_cast<size_t>(layer.n_in())});
        temp_input.fill(0.0);
        temp_input[i] = x;
        
        // Forward pass
        Tensor temp_output = layer.forward(temp_input);
        return temp_output[j];
    }
    
    // Evaluate basis function derivative (simplified)
    template<typename KANLayerType>
    static double evaluate_basis_derivative(KANLayerType& layer, double x, int i, int j, int k) {
        // This is a simplified version
        // In production, implement proper basis function derivatives
        // For now, return a simple approximation
        return 1.0 / layer.grid_size();
    }
};

// Automatic differentiation helper
class Autodiff {
public:
    // Compute gradients through the model
    static std::vector<std::vector<double>> compute_gradients(
        const Tensor& loss_grad,
        const std::vector<Tensor>& activations,
        const std::vector<std::vector<double>>& parameters
    ) {
        // Simplified gradient computation
        // In production, implement full backpropagation
        
        std::vector<std::vector<double>> gradients;
        for (const auto& param_vec : parameters) {
            gradients.push_back(std::vector<double>(param_vec.size(), 0.0));
        }
        
        // Simple gradient: proportional to loss gradient
        if (!activations.empty() && activations.back().size() > 0) {
            double loss_sum = 0.0;
            for (size_t i = 0; i < loss_grad.size(); ++i) {
                loss_sum += std::abs(loss_grad[i]);
            }
            
            // Distribute gradient to parameters
            for (auto& grad_vec : gradients) {
                for (size_t i = 0; i < grad_vec.size(); ++i) {
                    grad_vec[i] = loss_sum / grad_vec.size();
                }
            }
        }
        
        return gradients;
    }
};

