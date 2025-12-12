#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <functional>

enum class KANBasis {
    BSpline,
    Chebyshev,
    Sinc,
    Fourier,
    RBF,
    PiecewiseLinear
};

class KANLayer {
public:
    KANLayer(int n_in, int n_out, int grid_size, KANBasis basis)
        : n_in_(n_in), n_out_(n_out), grid_size_(grid_size), basis_(basis) {
        // Initialize parameters: each edge function has grid_size coefficients
        params_.resize(n_out * n_in * grid_size, 0.0);
    }
    
    virtual ~KANLayer() = default;
    
    // Forward pass
    virtual Tensor forward(const Tensor& x) {
        if (x.shape().size() != 1 || x.shape()[0] != n_in_) {
            throw std::runtime_error("Input tensor shape mismatch");
        }
        
        Tensor output({static_cast<size_t>(n_out_)});
        output.fill(0.0);
        
        for (int j = 0; j < n_out_; ++j) {
            for (int i = 0; i < n_in_; ++i) {
                double x_val = x[i];
                double phi_val = evaluate_phi(x_val, i, j);
                output[j] += phi_val;
            }
        }
        
        return output;
    }
    
    // Get parameters
    std::vector<double>& parameters() { return params_; }
    const std::vector<double>& parameters() const { return params_; }
    
    // Get basis type
    KANBasis basis_type() const { return basis_; }
    
    int n_in() const { return n_in_; }
    int n_out() const { return n_out_; }
    int grid_size() const { return grid_size_; }
    
protected:
    int n_in_;
    int n_out_;
    int grid_size_;
    KANBasis basis_;
    std::vector<double> params_;
    
    // Evaluate univariate function φ_{j,i}(x)
    virtual double evaluate_phi(double x, int i, int j) = 0;
    
    // Get parameter index for edge (i, j) at grid point k
    size_t param_index(int i, int j, int k) const {
        return (j * n_in_ + i) * grid_size_ + k;
    }
};

