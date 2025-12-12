#pragma once

#include "kan_layer.hpp"
#include <cmath>

class RBFKANLayer : public KANLayer {
public:
    RBFKANLayer(int n_in, int n_out, int grid_size, double epsilon = 1.0)
        : KANLayer(n_in, n_out, grid_size, KANBasis::RBF), epsilon_(epsilon) {
        // Initialize centers uniformly
        grid_min_ = -2.0;
        grid_max_ = 2.0;
        grid_spacing_ = (grid_max_ - grid_min_) / (grid_size - 1);
    }
    
protected:
    double evaluate_phi(double x, int i, int j) override {
        double result = 0.0;
        
        for (int k = 0; k < grid_size_; ++k) {
            double center = grid_min_ + k * grid_spacing_;
            double d = (x - center) / epsilon_;
            double rbf_val = std::exp(-d * d);
            result += params_[param_index(i, j, k)] * rbf_val;
        }
        
        return result;
    }
    
private:
    double epsilon_;
    double grid_min_, grid_max_, grid_spacing_;
};



