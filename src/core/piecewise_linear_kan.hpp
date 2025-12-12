#pragma once

#include "kan_layer.hpp"
#include <algorithm>
#include <cmath>

class PiecewiseLinearKANLayer : public KANLayer {
public:
    PiecewiseLinearKANLayer(int n_in, int n_out, int grid_size)
        : KANLayer(n_in, n_out, grid_size, KANBasis::PiecewiseLinear) {
        grid_min_ = -2.0;
        grid_max_ = 2.0;
        grid_spacing_ = (grid_max_ - grid_min_) / (grid_size - 1);
    }
    
protected:
    double evaluate_phi(double x, int i, int j) override {
        // Clamp to grid range
        x = std::max(grid_min_, std::min(grid_max_, x));
        
        // Find the two grid points to interpolate between
        double normalized = (x - grid_min_) / grid_spacing_;
        int k = static_cast<int>(normalized);
        double t = normalized - k;
        
        // Clamp k to valid range
        k = std::max(0, std::min(grid_size_ - 2, k));
        
        // Linear interpolation: (1-t)*coeff[k] + t*coeff[k+1]
        double coeff_k = params_[param_index(i, j, k)];
        double coeff_k1 = params_[param_index(i, j, k + 1)];
        
        return (1.0 - t) * coeff_k + t * coeff_k1;
    }
    
private:
    double grid_min_, grid_max_, grid_spacing_;
};

