#pragma once

#include "kan_layer.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class FourierKANLayer : public KANLayer {
public:
    FourierKANLayer(int n_in, int n_out, int grid_size)
        : KANLayer(n_in, n_out, grid_size, KANBasis::Fourier) {
        // grid_size should be odd: N modes on each side + DC
        N_ = (grid_size - 1) / 2;
        h_ = 2.0 * M_PI / grid_size;
    }
    
protected:
    double evaluate_phi(double x, int i, int j) override {
        double result = 0.0;
        
        // DC component
        result += params_[param_index(i, j, 0)];
        
        // Cosine and sine terms
        for (int k = 1; k <= N_ && (2*k-1) < grid_size_ && (2*k) < grid_size_; ++k) {
            double cos_val = std::cos(k * h_ * x);
            double sin_val = std::sin(k * h_ * x);
            result += params_[param_index(i, j, 2*k-1)] * cos_val;
            result += params_[param_index(i, j, 2*k)] * sin_val;
        }
        
        return result;
    }
    
private:
    int N_;
    double h_;
};



