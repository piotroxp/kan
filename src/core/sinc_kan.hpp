#pragma once

#include "kan_layer.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SincKANLayer : public KANLayer {
public:
    SincKANLayer(int n_in, int n_out, int grid_size, double h = M_PI / 5.0)
        : KANLayer(n_in, n_out, grid_size, KANBasis::Sinc), h_(h) {
        // Grid centered around 0
        int N = (grid_size - 1) / 2;
        grid_center_ = 0.0;
    }
    
protected:
    double evaluate_phi(double x, int i, int j) override {
        int N = (grid_size_ - 1) / 2;
        double result = 0.0;
        
        for (int k = -N; k <= N && (k + N) < grid_size_; ++k) {
            double sinc_val = sinc(M_PI * (x - k * h_) / h_);
            size_t param_idx = param_index(i, j, k + N);
            if (param_idx < params_.size()) {
                result += params_[param_idx] * sinc_val;
            }
        }
        
        return result;
    }
    
private:
    double h_;
    double grid_center_;
    
    static double sinc(double x) {
        if (std::abs(x) < 1e-8) {
            return 1.0;
        }
        return std::sin(x) / x;
    }
};

