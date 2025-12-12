#pragma once

#include "kan_layer.hpp"
#include <cmath>

class BSplineKANLayer : public KANLayer {
public:
    BSplineKANLayer(int n_in, int n_out, int grid_size, int degree = 3)
        : KANLayer(n_in, n_out, grid_size, KANBasis::BSpline), degree_(degree) {
        // Initialize grid bounds
        grid_min_ = -2.0;
        grid_max_ = 2.0;
        grid_spacing_ = (grid_max_ - grid_min_) / (grid_size - 1);
    }
    
protected:
    double evaluate_phi(double x, int i, int j) override {
        // Clamp x to grid range
        x = std::max(grid_min_, std::min(grid_max_, x));
        
        // Find which B-spline basis functions are active
        double result = 0.0;
        for (int k = 0; k < grid_size_; ++k) {
            double grid_point = grid_min_ + k * grid_spacing_;
            double basis_val = evaluate_bspline(x, grid_point, degree_);
            double coeff = params_[param_index(i, j, k)];
            result += coeff * basis_val;
        }
        return result;
    }
    
private:
    int degree_;
    double grid_min_, grid_max_, grid_spacing_;
    
    // Evaluate cubic B-spline B3(t) where t = (x - grid_point) / grid_spacing
    double evaluate_bspline(double x, double grid_point, int deg) {
        double t = std::abs((x - grid_point) / grid_spacing_);
        
        if (deg == 3) {
            // Cubic B-spline
            if (t < 1.0) {
                return (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0;
            }
            if (t < 2.0) {
                double u = 2.0 - t;
                return (u * u * u) / 6.0;
            }
            return 0.0;
        } else if (deg == 1) {
            // Linear B-spline
            if (t < 1.0) {
                return 1.0 - t;
            }
            return 0.0;
        }
        
        // Default: linear
        if (t < 1.0) {
            return 1.0 - t;
        }
        return 0.0;
    }
};



