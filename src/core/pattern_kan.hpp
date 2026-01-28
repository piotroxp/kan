#pragma once

#include "kan_layer.hpp"
#include <algorithm>
#include <cmath>

class PatternKANLayer : public KANLayer {
public:
    PatternKANLayer(int n_in,
                    int n_out,
                    int grid_size,
                    double frequency = 3.0,
                    double phase = 1.0,
                    double amplitude = 1.0)
        : KANLayer(n_in, n_out, grid_size, KANBasis::Pattern),
          frequency_(frequency),
          phase_(phase),
          amplitude_(amplitude) {
        grid_min_ = -2.0;
        grid_max_ = 2.0;
        grid_spacing_ = (grid_max_ - grid_min_) / (grid_size - 1);
    }

protected:
    double evaluate_phi(double x, int i, int j) override {
        x = std::max(grid_min_, std::min(grid_max_, x));

        double result = 0.0;
        for (int k = 0; k < grid_size_; ++k) {
            double grid_point = grid_min_ + k * grid_spacing_;
            double basis_val = evaluate_pattern(x, grid_point);
            double coeff = params_[param_index(i, j, k)];
            result += coeff * basis_val;
        }
        return result;
    }

private:
    double frequency_;
    double phase_;
    double amplitude_;
    double grid_min_;
    double grid_max_;
    double grid_spacing_;

    double evaluate_pattern(double x, double grid_point) const {
        double offset = (x - grid_point) / grid_spacing_;
        constexpr double kPi = 3.14159265358979323846;
        double arg = offset / kPi;
        double base = 0.5 + std::sin(frequency_ * arg);
        base = std::max(0.0, base);
        double pattern = std::pow(base, amplitude_);
        double phase_term = std::exp(phase_ * arg);
        double gate = 1.0 - std::exp(-phase_ * arg);
        return pattern * phase_term * gate;
    }
};
