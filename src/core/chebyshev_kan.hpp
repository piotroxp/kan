#pragma once

#include "kan_layer.hpp"
#include <cmath>

class ChebyshevKANLayer : public KANLayer {
public:
    ChebyshevKANLayer(int n_in, int n_out, int grid_size, int chebyshev_order = 8)
        : KANLayer(n_in, n_out, grid_size, KANBasis::Chebyshev), 
          chebyshev_order_(chebyshev_order) {}
    
protected:
    double evaluate_phi(double x, int i, int j) override {
        // Map x to [-1, 1] via tanh
        double z = std::tanh(x);
        
        // Evaluate Chebyshev polynomial expansion
        double result = 0.0;
        
        // T_0(z) = 1, T_1(z) = z
        double Tkm1 = 1.0;
        double Tk = z;
        
        // c[0] * T_0 + c[1] * T_1
        if (chebyshev_order_ > 0) {
            result += params_[param_index(i, j, 0)] * Tkm1;
        }
        if (chebyshev_order_ > 1) {
            result += params_[param_index(i, j, 1)] * Tk;
        }
        
        // Recurrence: T_{k+1} = 2z*T_k - T_{k-1}
        for (int k = 2; k < chebyshev_order_ && k < grid_size_; ++k) {
            double Tkp1 = 2.0 * z * Tk - Tkm1;
            result += params_[param_index(i, j, k)] * Tkp1;
            Tkm1 = Tk;
            Tk = Tkp1;
        }
        
        return result;
    }
    
private:
    int chebyshev_order_;
};


