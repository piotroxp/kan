#pragma once

#include <complex>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Wavefunction {
public:
    Wavefunction(int N, double L) 
        : N_(N), L_(L), dx_(2.0 * L / N) {
        psi_.resize(N);
        normalization_ = 1.0;
    }
    
    // Compute squeezed coherent state wavefunction
    // ψ(x) = N exp[-(x-α)²/(4σ²) + iβx + iγ]
    void compute_squeezed_coherent(
        double alpha,  // displacement
        double beta,   // chirp/frequency
        double gamma,  // phase
        double sigma   // squeezing parameter
    ) {
        // Normalization constant: N = (2πσ²)^(-1/4)
        normalization_ = std::pow(2.0 * M_PI * sigma * sigma, -0.25);
        
        for (int j = 0; j < N_; ++j) {
            double x = grid_point(j);
            
            // Real part: exp[-(x-α)²/(4σ²)] * cos(βx + γ)
            // Imaginary part: exp[-(x-α)²/(4σ²)] * sin(βx + γ)
            double arg = -(x - alpha) * (x - alpha) / (4.0 * sigma * sigma);
            double phase = beta * x + gamma;
            
            double real_part = normalization_ * std::exp(arg) * std::cos(phase);
            double imag_part = normalization_ * std::exp(arg) * std::sin(phase);
            
            psi_[j] = std::complex<double>(real_part, imag_part);
        }
        
        // Normalize
        normalize();
    }
    
    // Normalize wavefunction: |ψ|₂² = Δx Σ|ψ[j]|² = 1
    void normalize() {
        double norm_sq = 0.0;
        for (const auto& val : psi_) {
            norm_sq += dx_ * (val.real() * val.real() + val.imag() * val.imag());
        }
        
        if (norm_sq > 1e-10) {
            double norm = std::sqrt(norm_sq);
            for (auto& val : psi_) {
                val /= norm;
            }
        }
    }
    
    // Compute inner product (overlap) with another wavefunction
    std::complex<double> inner_product(const Wavefunction& other) const {
        if (N_ != other.N_ || std::abs(dx_ - other.dx_) > 1e-10) {
            throw std::runtime_error("Wavefunction dimensions mismatch");
        }
        
        std::complex<double> result(0.0, 0.0);
        for (int j = 0; j < N_; ++j) {
            result += dx_ * std::conj(psi_[j]) * other.psi_[j];
        }
        return result;
    }
    
    // Compute Born-rule fidelity: |⟨ψ₁|ψ₂⟩|²
    double fidelity(const Wavefunction& other) const {
        std::complex<double> overlap = inner_product(other);
        double real_part = overlap.real();
        double imag_part = overlap.imag();
        return real_part * real_part + imag_part * imag_part;
    }
    
    // Access wavefunction values
    const std::vector<std::complex<double>>& values() const { return psi_; }
    std::vector<std::complex<double>>& values() { return psi_; }
    
    // Grid properties
    int size() const { return N_; }
    double grid_spacing() const { return dx_; }
    double grid_point(int j) const { 
        return -L_ + j * dx_; 
    }
    double grid_range() const { return L_; }
    
private:
    int N_;              // Grid size
    double L_;           // Grid range [-L, L]
    double dx_;          // Grid spacing: 2L/N
    double normalization_; // (2πσ²)^(-1/4)
    std::vector<std::complex<double>> psi_; // Wavefunction values
};


