#pragma once

#include "../core/tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// AdamW optimizer
class AdamW {
public:
    AdamW(
        double learning_rate = 1e-4,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8,
        double weight_decay = 1e-5
    ) : lr_(learning_rate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        step_(0) {}
    
    // Initialize optimizer state for parameters
    void initialize(const std::vector<std::vector<double>>& parameters) {
        m_.clear();
        v_.clear();
        
        for (const auto& param_vec : parameters) {
            m_.push_back(std::vector<double>(param_vec.size(), 0.0));
            v_.push_back(std::vector<double>(param_vec.size(), 0.0));
        }
    }
    
    // Update parameters using gradients
    void step(
        std::vector<std::vector<double>>& parameters,
        const std::vector<std::vector<double>>& gradients
    ) {
        if (m_.empty()) {
            initialize(parameters);
        }
        
        step_++;
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_);
        
        for (size_t i = 0; i < parameters.size(); ++i) {
            for (size_t j = 0; j < parameters[i].size(); ++j) {
                // Update biased first moment estimate
                m_[i][j] = beta1_ * m_[i][j] + (1.0 - beta1_) * gradients[i][j];
                
                // Update biased second raw moment estimate
                v_[i][j] = beta2_ * v_[i][j] + (1.0 - beta2_) * gradients[i][j] * gradients[i][j];
                
                // Compute bias-corrected estimates
                double m_hat = m_[i][j] / bias_correction1;
                double v_hat = v_[i][j] / bias_correction2;
                
                // Update parameter with weight decay
                parameters[i][j] -= lr_ * (m_hat / (std::sqrt(v_hat) + epsilon_) + weight_decay_ * parameters[i][j]);
            }
        }
    }
    
    // Set learning rate
    void set_learning_rate(double lr) {
        lr_ = lr;
    }
    
    double learning_rate() const { return lr_; }
    size_t step() const { return step_; }
    
    // Reset optimizer state
    void reset() {
        m_.clear();
        v_.clear();
        step_ = 0;
    }
    
private:
    double lr_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    size_t step_;
    
    // First and second moment estimates
    std::vector<std::vector<double>> m_;
    std::vector<std::vector<double>> v_;
};

// Learning rate scheduler (cosine annealing)
class CosineAnnealingScheduler {
public:
    CosineAnnealingScheduler(
        double initial_lr,
        double min_lr,
        size_t max_steps
    ) : initial_lr_(initial_lr),
        min_lr_(min_lr),
        max_steps_(max_steps) {}
    
    double get_lr(size_t step) const {
        if (step >= max_steps_) {
            return min_lr_;
        }
        
        double cosine = std::cos(M_PI * step / max_steps_);
        return min_lr_ + (initial_lr_ - min_lr_) * (1.0 + cosine) / 2.0;
    }
    
private:
    double initial_lr_;
    double min_lr_;
    size_t max_steps_;
};
