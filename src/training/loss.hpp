#pragma once

#include "../core/tensor.hpp"
#include "../quantum/wavefunction.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Multi-task loss function for speech model training
class MultiTaskLoss {
public:
    MultiTaskLoss(
        double audio_weight = 1.0,
        double quantum_weight = 1.0,
        double classification_weight = 1.0,
        double l2_weight = 1e-5,
        double normalization_weight = 0.1
    ) : audio_weight_(audio_weight),
        quantum_weight_(quantum_weight),
        classification_weight_(classification_weight),
        l2_weight_(l2_weight),
        normalization_weight_(normalization_weight) {}
    
    // Audio reconstruction loss (MSE)
    double audio_reconstruction_loss(const Tensor& pred, const Tensor& target) {
        if (pred.size() != target.size()) {
            throw std::runtime_error("Prediction and target size mismatch");
        }
        
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < pred.size(); ++i) {
            double diff = pred[i] - target[i];
            sum_squared_error += diff * diff;
        }
        
        return sum_squared_error / pred.size();
    }
    
    // Quantum embedding loss using Born-rule fidelity
    double quantum_fidelity_loss(
        const std::vector<Wavefunction>& embeddings,
        const std::vector<std::pair<size_t, size_t>>& positive_pairs,
        const std::vector<std::pair<size_t, size_t>>& negative_pairs,
        double margin = 0.5
    ) {
        double loss = 0.0;
        
        // Positive pairs: maximize fidelity (minimize 1 - fidelity)
        for (const auto& pair : positive_pairs) {
            double fidelity = embeddings[pair.first].fidelity(embeddings[pair.second]);
            loss += (1.0 - fidelity) * (1.0 - fidelity);  // Squared loss
        }
        
        // Negative pairs: minimize fidelity (maximize margin - fidelity)
        for (const auto& pair : negative_pairs) {
            double fidelity = embeddings[pair.first].fidelity(embeddings[pair.second]);
            if (fidelity > margin) {
                loss += (fidelity - margin) * (fidelity - margin);
            }
        }
        
        size_t total_pairs = positive_pairs.size() + negative_pairs.size();
        return total_pairs > 0 ? loss / total_pairs : 0.0;
    }
    
    // Classification loss (multi-label binary cross-entropy)
    double classification_loss(const Tensor& logits, const Tensor& labels) {
        if (logits.size() != labels.size()) {
            throw std::runtime_error("Logits and labels size mismatch");
        }
        
        double loss = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            double logit = logits[i];
            double label = labels[i];
            
            // Binary cross-entropy: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
            // Using log-sum-exp trick for numerical stability
            double sigmoid = 1.0 / (1.0 + std::exp(-logit));
            sigmoid = std::max(1e-15, std::min(1.0 - 1e-15, sigmoid));  // Clamp
            
            loss -= label * std::log(sigmoid) + (1.0 - label) * std::log(1.0 - sigmoid);
        }
        
        return loss / logits.size();
    }
    
    // Wavefunction normalization penalty
    double normalization_penalty(const std::vector<Wavefunction>& embeddings) {
        double penalty = 0.0;
        
        for (const auto& psi : embeddings) {
            double norm_sq = 0.0;
            for (const auto& val : psi.values()) {
                double mag_sq = val.real() * val.real() + val.imag() * val.imag();
                norm_sq += psi.grid_spacing() * mag_sq;
            }
            
            // Penalty for deviation from 1.0
            double deviation = norm_sq - 1.0;
            penalty += deviation * deviation;
        }
        
        return embeddings.empty() ? 0.0 : penalty / embeddings.size();
    }
    
    // L2 regularization on parameters
    double l2_regularization(const std::vector<std::vector<double>>& parameters) {
        double sum_sq = 0.0;
        size_t count = 0;
        
        for (const auto& param_vec : parameters) {
            for (double p : param_vec) {
                sum_sq += p * p;
                count++;
            }
        }
        
        return count > 0 ? sum_sq / count : 0.0;
    }
    
    // Combined loss
    struct LossComponents {
        double total = 0.0;
        double audio = 0.0;
        double quantum = 0.0;
        double classification = 0.0;
        double l2 = 0.0;
        double normalization = 0.0;
    };
    
    LossComponents compute(
        const Tensor& audio_pred,
        const Tensor& audio_target,
        const std::vector<Wavefunction>& quantum_embeddings,
        const std::vector<std::pair<size_t, size_t>>& positive_pairs,
        const std::vector<std::pair<size_t, size_t>>& negative_pairs,
        const Tensor& classification_logits,
        const Tensor& classification_labels,
        const std::vector<std::vector<double>>& parameters
    ) {
        LossComponents components;
        
        // Component losses
        components.audio = audio_reconstruction_loss(audio_pred, audio_target);
        components.quantum = quantum_fidelity_loss(quantum_embeddings, positive_pairs, negative_pairs);
        components.classification = classification_loss(classification_logits, classification_labels);
        components.l2 = l2_regularization(parameters);
        components.normalization = normalization_penalty(quantum_embeddings);
        
        // Weighted sum
        components.total = 
            audio_weight_ * components.audio +
            quantum_weight_ * components.quantum +
            classification_weight_ * components.classification +
            l2_weight_ * components.l2 +
            normalization_weight_ * components.normalization;
        
        return components;
    }
    
private:
    double audio_weight_;
    double quantum_weight_;
    double classification_weight_;
    double l2_weight_;
    double normalization_weight_;
};



