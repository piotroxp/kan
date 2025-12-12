#pragma once

#include "../core/tensor.hpp"
#include "../quantum/wavefunction.hpp"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

// Evaluation metrics for multi-label classification
class EvaluationMetrics {
public:
    // Mean Average Precision (mAP) for multi-label classification
    static double mean_average_precision(
        const std::vector<Tensor>& predictions,  // [batch, num_classes] - logits
        const std::vector<Tensor>& labels         // [batch, num_classes] - binary labels
    ) {
        if (predictions.empty() || predictions.size() != labels.size()) {
            return 0.0;
        }
        
        size_t num_classes = predictions[0].size();
        double total_ap = 0.0;
        
        // Compute AP for each class
        for (size_t c = 0; c < num_classes; ++c) {
            // Collect scores and labels for this class
            std::vector<std::pair<double, bool>> class_data;
            
            for (size_t i = 0; i < predictions.size(); ++i) {
                // Convert logit to probability (sigmoid)
                double score = 1.0 / (1.0 + std::exp(-predictions[i][c]));
                bool label = labels[i][c] > 0.5;
                class_data.push_back({score, label});
            }
            
            // Sort by score (descending)
            std::sort(class_data.begin(), class_data.end(),
                     [](const auto& a, const auto& b) {
                         return a.first > b.first;
                     });
            
            // Compute precision at each threshold
            double ap = 0.0;
            size_t true_positives = 0;
            size_t false_positives = 0;
            size_t total_positives = 0;
            
            // Count total positives
            for (const auto& item : class_data) {
                if (item.second) {
                    total_positives++;
                }
            }
            
            if (total_positives == 0) {
                continue;  // Skip if no positives for this class
            }
            
            // Compute AP using 11-point interpolation
            std::vector<double> precisions;
            for (size_t i = 0; i < class_data.size(); ++i) {
                if (class_data[i].second) {
                    true_positives++;
                } else {
                    false_positives++;
                }
                
                double precision = (true_positives + false_positives > 0) ?
                    static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
                precisions.push_back(precision);
            }
            
            // Average precision (simplified - in production use proper interpolation)
            double sum_precision = 0.0;
            for (double p : precisions) {
                sum_precision += p;
            }
            ap = (precisions.size() > 0) ? sum_precision / precisions.size() : 0.0;
            
            total_ap += ap;
        }
        
        return total_ap / num_classes;
    }
    
    // F1-score per class
    static std::vector<double> f1_score_per_class(
        const std::vector<Tensor>& predictions,
        const std::vector<Tensor>& labels,
        double threshold = 0.5
    ) {
        if (predictions.empty() || predictions.size() != labels.size()) {
            return {};
        }
        
        size_t num_classes = predictions[0].size();
        std::vector<double> f1_scores(num_classes, 0.0);
        
        for (size_t c = 0; c < num_classes; ++c) {
            size_t true_positives = 0;
            size_t false_positives = 0;
            size_t false_negatives = 0;
            
            for (size_t i = 0; i < predictions.size(); ++i) {
                // Convert logit to probability
                double prob = 1.0 / (1.0 + std::exp(-predictions[i][c]));
                bool pred = prob > threshold;
                bool label = labels[i][c] > 0.5;
                
                if (pred && label) {
                    true_positives++;
                } else if (pred && !label) {
                    false_positives++;
                } else if (!pred && label) {
                    false_negatives++;
                }
            }
            
            // Compute F1-score
            double precision = (true_positives + false_positives > 0) ?
                static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
            double recall = (true_positives + false_negatives > 0) ?
                static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
            
            if (precision + recall > 0) {
                f1_scores[c] = 2.0 * precision * recall / (precision + recall);
            }
        }
        
        return f1_scores;
    }
    
    // Macro-averaged F1-score
    static double macro_f1_score(
        const std::vector<Tensor>& predictions,
        const std::vector<Tensor>& labels,
        double threshold = 0.5
    ) {
        auto f1_per_class = f1_score_per_class(predictions, labels, threshold);
        if (f1_per_class.empty()) {
            return 0.0;
        }
        
        double sum_f1 = 0.0;
        for (double f1 : f1_per_class) {
            sum_f1 += f1;
        }
        
        return sum_f1 / f1_per_class.size();
    }
    
    // Micro-averaged F1-score
    static double micro_f1_score(
        const std::vector<Tensor>& predictions,
        const std::vector<Tensor>& labels,
        double threshold = 0.5
    ) {
        if (predictions.empty() || predictions.size() != labels.size()) {
            return 0.0;
        }
        
        size_t total_tp = 0;
        size_t total_fp = 0;
        size_t total_fn = 0;
        
        size_t num_classes = predictions[0].size();
        
        for (size_t c = 0; c < num_classes; ++c) {
            for (size_t i = 0; i < predictions.size(); ++i) {
                double prob = 1.0 / (1.0 + std::exp(-predictions[i][c]));
                bool pred = prob > threshold;
                bool label = labels[i][c] > 0.5;
                
                if (pred && label) {
                    total_tp++;
                } else if (pred && !label) {
                    total_fp++;
                } else if (!pred && label) {
                    total_fn++;
                }
            }
        }
        
        double precision = (total_tp + total_fp > 0) ?
            static_cast<double>(total_tp) / (total_tp + total_fp) : 0.0;
        double recall = (total_tp + total_fn > 0) ?
            static_cast<double>(total_tp) / (total_tp + total_fn) : 0.0;
        
        if (precision + recall > 0) {
            return 2.0 * precision * recall / (precision + recall);
        }
        
        return 0.0;
    }
    
    // Quantum embedding metrics
    struct QuantumMetrics {
        double avg_fidelity = 0.0;
        double avg_normalization = 0.0;
        double clustering_score = 0.0;
    };
    
    // Compute quantum embedding metrics
    static QuantumMetrics compute_quantum_metrics(
        const std::vector<Wavefunction>& embeddings
    ) {
        QuantumMetrics metrics;
        
        if (embeddings.empty()) {
            return metrics;
        }
        
        // Average fidelity (pairwise)
        double total_fidelity = 0.0;
        size_t pairs = 0;
        
        for (size_t i = 0; i < embeddings.size(); ++i) {
            for (size_t j = i + 1; j < embeddings.size(); ++j) {
                total_fidelity += embeddings[i].fidelity(embeddings[j]);
                pairs++;
            }
        }
        
        metrics.avg_fidelity = (pairs > 0) ? total_fidelity / pairs : 0.0;
        
        // Average normalization
        double total_norm = 0.0;
        for (const auto& psi : embeddings) {
            double norm_sq = 0.0;
            for (const auto& val : psi.values()) {
                double mag_sq = val.real() * val.real() + val.imag() * val.imag();
                norm_sq += psi.grid_spacing() * mag_sq;
            }
            total_norm += std::abs(norm_sq - 1.0);  // Deviation from 1.0
        }
        
        metrics.avg_normalization = total_norm / embeddings.size();
        
        // Clustering score (simplified - in production use silhouette score)
        metrics.clustering_score = metrics.avg_fidelity;
        
        return metrics;
    }
};

