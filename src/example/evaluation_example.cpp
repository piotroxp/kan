// Example: Using evaluation metrics
#include "training/evaluation.hpp"
#include "training/backprop.hpp"
#include "model/speech_model.hpp"
#include <iostream>

int main() {
    std::cout << "=== Testing Evaluation Metrics ===" << std::endl;
    
    // Create dummy predictions and labels
    std::vector<Tensor> predictions;
    std::vector<Tensor> labels;
    
    for (int i = 0; i < 10; ++i) {
        Tensor pred({200});
        Tensor label({200});
        
        // Random predictions
        for (size_t j = 0; j < 200; ++j) {
            pred[j] = (i + j) % 10 - 5.0;  // Logits
            label[j] = (j < 3) ? 1.0 : 0.0;  // First 3 classes active
        }
        
        predictions.push_back(pred);
        labels.push_back(label);
    }
    
    // Compute metrics
    double map = EvaluationMetrics::mean_average_precision(predictions, labels);
    double macro_f1 = EvaluationMetrics::macro_f1_score(predictions, labels);
    double micro_f1 = EvaluationMetrics::micro_f1_score(predictions, labels);
    
    std::cout << "mAP: " << map << std::endl;
    std::cout << "Macro F1: " << macro_f1 << std::endl;
    std::cout << "Micro F1: " << micro_f1 << std::endl;
    
    std::cout << "\n=== Evaluation metrics test complete ===" << std::endl;
    
    return 0;
}



