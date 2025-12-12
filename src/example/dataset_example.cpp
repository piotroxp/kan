// Example: Using FSD50K data loader
#include "data/fsd50k_data_loader.hpp"
#include "model/speech_model.hpp"
#include "training/evaluation.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "=== FSD50K Data Loader Example ===" << std::endl;
    
    // Dataset path (adjust as needed)
    std::string dataset_path = (argc > 1) ? argv[1] : "/path/to/FSD50K";
    
    try {
        // Create data loader
        FSD50KDataLoader loader(dataset_path, "dev", 8);
        
        std::cout << "Dataset: " << dataset_path << std::endl;
        std::cout << "Split: dev" << std::endl;
        std::cout << "Total clips: " << loader.num_clips() << std::endl;
        std::cout << "Number of batches: " << loader.num_batches() << std::endl;
        std::cout << "Number of classes: " << loader.num_classes() << std::endl;
        std::cout << std::endl;
        
        // Load a batch
        std::cout << "Loading batch 0..." << std::endl;
        auto batch = loader.get_batch(0);
        
        std::cout << "Batch size: " << batch.audio.size() << std::endl;
        for (size_t i = 0; i < batch.audio.size() && i < 3; ++i) {
            std::cout << "  Clip " << i << ": " << batch.clip_names[i] 
                      << " (" << batch.audio[i].num_samples() << " samples)" << std::endl;
        }
        
        // Test with model
        std::cout << "\nTesting with SpeechModel..." << std::endl;
        SpeechModel model;
        
        std::vector<Tensor> predictions;
        std::vector<Tensor> labels;
        
        for (size_t i = 0; i < batch.audio.size() && i < 5; ++i) {
            auto output = model.forward(batch.audio[i]);
            predictions.push_back(output.classification_logits);
            labels.push_back(batch.labels[i]);
        }
        
        // Compute metrics
        double map = EvaluationMetrics::mean_average_precision(predictions, labels);
        double macro_f1 = EvaluationMetrics::macro_f1_score(predictions, labels);
        
        std::cout << "Metrics on batch:" << std::endl;
        std::cout << "  mAP: " << map << std::endl;
        std::cout << "  Macro F1: " << macro_f1 << std::endl;
        
        std::cout << "\n=== Data loader example complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Note: If dataset path is incorrect, synthetic data will be generated" << std::endl;
        return 1;
    }
    
    return 0;
}

