#include <iostream>
#include "model/speech_model.hpp"
#include "inference/inference_engine.hpp"
#include "audio/audio_buffer.hpp"
#include "core/tensor.hpp"

int main() {
    std::cout << "=== Full Model Integration Example ===\n\n";
    
    // 1. Create model
    std::cout << "1. Creating SpeechModel...\n";
    SpeechModel model(80, 256, 200, 1024);
    std::cout << "   Model created with 200 output classes\n\n";
    
    // 2. Create synthetic audio
    std::cout << "2. Creating synthetic audio...\n";
    AudioBuffer audio(44100, 44100, 1);  // 1 second
    double frequency = 440.0;
    for (size_t i = 0; i < audio.num_samples(); ++i) {
        double t = static_cast<double>(i) / audio.sample_rate();
        audio[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * frequency * t));
    }
    std::cout << "   Audio: " << audio.duration() << " seconds, " 
              << audio.num_samples() << " samples\n\n";
    
    // 3. Forward pass
    std::cout << "3. Running forward pass...\n";
    auto output = model.forward(audio);
    
    std::cout << "   Audio features shape: [" 
              << output.audio_features.shape()[0] << ", "
              << output.audio_features.shape()[1] << "]\n";
    std::cout << "   Quantum embeddings: " << output.quantum_embeddings.size() << "\n";
    std::cout << "   Semantic representation shape: [" 
              << output.semantic_representation.shape()[0] << "]\n";
    std::cout << "   Classification logits shape: [" 
              << output.classification_logits.shape()[0] << "]\n\n";
    
    // 4. Get top predictions
    std::cout << "4. Top 5 predictions:\n";
    std::vector<std::pair<int, double>> predictions;
    for (size_t i = 0; i < output.classification_logits.size() && i < 5; ++i) {
        double prob = 1.0 / (1.0 + std::exp(-output.classification_logits[i]));
        predictions.push_back({static_cast<int>(i), prob});
    }
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (const auto& [idx, prob] : predictions) {
        std::cout << "   Class " << idx << ": " << prob << "\n";
    }
    std::cout << "\n";
    
    // 5. Inference engine
    std::cout << "5. Testing InferenceEngine...\n";
    InferenceEngine engine("", 44100, 22050);
    
    // Process audio
    auto inference_output = engine.process_audio(audio);
    auto top_preds = engine.get_top_predictions(5);
    
    std::cout << "   Processed audio through inference engine\n";
    std::cout << "   Top predictions: " << top_preds.size() << "\n\n";
    
    std::cout << "=== Full model integration completed successfully! ===\n";
    
    return 0;
}


