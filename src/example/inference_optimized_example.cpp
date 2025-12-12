// Example: Optimized inference with quantization
#include "inference/optimized_inference.hpp"
#include "model/speech_model.hpp"
#include "audio/audio_buffer.hpp"
#include "audio/preprocessing.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    std::cout << "=== Optimized Inference Example ===" << std::endl;
    std::cout << std::endl;
    
    // Create model
    SpeechModel model;
    
    // Create optimized inference engine
    std::cout << "1. Creating optimized inference engine..." << std::endl;
    OptimizedInferenceEngine inference(model, true, false);  // Quantization on, GPU off for now
    
    std::cout << "   Quantization error: " << inference.get_quantization_error() << std::endl;
    std::cout << "   Model size (float): " << (inference.get_float_size_bytes() / 1024.0) << " KB" << std::endl;
    std::cout << "   Model size (quantized): " << (inference.get_quantized_size_bytes() / 1024.0) << " KB" << std::endl;
    std::cout << "   Compression ratio: " 
              << (static_cast<double>(inference.get_float_size_bytes()) / inference.get_quantized_size_bytes())
              << "x" << std::endl;
    std::cout << std::endl;
    
    // Create test audio
    std::cout << "2. Creating test audio..." << std::endl;
    AudioBuffer audio(22050, 44100, 1);  // 0.5 seconds
    for (size_t i = 0; i < audio.num_samples(); ++i) {
        double t = static_cast<double>(i) / audio.sample_rate();
        audio[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * 440.0 * t));
    }
    AudioPreprocessor::normalize(audio);
    std::cout << "   Audio: " << audio.num_samples() << " samples, " 
              << audio.duration() << " seconds" << std::endl;
    std::cout << std::endl;
    
    // Test inference speed
    std::cout << "3. Testing inference speed..." << std::endl;
    const int num_runs = 10;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        auto output = inference.forward(audio);
        (void)output;  // Suppress unused warning
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * num_runs);
    double throughput = 1000.0 / avg_time_ms;
    
    std::cout << "   Average inference time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "   Throughput: " << throughput << " samples/second" << std::endl;
    std::cout << std::endl;
    
    // Test batch inference
    std::cout << "4. Testing batch inference..." << std::endl;
    std::vector<AudioBuffer> batch(8, audio);
    
    start = std::chrono::high_resolution_clock::now();
    auto batch_outputs = inference.forward_batch(batch);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double batch_time_ms = duration.count() / 1000.0;
    
    std::cout << "   Batch size: " << batch.size() << std::endl;
    std::cout << "   Batch time: " << batch_time_ms << " ms" << std::endl;
    std::cout << "   Time per sample: " << (batch_time_ms / batch.size()) << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Optimized inference example complete ===" << std::endl;
    
    return 0;
}

