#pragma once

#include "../model/speech_model.hpp"
#include "../audio/audio_buffer.hpp"
#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Circular buffer for streaming audio
class CircularBuffer {
public:
    CircularBuffer(size_t capacity) : capacity_(capacity) {
        buffer_.resize(capacity, 0.0f);
        write_pos_ = 0;
        size_ = 0;
    }
    
    void push(float sample) {
        buffer_[write_pos_] = sample;
        write_pos_ = (write_pos_ + 1) % capacity_;
        if (size_ < capacity_) {
            size_++;
        }
    }
    
    void push_batch(const std::vector<float>& samples) {
        for (float s : samples) {
            push(s);
        }
    }
    
    AudioBuffer get_window(size_t window_size) {
        AudioBuffer result(window_size, 44100, 1);
        
        size_t start = (write_pos_ - window_size + capacity_) % capacity_;
        for (size_t i = 0; i < window_size && i < size_; ++i) {
            size_t idx = (start + i) % capacity_;
            result[i] = buffer_[idx];
        }
        
        return result;
    }
    
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
private:
    std::vector<float> buffer_;
    size_t capacity_;
    size_t write_pos_;
    size_t size_;
};

// Inference engine for real-time processing
class InferenceEngine {
public:
    InferenceEngine(
        const std::string& model_path = "",
        size_t buffer_size = 44100,  // 1 second at 44.1 kHz
        size_t hop_size = 22050      // 0.5 second hop
    ) : model_(80, 256, 200, 1024),
        audio_buffer_(buffer_size),
        hop_size_(hop_size),
        buffer_size_(buffer_size) {
        
        // Load model if path provided
        if (!model_path.empty()) {
            // In production, load from checkpoint
        }
    }
    
    // Process audio stream chunk
    void process_stream(const std::vector<float>& audio_chunk) {
        audio_buffer_.push_batch(audio_chunk);
        
        // Process if we have enough data
        if (audio_buffer_.size() >= buffer_size_) {
            AudioBuffer window = audio_buffer_.get_window(buffer_size_);
            process_audio(window);
        }
    }
    
    // Process complete audio buffer
    SpeechModel::ModelOutput process_audio(const AudioBuffer& audio) {
        last_output_ = model_.forward(audio);
        return last_output_;
    }
    
    // Get current predictions
    std::vector<std::pair<int, double>> get_top_predictions(int top_k = 5) {
        if (last_output_.classification_logits.size() == 0) {
            return {};
        }
        
        // Convert logits to probabilities (sigmoid for multi-label)
        std::vector<std::pair<int, double>> predictions;
        const auto& logits = last_output_.classification_logits;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            double prob = 1.0 / (1.0 + std::exp(-logits[i]));  // Sigmoid
            predictions.push_back({static_cast<int>(i), prob});
        }
        
        // Sort by probability
        std::sort(predictions.begin(), predictions.end(),
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;
                  });
        
        // Return top-k
        if (top_k > 0 && top_k < static_cast<int>(predictions.size())) {
            predictions.resize(top_k);
        }
        
        return predictions;
    }
    
    // Get classification as string (for FSD50K classes)
    std::string get_classification_string(const std::vector<std::string>& class_names, 
                                         double threshold = 0.5) {
        auto predictions = get_top_predictions(10);
        std::string result;
        
        for (const auto& [idx, prob] : predictions) {
            if (prob >= threshold && idx < static_cast<int>(class_names.size())) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += class_names[idx] + " (" + std::to_string(prob) + ")";
            }
        }
        
        return result;
    }
    
    // Get quantum embedding similarity
    double get_quantum_similarity(const Wavefunction& other) {
        if (last_output_.quantum_embeddings.empty()) {
            return 0.0;
        }
        
        return last_output_.quantum_embeddings[0].fidelity(other);
    }
    
private:
    SpeechModel model_;
    CircularBuffer audio_buffer_;
    size_t hop_size_;
    size_t buffer_size_;
    SpeechModel::ModelOutput last_output_;
};
