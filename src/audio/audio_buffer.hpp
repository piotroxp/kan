#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

// Audio buffer for raw PCM data
// FSD50K format: PCM 16-bit, 44.1 kHz mono
class AudioBuffer {
public:
    AudioBuffer() : sample_rate_(44100), channels_(1) {}
    
    AudioBuffer(size_t num_samples, int sample_rate = 44100, int channels = 1)
        : samples_(num_samples, 0.0f), sample_rate_(sample_rate), channels_(channels) {}
    
    // Load from 16-bit PCM data
    void load_from_pcm16(const int16_t* data, size_t num_samples, 
                         int sample_rate = 44100, int channels = 1) {
        samples_.resize(num_samples);
        sample_rate_ = sample_rate;
        channels_ = channels;
        
        // Convert from int16 [-32768, 32767] to float [-1.0, 1.0]
        for (size_t i = 0; i < num_samples; ++i) {
            samples_[i] = static_cast<float>(data[i]) / 32768.0f;
        }
    }
    
    // Get samples
    const std::vector<float>& samples() const { return samples_; }
    std::vector<float>& samples() { return samples_; }
    
    // Access individual samples
    float& operator[](size_t idx) { return samples_[idx]; }
    const float& operator[](size_t idx) const { return samples_[idx]; }
    
    // Properties
    size_t num_samples() const { return samples_.size(); }
    int sample_rate() const { return sample_rate_; }
    int channels() const { return channels_; }
    double duration() const { 
        return static_cast<double>(samples_.size()) / (sample_rate_ * channels_); 
    }
    
    // Resize
    void resize(size_t num_samples) {
        samples_.resize(num_samples);
    }
    
    // Clear
    void clear() {
        samples_.clear();
    }
    
private:
    std::vector<float> samples_;  // Normalized to [-1.0, 1.0]
    int sample_rate_;
    int channels_;
};

// Simple WAV file reader (minimal implementation)
// For production, consider using libsndfile
class WAVReader {
public:
    // Read WAV file (basic implementation)
    // Returns true on success
    static bool read(const std::string& filename, AudioBuffer& buffer) {
        // For now, return false - will implement with libsndfile or manual parsing
        // This is a placeholder
        return false;
    }
};

