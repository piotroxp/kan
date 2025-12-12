#pragma once

#include "audio_buffer.hpp"
#include <vector>
#include <random>
#include <algorithm>

// Audio preprocessing utilities
class AudioPreprocessor {
public:
    // Normalize audio to [-1, 1] range
    static void normalize(AudioBuffer& audio) {
        auto& samples = audio.samples();
        if (samples.empty()) return;
        
        // Find max absolute value
        float max_val = 0.0f;
        for (float sample : samples) {
            max_val = std::max(max_val, std::abs(sample));
        }
        
        // Normalize
        if (max_val > 1e-6f) {
            float scale = 1.0f / max_val;
            for (float& sample : samples) {
                sample *= scale;
            }
        }
    }
    
    // Pad or truncate to target length
    static void pad_or_truncate(AudioBuffer& audio, size_t target_samples) {
        auto& samples = audio.samples();
        if (samples.size() == target_samples) return;
        
        if (samples.size() < target_samples) {
            // Pad with zeros
            samples.resize(target_samples, 0.0f);
        } else {
            // Truncate
            samples.resize(target_samples);
        }
    }
    
    // Time stretching (simple resampling)
    static AudioBuffer time_stretch(const AudioBuffer& audio, double factor) {
        if (factor <= 0.0 || std::abs(factor - 1.0) < 1e-6) {
            return audio;  // No change
        }
        
        size_t new_size = static_cast<size_t>(audio.num_samples() / factor);
        AudioBuffer result(new_size, audio.sample_rate(), audio.channels());
        
        const auto& input = audio.samples();
        auto& output = result.samples();
        
        // Linear interpolation
        for (size_t i = 0; i < new_size; ++i) {
            double src_idx = i * factor;
            size_t idx0 = static_cast<size_t>(src_idx);
            size_t idx1 = std::min(idx0 + 1, input.size() - 1);
            double t = src_idx - idx0;
            
            output[i] = static_cast<float>((1.0 - t) * input[idx0] + t * input[idx1]);
        }
        
        return result;
    }
    
    // Pitch shifting (simple frequency scaling)
    static AudioBuffer pitch_shift(const AudioBuffer& audio, double semitones) {
        double factor = std::pow(2.0, semitones / 12.0);
        return time_stretch(audio, 1.0 / factor);
    }
    
    // Add Gaussian noise
    static void add_noise(AudioBuffer& audio, double noise_level, unsigned int seed = 0) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
        std::normal_distribution<float> dist(0.0f, static_cast<float>(noise_level));
        
        auto& samples = audio.samples();
        for (float& sample : samples) {
            sample += dist(gen);
            // Clamp to [-1, 1]
            sample = std::max(-1.0f, std::min(1.0f, sample));
        }
    }
    
    // Random time shift (circular shift)
    static void random_time_shift(AudioBuffer& audio, double max_shift_seconds, unsigned int seed = 0) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
        int max_shift_samples = static_cast<int>(max_shift_seconds * audio.sample_rate());
        std::uniform_int_distribution<int> dist(-max_shift_samples, max_shift_samples);
        
        int shift = dist(gen);
        if (shift == 0) return;
        
        auto& samples = audio.samples();
        std::vector<float> shifted(samples.size());
        
        for (size_t i = 0; i < samples.size(); ++i) {
            int src_idx = (static_cast<int>(i) - shift + static_cast<int>(samples.size())) 
                         % static_cast<int>(samples.size());
            shifted[i] = samples[src_idx];
        }
        
        samples = shifted;
    }
    
    // Apply all augmentations randomly
    static AudioBuffer augment(const AudioBuffer& audio, 
                              double time_stretch_range = 0.1,
                              double pitch_shift_range = 2.0,
                              double noise_level = 0.01,
                              unsigned int seed = 0) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
        std::uniform_real_distribution<double> time_dist(1.0 - time_stretch_range, 
                                                          1.0 + time_stretch_range);
        std::uniform_real_distribution<double> pitch_dist(-pitch_shift_range, pitch_shift_range);
        std::bernoulli_distribution apply_noise(0.5);
        
        AudioBuffer result = audio;
        
        // Time stretching
        double time_factor = time_dist(gen);
        result = time_stretch(result, time_factor);
        
        // Pitch shifting
        double semitones = pitch_dist(gen);
        result = pitch_shift(result, semitones);
        
        // Add noise (50% chance)
        if (apply_noise(gen)) {
            add_noise(result, noise_level, gen());
        }
        
        return result;
    }
};



