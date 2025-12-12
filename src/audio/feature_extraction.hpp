#pragma once

#include "audio_buffer.hpp"
#include "../core/tensor.hpp"
#include "../core/sinc_kan.hpp"
#include <vector>
#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Mel-spectrogram extractor using SincKAN
class SincKANMelSpectrogram {
public:
    SincKANMelSpectrogram(
        int n_mels = 80,
        int n_fft = 2048,
        int hop_length = 512,
        int win_length = 2048,
        double fmin = 0.0,
        double fmax = 22050.0,
        int sample_rate = 44100
    ) : n_mels_(n_mels),
        n_fft_(n_fft),
        hop_length_(hop_length),
        win_length_(win_length),
        fmin_(fmin),
        fmax_(fmax),
        sample_rate_(sample_rate) {
        
        // Initialize mel filter bank
        init_mel_filters();
        
        // Note: SincKAN kernels are not used in the current implementation
        // They can be added later for learned frequency decomposition
        // frequency_kernels_ would need to be initialized differently
    }
    
    // Process audio buffer to mel-spectrogram
    Tensor process(const AudioBuffer& audio) {
        // Step 1: Compute STFT
        auto stft = compute_stft(audio);
        
        // Step 2: Convert to magnitude spectrogram
        auto magnitude = compute_magnitude(stft);
        
        // Step 3: Apply mel filter bank
        auto mel_spec = apply_mel_filters(magnitude);
        
        // Step 4: Apply SincKAN processing (optional enhancement)
        // For now, return mel spectrogram directly
        // SincKAN can be used for learned frequency decomposition
        
        return mel_spec;
    }
    
    // Get output shape for a given audio duration
    std::vector<size_t> output_shape(double duration_seconds) const {
        size_t num_frames = static_cast<size_t>(
            std::ceil((duration_seconds * sample_rate_ - win_length_) / hop_length_) + 1
        );
        return {num_frames, static_cast<size_t>(n_mels_)};
    }
    
private:
    int n_mels_;
    int n_fft_;
    int hop_length_;
    int win_length_;
    double fmin_;
    double fmax_;
    int sample_rate_;
    
    // SincKAN kernels for learned frequency decomposition (optional, for future use)
    // std::vector<SincKANLayer> frequency_kernels_;
    std::vector<std::vector<double>> mel_filters_;  // Mel filter bank
    
    // Initialize mel filter bank
    void init_mel_filters() {
        mel_filters_.resize(n_mels_);
        
        // Convert frequency to mel scale
        auto hz_to_mel = [](double hz) {
            return 2595.0 * std::log10(1.0 + hz / 700.0);
        };
        
        auto mel_to_hz = [](double mel) {
            return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
        };
        
        // Compute mel frequencies
        double mel_max = hz_to_mel(fmax_);
        double mel_min = hz_to_mel(fmin_);
        double mel_spacing = (mel_max - mel_min) / (n_mels_ + 1);
        
        // FFT bin frequencies
        std::vector<double> fft_freqs(n_fft_ / 2 + 1);
        for (int i = 0; i <= n_fft_ / 2; ++i) {
            fft_freqs[i] = static_cast<double>(i) * sample_rate_ / n_fft_;
        }
        
        // Create triangular mel filters
        for (int i = 0; i < n_mels_; ++i) {
            mel_filters_[i].resize(n_fft_ / 2 + 1, 0.0);
            
            double mel_center = mel_min + (i + 1) * mel_spacing;
            double mel_left = mel_min + i * mel_spacing;
            double mel_right = mel_min + (i + 2) * mel_spacing;
            
            double hz_center = mel_to_hz(mel_center);
            double hz_left = mel_to_hz(mel_left);
            double hz_right = mel_to_hz(mel_right);
            
            // Triangular filter
            for (int j = 0; j <= n_fft_ / 2; ++j) {
                double freq = fft_freqs[j];
                if (freq >= hz_left && freq <= hz_right) {
                    if (freq <= hz_center) {
                        mel_filters_[i][j] = (freq - hz_left) / (hz_center - hz_left);
                    } else {
                        mel_filters_[i][j] = (hz_right - freq) / (hz_right - hz_center);
                    }
                }
            }
        }
    }
    
    // Compute STFT (Short-Time Fourier Transform)
    std::vector<std::vector<std::complex<double>>> compute_stft(const AudioBuffer& audio) {
        size_t num_frames = output_shape(audio.duration())[0];
        std::vector<std::vector<std::complex<double>>> stft(
            num_frames, 
            std::vector<std::complex<double>>(n_fft_ / 2 + 1)
        );
        
        // Hanning window
        std::vector<double> window(win_length_);
        for (int i = 0; i < win_length_; ++i) {
            window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (win_length_ - 1)));
        }
        
        const auto& samples = audio.samples();
        size_t num_samples = samples.size();
        
        // Process each frame
        for (size_t frame = 0; frame < num_frames; ++frame) {
            size_t start = frame * hop_length_;
            
            // Apply window and compute FFT
            std::vector<std::complex<double>> frame_data(n_fft_, 0.0);
            for (int i = 0; i < win_length_ && (start + i) < num_samples; ++i) {
                frame_data[i] = samples[start + i] * window[i];
            }
            
            // Simple DFT (for production, use FFTW or similar)
            for (int k = 0; k <= n_fft_ / 2; ++k) {
                std::complex<double> sum(0.0, 0.0);
                for (int n = 0; n < n_fft_; ++n) {
                    double angle = -2.0 * M_PI * k * n / n_fft_;
                    sum += frame_data[n] * std::complex<double>(std::cos(angle), std::sin(angle));
                }
                stft[frame][k] = sum;
            }
        }
        
        return stft;
    }
    
    // Compute magnitude spectrogram
    Tensor compute_magnitude(const std::vector<std::vector<std::complex<double>>>& stft) {
        size_t num_frames = stft.size();
        size_t num_bins = stft[0].size();
        
        Tensor magnitude({num_frames, num_bins});
        
        for (size_t t = 0; t < num_frames; ++t) {
            for (size_t f = 0; f < num_bins; ++f) {
                double real = stft[t][f].real();
                double imag = stft[t][f].imag();
                magnitude.at({t, f}) = std::sqrt(real * real + imag * imag);
            }
        }
        
        return magnitude;
    }
    
    // Apply mel filter bank
    Tensor apply_mel_filters(const Tensor& magnitude) {
        size_t num_frames = magnitude.shape()[0];
        size_t num_bins = magnitude.shape()[1];
        
        Tensor mel_spec({num_frames, static_cast<size_t>(n_mels_)});
        mel_spec.fill(0.0);
        
        for (size_t t = 0; t < num_frames; ++t) {
            for (size_t m = 0; m < static_cast<size_t>(n_mels_); ++m) {
                double sum = 0.0;
                for (size_t f = 0; f < num_bins; ++f) {
                    sum += magnitude.at({t, f}) * mel_filters_[m][f];
                }
                // Log scale (add small epsilon to avoid log(0))
                mel_spec.at({t, m}) = std::log(sum + 1e-10);
            }
        }
        
        return mel_spec;
    }
};

