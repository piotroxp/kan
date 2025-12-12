#pragma once

#include "../audio/feature_extraction.hpp"
#include "../core/tensor.hpp"
#include "rocm_manager.hpp"
#include "feature_kernels.hpp"
#include <vector>
#include <memory>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

// GPU-accelerated mel-spectrogram extractor
class GPUMelSpectrogram {
public:
    GPUMelSpectrogram(
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
        sample_rate_(sample_rate),
        manager_(),
        use_gpu_(manager_.is_gpu_available()) {
        
        // Initialize mel filters on CPU first
        init_mel_filters();
        
        // Copy mel filters to GPU if available
        if (use_gpu_) {
            size_t mel_filters_size = n_mels_ * (n_fft_ / 2 + 1) * sizeof(float);
            gpu_mel_filters_ = manager_.allocate(mel_filters_size);
            
            // Convert to float and copy
            std::vector<float> mel_filters_float;
            for (const auto& filter : mel_filters_) {
                for (double val : filter) {
                    mel_filters_float.push_back(static_cast<float>(val));
                }
            }
            manager_.copy_to_device(mel_filters_float.data(), gpu_mel_filters_,
                                   mel_filters_size);
        }
    }
    
    ~GPUMelSpectrogram() {
        if (use_gpu_ && gpu_mel_filters_) {
            manager_.free(gpu_mel_filters_);
        }
    }
    
    // Process single audio buffer (GPU or CPU)
    Tensor process(const AudioBuffer& audio) {
        if (use_gpu_) {
            return process_gpu(audio);
        } else {
            // CPU fallback
            SincKANMelSpectrogram cpu_extractor(n_mels_, n_fft_, hop_length_, 
                                                win_length_, fmin_, fmax_, sample_rate_);
            return cpu_extractor.process(audio);
        }
    }
    
    // Process batch on GPU
    std::vector<Tensor> process_batch(const std::vector<AudioBuffer>& audio_batch) {
        if (use_gpu_) {
            return process_batch_gpu(audio_batch);
        } else {
            // CPU fallback
            std::vector<Tensor> results;
            SincKANMelSpectrogram cpu_extractor(n_mels_, n_fft_, hop_length_,
                                                win_length_, fmin_, fmax_, sample_rate_);
            for (const auto& audio : audio_batch) {
                results.push_back(cpu_extractor.process(audio));
            }
            return results;
        }
    }
    
    bool is_using_gpu() const { return use_gpu_; }
    
private:
    int n_mels_, n_fft_, hop_length_, win_length_;
    double fmin_, fmax_;
    int sample_rate_;
    
    ROCmMemoryManager manager_;
    bool use_gpu_;
    void* gpu_mel_filters_ = nullptr;
    
    std::vector<std::vector<double>> mel_filters_;
    
    void init_mel_filters() {
        // Same as SincKANMelSpectrogram::init_mel_filters()
        mel_filters_.resize(n_mels_);
        
        auto hz_to_mel = [](double hz) {
            return 2595.0 * std::log10(1.0 + hz / 700.0);
        };
        
        auto mel_to_hz = [](double mel) {
            return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
        };
        
        double mel_max = hz_to_mel(fmax_);
        double mel_min = hz_to_mel(fmin_);
        double mel_spacing = (mel_max - mel_min) / (n_mels_ + 1);
        
        std::vector<double> fft_freqs(n_fft_ / 2 + 1);
        for (int i = 0; i <= n_fft_ / 2; ++i) {
            fft_freqs[i] = static_cast<double>(i) * sample_rate_ / n_fft_;
        }
        
        for (int i = 0; i < n_mels_; ++i) {
            mel_filters_[i].resize(n_fft_ / 2 + 1, 0.0);
            
            double mel_center = mel_min + (i + 1) * mel_spacing;
            double mel_left = mel_min + i * mel_spacing;
            double mel_right = mel_min + (i + 2) * mel_spacing;
            
            double hz_center = mel_to_hz(mel_center);
            double hz_left = mel_to_hz(mel_left);
            double hz_right = mel_to_hz(mel_right);
            
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
    
    Tensor process_gpu(const AudioBuffer& audio) {
#ifdef USE_HIP
        // For single audio, use batched kernel with batch_size=1
        std::vector<AudioBuffer> single_batch = {audio};
        auto results = process_batch_gpu(single_batch);
        return results[0];
#else
        throw std::runtime_error("GPU not available");
#endif
    }
    
    std::vector<Tensor> process_batch_gpu(const std::vector<AudioBuffer>& audio_batch) {
#ifdef USE_HIP
        int batch_size = audio_batch.size();
        int num_samples = audio_batch[0].num_samples();
        int num_bins = n_fft_ / 2 + 1;
        
        // Calculate num_frames
        size_t num_frames = static_cast<size_t>(
            std::ceil((num_samples - win_length_) / static_cast<double>(hop_length_)) + 1
        );
        
        // Allocate GPU memory
        size_t audio_size = batch_size * num_samples * sizeof(float);
        size_t mel_spec_size = batch_size * num_frames * n_mels_ * sizeof(float);
        
        void* gpu_audio = manager_.allocate(audio_size);
        void* gpu_mel_spec = manager_.allocate(mel_spec_size);
        
        // Copy audio to GPU
        std::vector<float> audio_flat;
        audio_flat.reserve(batch_size * num_samples);
        for (const auto& audio : audio_batch) {
            for (size_t i = 0; i < audio.num_samples(); ++i) {
                audio_flat.push_back(static_cast<float>(audio[i]));
            }
        }
        manager_.copy_to_device(audio_flat.data(), gpu_audio, audio_size);
        
        // Launch batched mel-spectrogram kernel
        try {
            GPU::launch_batched_mel_spectrogram(
                static_cast<const float*>(gpu_audio),
                static_cast<const float*>(gpu_mel_filters_),
                static_cast<float*>(gpu_mel_spec),
                batch_size,
                num_samples,
                static_cast<int>(num_frames),
                num_bins,
                n_mels_,
                n_fft_,
                hop_length_,
                win_length_,
                nullptr
            );
            
            // Check for launch errors immediately
#ifdef USE_HIP
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                std::cerr << "[GPU ERROR] Batched mel-spectrogram launch failed: " 
                          << hipGetErrorString(err) << " (code: " << err << ")" << std::endl;
                manager_.free(gpu_audio);
                manager_.free(gpu_mel_spec);
                // Fall back to CPU
                return process_batch_cpu(audio_batch);
            }
            
            // Synchronize to catch any errors
            err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                std::cerr << "[GPU ERROR] Kernel synchronization failed: " 
                          << hipGetErrorString(err) << " (code: " << err << ")" << std::endl;
                manager_.free(gpu_audio);
                manager_.free(gpu_mel_spec);
                return process_batch_cpu(audio_batch);
            }
#endif
        } catch (...) {
            std::cerr << "[GPU ERROR] Exception during kernel launch, falling back to CPU" << std::endl;
            manager_.free(gpu_audio);
            manager_.free(gpu_mel_spec);
            return process_batch_cpu(audio_batch);
        }
        
        // Synchronize
        manager_.synchronize();
        
        // Copy results back
        std::vector<float> mel_spec_flat(batch_size * num_frames * n_mels_);
        manager_.copy_to_host(gpu_mel_spec, mel_spec_flat.data(), mel_spec_size);
        
        // Convert to Tensor vector
        std::vector<Tensor> results;
        for (int b = 0; b < batch_size; ++b) {
            Tensor mel_spec({num_frames, static_cast<size_t>(n_mels_)});
            for (size_t t = 0; t < num_frames; ++t) {
                for (int m = 0; m < n_mels_; ++m) {
                    size_t idx = b * num_frames * n_mels_ + t * n_mels_ + m;
                    mel_spec.at({t, static_cast<size_t>(m)}) = mel_spec_flat[idx];
                }
            }
            results.push_back(mel_spec);
        }
        
        // Free GPU memory
        manager_.free(gpu_audio);
        manager_.free(gpu_mel_spec);
        
        return results;
#else
        throw std::runtime_error("GPU not available");
#endif
    }
    
    // CPU fallback for batch processing
    std::vector<Tensor> process_batch_cpu(const std::vector<AudioBuffer>& audio_batch) {
        std::vector<Tensor> results;
        SincKANMelSpectrogram cpu_extractor(n_mels_, n_fft_, hop_length_,
                                            win_length_, fmin_, fmax_, sample_rate_);
        for (const auto& audio : audio_batch) {
            results.push_back(cpu_extractor.process(audio));
        }
        return results;
    }
};

