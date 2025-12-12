// GPU kernels for feature extraction (mel-spectrogram)
// Optimized for AMD Radeon RX 7900 XTX (RDNA 3)

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_SIZE 256

// Compute magnitude from complex STFT
__global__ void compute_magnitude_kernel(
    const float* __restrict__ stft_real,
    const float* __restrict__ stft_imag,
    float* __restrict__ magnitude,
    int num_frames,
    int num_bins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * num_bins;
    
    if (idx < total) {
        int frame = idx / num_bins;
        int bin = idx % num_bins;
        int stft_idx = frame * num_bins + bin;
        
        float real = stft_real[stft_idx];
        float imag = stft_imag[stft_idx];
        magnitude[stft_idx] = __fsqrt_rn(real * real + imag * imag);
    }
}

// Apply mel filter bank (optimized for GPU)
__global__ void apply_mel_filters_kernel(
    const float* __restrict__ magnitude,
    const float* __restrict__ mel_filters,  // [n_mels][num_bins]
    float* __restrict__ mel_spec,           // [num_frames][n_mels]
    int num_frames,
    int num_bins,
    int n_mels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * n_mels;
    
    if (idx < total) {
        int frame = idx / n_mels;
        int mel = idx % n_mels;
        
        float sum = 0.0f;
        for (int f = 0; f < num_bins; ++f) {
            float mag_val = magnitude[frame * num_bins + f];
            float filter_val = mel_filters[mel * num_bins + f];
            sum += mag_val * filter_val;
        }
        
        // Log scale with epsilon
        mel_spec[frame * n_mels + mel] = __logf(sum + 1e-10f);
    }
}

// Batched mel-spectrogram computation
__global__ void batched_mel_spectrogram_kernel(
    const float* __restrict__ audio_batch,      // [batch_size][num_samples]
    const float* __restrict__ mel_filters,     // [n_mels][num_bins]
    float* __restrict__ mel_spec_batch,        // [batch_size][num_frames][n_mels]
    int batch_size,
    int num_samples,
    int num_frames,
    int num_bins,
    int n_mels,
    int n_fft,
    int hop_length,
    int win_length
) {
    int batch_idx = blockIdx.x;
    int frame_mel_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Process each (frame, mel) pair
    int total_pairs = num_frames * n_mels;
    for (int i = frame_mel_idx; i < total_pairs; i += blockDim.x) {
        int frame = i / n_mels;
        int mel = i % n_mels;
        
        // Compute STFT for this frame (simplified - in production use cuFFT)
        float sum = 0.0f;
        int start = frame * hop_length;
        
        for (int f = 0; f < num_bins; ++f) {
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            
            // Simple DFT for this frequency bin
            for (int n = 0; n < win_length && (start + n) < num_samples; ++n) {
                float sample = audio_batch[batch_idx * num_samples + start + n];
                float window = 0.5f * (1.0f - __cosf(2.0f * M_PI * n / (win_length - 1)));
                float angle = -2.0f * M_PI * f * n / n_fft;
                
                real_sum += sample * window * __cosf(angle);
                imag_sum += sample * window * __sinf(angle);
            }
            
            float magnitude = __fsqrt_rn(real_sum * real_sum + imag_sum * imag_sum);
            float filter_val = mel_filters[mel * num_bins + f];
            sum += magnitude * filter_val;
        }
        
        mel_spec_batch[batch_idx * num_frames * n_mels + frame * n_mels + mel] = __logf(sum + 1e-10f);
    }
}

// Launch wrappers
namespace GPU {
    void launch_compute_magnitude(
        const float* stft_real,
        const float* stft_imag,
        float* magnitude,
        int num_frames,
        int num_bins,
        hipStream_t stream = nullptr
    ) {
        int total = num_frames * num_bins;
        dim3 block(BLOCK_SIZE);
        dim3 grid((total + block.x - 1) / block.x);
        compute_magnitude_kernel<<<grid, block, 0, stream>>>(
            stft_real, stft_imag, magnitude, num_frames, num_bins
        );
    }
    
    void launch_apply_mel_filters(
        const float* magnitude,
        const float* mel_filters,
        float* mel_spec,
        int num_frames,
        int num_bins,
        int n_mels,
        hipStream_t stream = nullptr
    ) {
        int total = num_frames * n_mels;
        dim3 block(BLOCK_SIZE);
        dim3 grid((total + block.x - 1) / block.x);
        apply_mel_filters_kernel<<<grid, block, 0, stream>>>(
            magnitude, mel_filters, mel_spec, num_frames, num_bins, n_mels
        );
    }
    
    void launch_batched_mel_spectrogram(
        const float* audio_batch,
        const float* mel_filters,
        float* mel_spec_batch,
        int batch_size,
        int num_samples,
        int num_frames,
        int num_bins,
        int n_mels,
        int n_fft,
        int hop_length,
        int win_length,
        hipStream_t stream = nullptr
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid(batch_size);
        batched_mel_spectrogram_kernel<<<grid, block, 0, stream>>>(
            audio_batch, mel_filters, mel_spec_batch,
            batch_size, num_samples, num_frames, num_bins, n_mels,
            n_fft, hop_length, win_length
        );
    }
}

#endif // USE_HIP

