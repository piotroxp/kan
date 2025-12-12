// GPU kernels for feature extraction (mel-spectrogram)
// Optimized for AMD Radeon RX 7900 XTX (RDNA 3)

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cmath>
#include <iostream>

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

// Optimized: Compute magnitude for one frame (used by batched kernel)
__device__ void compute_frame_magnitude(
    const float* audio,
    float* magnitude_out,
    int start,
    int num_samples,
    int n_fft,
    int win_length,
    int num_bins
) {
    // Apply window and compute magnitude for all frequency bins in parallel
    int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (bin_idx >= num_bins) return;
    
    float real_sum = 0.0f;
    float imag_sum = 0.0f;
    
    for (int n = 0; n < win_length && (start + n) < num_samples; ++n) {
        float sample = audio[start + n];
        float window = 0.5f * (1.0f - __cosf(2.0f * M_PI * n / (win_length - 1)));
        float angle = -2.0f * M_PI * bin_idx * n / n_fft;
        
        real_sum += sample * window * __cosf(angle);
        imag_sum += sample * window * __sinf(angle);
    }
    
    magnitude_out[bin_idx] = __fsqrt_rn(real_sum * real_sum + imag_sum * imag_sum);
}

// Batched mel-spectrogram computation (optimized with better parallelism)
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
    // Each thread processes one (frame, mel) pair
    int batch_idx = blockIdx.z;
    int frame_idx = blockIdx.y;
    int mel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || frame_idx >= num_frames || mel_idx >= n_mels) return;
    
    int start = frame_idx * hop_length;
    
    // Compute magnitude for this frame (shared across all mel bins)
    // Use shared memory for magnitude if possible, otherwise compute per mel
    float sum = 0.0f;
    
    // Compute magnitude for each frequency bin and apply mel filter
    for (int f = 0; f < num_bins; ++f) {
        // Compute STFT magnitude for this bin
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        
        for (int n = 0; n < win_length && (start + n) < num_samples; ++n) {
            float sample = audio_batch[batch_idx * num_samples + start + n];
            float window = 0.5f * (1.0f - __cosf(2.0f * M_PI * n / (win_length - 1)));
            float angle = -2.0f * M_PI * f * n / n_fft;
            
            real_sum += sample * window * __cosf(angle);
            imag_sum += sample * window * __sinf(angle);
        }
        
        float magnitude = __fsqrt_rn(real_sum * real_sum + imag_sum * imag_sum);
        float filter_val = mel_filters[mel_idx * num_bins + f];
        sum += magnitude * filter_val;
    }
    
    int output_idx = batch_idx * num_frames * n_mels + frame_idx * n_mels + mel_idx;
    mel_spec_batch[output_idx] = __logf(sum + 1e-10f);
}

// Optimized batched mel-spectrogram kernel (better parallelism)
__global__ void batched_mel_spectrogram_optimized_kernel(
    const float* __restrict__ audio_batch,
    const float* __restrict__ mel_filters,
    float* __restrict__ mel_spec_batch,
    int batch_size,
    int num_samples,
    int num_frames,
    int num_bins,
    int n_mels,
    int n_fft,
    int hop_length,
    int win_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ops = batch_size * num_frames * n_mels;
    
    if (idx >= total_ops) return;
    
    // Decompose index
    int batch_idx = idx / (num_frames * n_mels);
    int remainder = idx % (num_frames * n_mels);
    int frame_idx = remainder / n_mels;
    int mel_idx = remainder % n_mels;
    
    int start = frame_idx * hop_length;
    
    // Compute STFT magnitude and apply mel filter
    float sum = 0.0f;
    for (int f = 0; f < num_bins; ++f) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        
        // Optimized DFT computation
        for (int n = 0; n < win_length && (start + n) < num_samples; ++n) {
            float sample = audio_batch[batch_idx * num_samples + start + n];
            float window = 0.5f * (1.0f - __cosf(2.0f * M_PI * n / (win_length - 1)));
            float angle = -2.0f * M_PI * f * n / n_fft;
            
            real_sum += sample * window * __cosf(angle);
            imag_sum += sample * window * __sinf(angle);
        }
        
        float magnitude = __fsqrt_rn(real_sum * real_sum + imag_sum * imag_sum);
        float filter_val = mel_filters[mel_idx * num_bins + f];
        sum += magnitude * filter_val;
    }
    
    int output_idx = batch_idx * num_frames * n_mels + frame_idx * n_mels + mel_idx;
    mel_spec_batch[output_idx] = __logf(sum + 1e-10f);
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
        // For now, use the original kernel (will optimize with rocFFT later)
        // Optimized grid: (n_mels, num_frames, batch_size) for better parallelism
        dim3 block(256);  // Threads per block for mel bins
        dim3 grid((n_mels + block.x - 1) / block.x, num_frames, batch_size);
        
        // Use a simpler, more efficient kernel that processes frames in parallel
        // Each thread handles one (frame, mel) pair
        int total_ops = batch_size * num_frames * n_mels;
        dim3 simple_block(256);
        dim3 simple_grid((total_ops + simple_block.x - 1) / simple_block.x);
        
        // Launch optimized kernel (will be replaced with rocFFT version)
        batched_mel_spectrogram_optimized_kernel<<<simple_grid, simple_block, 0, stream>>>(
            audio_batch, mel_filters, mel_spec_batch,
            batch_size, num_samples, num_frames, num_bins, n_mels,
            n_fft, hop_length, win_length
        );
        
        // Check for errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "[KERNEL ERROR] Batched mel-spectrogram launch failed: " 
                      << hipGetErrorString(err) << std::endl;
        }
    }
    
    // Optimized batched mel-spectrogram kernel (better parallelism)
    __global__ void batched_mel_spectrogram_optimized_kernel(
        const float* __restrict__ audio_batch,
        const float* __restrict__ mel_filters,
        float* __restrict__ mel_spec_batch,
        int batch_size,
        int num_samples,
        int num_frames,
        int num_bins,
        int n_mels,
        int n_fft,
        int hop_length,
        int win_length
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_ops = batch_size * num_frames * n_mels;
        
        if (idx >= total_ops) return;
        
        // Decompose index
        int batch_idx = idx / (num_frames * n_mels);
        int remainder = idx % (num_frames * n_mels);
        int frame_idx = remainder / n_mels;
        int mel_idx = remainder % n_mels;
        
        int start = frame_idx * hop_length;
        
        // Compute STFT magnitude and apply mel filter
        float sum = 0.0f;
        for (int f = 0; f < num_bins; ++f) {
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            
            // Optimized DFT computation
            for (int n = 0; n < win_length && (start + n) < num_samples; ++n) {
                float sample = audio_batch[batch_idx * num_samples + start + n];
                float window = 0.5f * (1.0f - __cosf(2.0f * M_PI * n / (win_length - 1)));
                float angle = -2.0f * M_PI * f * n / n_fft;
                
                real_sum += sample * window * __cosf(angle);
                imag_sum += sample * window * __sinf(angle);
            }
            
            float magnitude = __fsqrt_rn(real_sum * real_sum + imag_sum * imag_sum);
            float filter_val = mel_filters[mel_idx * num_bins + f];
            sum += magnitude * filter_val;
        }
        
        int output_idx = batch_idx * num_frames * n_mels + frame_idx * n_mels + mel_idx;
        mel_spec_batch[output_idx] = __logf(sum + 1e-10f);
    }
}

#endif // USE_HIP

