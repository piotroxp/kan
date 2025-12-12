#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>

namespace GPU {
    // Compute magnitude from complex STFT
    void launch_compute_magnitude(
        const float* stft_real,
        const float* stft_imag,
        float* magnitude,
        int num_frames,
        int num_bins,
        hipStream_t stream = nullptr
    );
    
    // Apply mel filter bank
    void launch_apply_mel_filters(
        const float* magnitude,
        const float* mel_filters,
        float* mel_spec,
        int num_frames,
        int num_bins,
        int n_mels,
        hipStream_t stream = nullptr
    );
    
    // Batched mel-spectrogram computation (entire batch on GPU)
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
    );
}

#endif // USE_HIP

