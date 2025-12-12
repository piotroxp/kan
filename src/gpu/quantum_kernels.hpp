#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>

namespace GPU {
    // Compute squeezed coherent state wavefunction
    void launch_compute_squeezed_coherent(
        const float* alpha,
        const float* beta,
        const float* gamma,
        float* wavefunction_real,
        float* wavefunction_imag,
        int batch_size,
        int grid_size,
        float L,
        float sigma,
        hipStream_t stream = nullptr
    );
    
    // Batched quantum embedding (extract alpha, beta, gamma)
    void launch_batched_quantum_embedding(
        const float* features_batch,
        const float* alpha_coeffs,
        const float* beta_coeffs,
        const float* gamma_coeffs,
        float* alpha_out,
        float* beta_out,
        float* gamma_out,
        int batch_size,
        int feature_dim,
        int chebyshev_order,
        hipStream_t stream = nullptr
    );
}

#endif // USE_HIP


