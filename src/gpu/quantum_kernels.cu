// GPU kernels for quantum embedding computation
// Optimized for AMD Radeon RX 7900 XTX (RDNA 3)

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_SIZE 256

// Compute squeezed coherent state wavefunction
__global__ void compute_squeezed_coherent_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    float* __restrict__ wavefunction_real,
    float* __restrict__ wavefunction_imag,
    int batch_size,
    int grid_size,
    float L,
    float sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * grid_size;
    
    if (idx < total) {
        int batch_idx = idx / grid_size;
        int grid_idx = idx % grid_size;
        
        float a = alpha[batch_idx];
        float b = beta[batch_idx];
        float g = gamma[batch_idx];
        
        // Grid position
        float x = -L / 2.0f + (grid_idx * L) / (grid_size - 1);
        
        // Squeezed coherent state: ψ(x) = N * exp[-(x-α)²/(4σ²) + iβx + iγ]
        float x_minus_alpha = x - a;
        float exp_arg = -(x_minus_alpha * x_minus_alpha) / (4.0f * sigma * sigma);
        float exp_val = __expf(exp_arg);
        
        float phase = b * x + g;
        float real_part = exp_val * __cosf(phase);
        float imag_part = exp_val * __sinf(phase);
        
        // Normalization (simplified)
        float norm = 1.0f / __fsqrt_rn(M_PI * 2.0f * sigma * sigma);
        
        wavefunction_real[idx] = real_part * norm;
        wavefunction_imag[idx] = imag_part * norm;
    }
}

// Batched quantum embedding (extract alpha, beta, gamma using Chebyshev KAN on GPU)
__global__ void batched_quantum_embedding_kernel(
    const float* __restrict__ features_batch,  // [batch_size][feature_dim]
    const float* __restrict__ alpha_coeffs,    // Chebyshev KAN coefficients [feature_dim][chebyshev_order+1]
    const float* __restrict__ beta_coeffs,
    const float* __restrict__ gamma_coeffs,
    float* __restrict__ alpha_out,
    float* __restrict__ beta_out,
    float* __restrict__ gamma_out,
    int batch_size,
    int feature_dim,
    int chebyshev_order
) {
    int batch_idx = blockIdx.x;
    int param_idx = threadIdx.x;  // 0=alpha, 1=beta, 2=gamma
    
    if (batch_idx >= batch_size || param_idx >= 3) return;
    
    float* output = (param_idx == 0) ? &alpha_out[batch_idx] : 
                    (param_idx == 1) ? &beta_out[batch_idx] : &gamma_out[batch_idx];
    const float* coeffs = (param_idx == 0) ? alpha_coeffs :
                          (param_idx == 1) ? beta_coeffs : gamma_coeffs;
    
    // Extract parameter using Chebyshev KAN
    float sum = 0.0f;
    for (int i = 0; i < feature_dim; ++i) {
        float x = features_batch[batch_idx * feature_dim + i];
        float z = tanhf(x);
        
        // Chebyshev polynomial evaluation
        float T0 = 1.0f;
        float T1 = z;
        float result = coeffs[i * (chebyshev_order + 1) + 0] * T0;
        if (chebyshev_order > 0) {
            result += coeffs[i * (chebyshev_order + 1) + 1] * T1;
        }
        
        for (int k = 2; k <= chebyshev_order; ++k) {
            float Tk = 2.0f * z * T1 - T0;
            result += coeffs[i * (chebyshev_order + 1) + k] * Tk;
            T0 = T1;
            T1 = Tk;
        }
        sum += result;
    }
    *output = sum;
}

// Launch wrappers
namespace GPU {
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
    ) {
        int total = batch_size * grid_size;
        dim3 block(BLOCK_SIZE);
        dim3 grid((total + block.x - 1) / block.x);
        compute_squeezed_coherent_kernel<<<grid, block, 0, stream>>>(
            alpha, beta, gamma, wavefunction_real, wavefunction_imag,
            batch_size, grid_size, L, sigma
        );
    }
    
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
    ) {
        dim3 block(3);  // One thread for alpha, beta, gamma
        dim3 grid(batch_size);
        batched_quantum_embedding_kernel<<<grid, block, 0, stream>>>(
            features_batch, alpha_coeffs, beta_coeffs, gamma_coeffs,
            alpha_out, beta_out, gamma_out,
            batch_size, feature_dim, chebyshev_order
        );
        
        // Check for errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            // Error handling in caller
        }
    }
}

#endif // USE_HIP

