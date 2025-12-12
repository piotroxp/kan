// HIP kernels for KAN layer operations
// Optimized for AMD Radeon RX 7900 XTX (RDNA 3 architecture, gfx1100)
// 96 Compute Units, 2 SIMDs per CU, 64 threads per wavefront

#ifdef USE_HIP
#include "hip_kernels.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// B-spline basis function evaluation (iterative for better performance)
__device__ inline float bspline_basis_iterative(float t, int k, int degree) {
    if (degree == 0) {
        return (t >= k && t < k + 1) ? 1.0f : 0.0f;
    }
    
    // Iterative evaluation for better performance
    float basis[4] = {0.0f};  // Support up to degree 3
    basis[0] = (t >= k && t < k + 1) ? 1.0f : 0.0f;
    
    for (int d = 1; d <= degree; ++d) {
        float w1 = (t - k) / d;
        float w2 = (k + d + 1 - t) / d;
        basis[d] = w1 * basis[d-1] + w2 * ((k+1 < k+d+1) ? basis[d-1] : 0.0f);
    }
    
    return basis[degree];
}

// Simplified B-spline for cubic (degree 3)
__device__ inline float bspline_cubic(float t, int k) {
    float u = t - k;
    if (u < 0.0f || u >= 1.0f) return 0.0f;
    
    float u2 = u * u;
    float u3 = u2 * u;
    
    // Cubic B-spline basis
    if (u < 1.0f) {
        return (1.0f/6.0f) * (u3);
    }
    return 0.0f;
}

// Chebyshev polynomial evaluation (optimized iterative)
__device__ inline float chebyshev_T_fast(int n, float x) {
    if (n == 0) return 1.0f;
    if (n == 1) return x;
    
    float T0 = 1.0f;
    float T1 = x;
    
    for (int i = 2; i <= n; ++i) {
        float Tn = 2.0f * x * T1 - T0;
        T0 = T1;
        T1 = Tn;
    }
    
    return T1;
}

// Sinc function with fast math
__device__ inline float sinc_fast(float x) {
    float abs_x = fabsf(x);
    if (abs_x < 1e-6f) return 1.0f;
    float pi_x = M_PI * x;
    return __sinf(pi_x) / pi_x;
}

// Gaussian RBF with fast exp
__device__ inline float gaussian_rbf_fast(float x, float center, float sigma) {
    float diff = x - center;
    float sigma2 = sigma * sigma;
    return __expf(-0.5f * diff * diff / sigma2);
}

// ============================================================================
// Kernel Implementations
// ============================================================================

// B-spline KAN forward kernel
__global__ void bspline_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int grid_size,
    int spline_degree
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_out) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < n_in; ++i) {
        float x_val = x_in[i];
        
        // Clamp x_val to grid range [0, grid_size-1]
        x_val = fmaxf(0.0f, fminf(x_val, (float)(grid_size - 1)));
        
        // Find grid interval
        int k = (int)floorf(x_val);
        k = max(0, min(k, grid_size - spline_degree - 2));
        
        // Evaluate B-spline (cubic for performance)
        float phi_val = 0.0f;
        if (spline_degree == 3) {
            // Cubic B-spline: use 4 basis functions
            for (int d = 0; d <= 3; ++d) {
                int idx = (j * n_in + i) * grid_size + k + d;
                if (idx < n_in * n_out * grid_size) {
                    float basis = bspline_cubic(x_val, k + d);
                    phi_val += coeffs[idx] * basis;
                }
            }
        } else {
            // General degree
            for (int d = 0; d <= spline_degree; ++d) {
                int idx = (j * n_in + i) * grid_size + k + d;
                if (idx < n_in * n_out * grid_size) {
                    float basis = bspline_basis_iterative(x_val, k, spline_degree);
                    phi_val += coeffs[idx] * basis;
                }
            }
        }
        
        sum += phi_val;
    }
    
    x_out[j] = sum;
}

// Chebyshev KAN forward kernel
__global__ void chebyshev_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int chebyshev_order
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_out) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < n_in; ++i) {
        float x_val = x_in[i];
        // Normalize to [-1, 1] range
        x_val = fmaxf(-1.0f, fminf(1.0f, x_val));
        
        float phi_val = 0.0f;
        for (int k = 0; k <= chebyshev_order; ++k) {
            int idx = (j * n_in + i) * (chebyshev_order + 1) + k;
            float Tk = chebyshev_T_fast(k, x_val);
            phi_val += coeffs[idx] * Tk;
        }
        
        sum += phi_val;
    }
    
    x_out[j] = sum;
}

// Sinc KAN forward kernel
__global__ void sinc_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int grid_size,
    float bandwidth
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_out) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < n_in; ++i) {
        float x_val = x_in[i];
        
        float phi_val = 0.0f;
        for (int k = 0; k < grid_size; ++k) {
            float grid_point = (float)k / (float)grid_size;
            float diff = (x_val - grid_point) * bandwidth;
            float sinc_val = sinc_fast(diff);
            
            int idx = (j * n_in + i) * grid_size + k;
            phi_val += coeffs[idx] * sinc_val;
        }
        
        sum += phi_val;
    }
    
    x_out[j] = sum;
}

// Fourier KAN forward kernel
__global__ void fourier_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs_real,
    const float* __restrict__ coeffs_imag,
    int n_in,
    int n_out,
    int num_modes
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_out) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < n_in; ++i) {
        float x_val = x_in[i];
        
        // DC component
        int dc_idx = (j * n_in + i) * (2 * num_modes + 1);
        float phi_val = coeffs_real[dc_idx];
        
        // Fourier modes
        for (int m = 1; m <= num_modes; ++m) {
            int real_idx = dc_idx + 2 * m - 1;
            int imag_idx = dc_idx + 2 * m;
            
            float angle = 2.0f * M_PI * m * x_val;
            float cos_val = __cosf(angle);
            float sin_val = __sinf(angle);
            
            phi_val += coeffs_real[real_idx] * cos_val - coeffs_imag[imag_idx] * sin_val;
        }
        
        sum += phi_val;
    }
    
    x_out[j] = sum;
}

// RBF KAN forward kernel
__global__ void rbf_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    const float* __restrict__ centers,
    int n_in,
    int n_out,
    int num_centers,
    float sigma
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_out) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < n_in; ++i) {
        float x_val = x_in[i];
        
        float phi_val = 0.0f;
        for (int c = 0; c < num_centers; ++c) {
            float center = centers[c];
            float rbf_val = gaussian_rbf_fast(x_val, center, sigma);
            
            int idx = (j * n_in + i) * num_centers + c;
            phi_val += coeffs[idx] * rbf_val;
        }
        
        sum += phi_val;
    }
    
    x_out[j] = sum;
}

// Piecewise Linear KAN forward kernel
__global__ void piecewise_linear_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int grid_size
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_out) return;
    
    float sum = 0.0f;
    
    for (int i = 0; i < n_in; ++i) {
        float x_val = x_in[i];
        
        // Clamp to [0, 1] and scale to grid
        x_val = fmaxf(0.0f, fminf(1.0f, x_val));
        float scaled = x_val * (grid_size - 1);
        
        // Linear interpolation
        int k = (int)floorf(scaled);
        k = max(0, min(k, grid_size - 2));
        float t = scaled - k;
        
        int idx1 = (j * n_in + i) * grid_size + k;
        int idx2 = idx1 + 1;
        
        float phi_val = coeffs[idx1] * (1.0f - t) + coeffs[idx2] * t;
        sum += phi_val;
    }
    
    x_out[j] = sum;
}

// ============================================================================
// Kernel Launch Wrappers (implementations)
// ============================================================================

namespace GPU {
    // Launch B-spline KAN kernel
    void launch_bspline_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size, int spline_degree,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        bspline_kan_forward_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs, n_in, n_out, grid_size, spline_degree
        );
    }
    
    // Launch Chebyshev KAN kernel
    void launch_chebyshev_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int chebyshev_order,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        chebyshev_kan_forward_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs, n_in, n_out, chebyshev_order
        );
    }
    
    // Launch Sinc KAN kernel
    void launch_sinc_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size, float bandwidth,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        sinc_kan_forward_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs, n_in, n_out, grid_size, bandwidth
        );
    }
    
    // Launch Fourier KAN kernel
    void launch_fourier_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs_real,
        const float* coeffs_imag,
        int n_in, int n_out, int num_modes,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        fourier_kan_forward_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs_real, coeffs_imag, n_in, n_out, num_modes
        );
    }
    
    // Launch RBF KAN kernel
    void launch_rbf_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        const float* centers,
        int n_in, int n_out, int num_centers, float sigma,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        rbf_kan_forward_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs, centers, n_in, n_out, num_centers, sigma
        );
    }
    
    // Launch Piecewise Linear KAN kernel
    void launch_piecewise_linear_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        piecewise_linear_kan_forward_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs, n_in, n_out, grid_size
        );
    }
}

#endif // USE_HIP
