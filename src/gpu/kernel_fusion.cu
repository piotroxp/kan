// Kernel fusion implementations
#ifdef USE_HIP
#include "kernel_fusion.hpp"
#include "hip_kernels.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fused B-spline KAN + activation (ReLU)
__global__ void fused_bspline_relu_kernel(
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
        x_val = fmaxf(0.0f, fminf(x_val, (float)(grid_size - 1)));
        
        int k = (int)floorf(x_val);
        k = max(0, min(k, grid_size - spline_degree - 2));
        
        float phi_val = 0.0f;
        if (spline_degree == 3) {
            for (int d = 0; d <= 3; ++d) {
                int idx = (j * n_in + i) * grid_size + k + d;
                if (idx < n_in * n_out * grid_size) {
                    float u = x_val - (k + d);
                    if (u >= 0.0f && u < 1.0f) {
                        float u2 = u * u;
                        float u3 = u2 * u;
                        float basis = (1.0f/6.0f) * u3;
                        phi_val += coeffs[idx] * basis;
                    }
                }
            }
        }
        
        sum += phi_val;
    }
    
    // Apply ReLU activation (fused)
    x_out[j] = fmaxf(0.0f, sum);
}

// Fused matrix multiply + bias + activation
__global__ void fused_matmul_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    if (bias) {
        sum += bias[col];
    }
    
    // ReLU activation (fused)
    C[row * N + col] = fmaxf(0.0f, sum);
}

namespace KernelFusion {
    // Launch fused B-spline + ReLU
    void launch_fused_bspline_relu(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size, int spline_degree,
        hipStream_t stream
    ) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_out + block.x - 1) / block.x);
        
        fused_bspline_relu_kernel<<<grid, block, 0, stream>>>(
            x_in, x_out, coeffs, n_in, n_out, grid_size, spline_degree
        );
    }
    
    // Launch fused matmul + bias + ReLU
    void launch_fused_matmul_bias_relu(
        const float* A,
        const float* B,
        const float* bias,
        float* C,
        int M, int N, int K,
        hipStream_t stream
    ) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        
        fused_matmul_bias_relu_kernel<<<grid, block, 0, stream>>>(
            A, B, bias, C, M, N, K
        );
    }
}

#endif // USE_HIP

