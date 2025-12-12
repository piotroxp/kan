// HIP kernels for KAN layer operations
// Optimized for AMD Radeon RX 7900 XTX (RDNA 3 architecture, gfx1100)
// 96 Compute Units, 2 SIMDs per CU, 64 threads per wavefront

#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Block size optimized for RDNA 3 (64 threads per wavefront, 256 threads per block recommended)
#define BLOCK_SIZE 256
#define WARP_SIZE 64

// ============================================================================
// Device Function Declarations
// ============================================================================

// B-spline basis function declarations (implemented in .cu file)
__device__ float bspline_basis_iterative(float t, int k, int degree);
__device__ float bspline_cubic(float t, int k);

// Chebyshev polynomial evaluation (declared in .cu)
__device__ float chebyshev_T_fast(int n, float x);

// Sinc function (declared in .cu)
__device__ float sinc_fast(float x);

// Gaussian RBF (declared in .cu)
__device__ float gaussian_rbf_fast(float x, float center, float sigma);

// ============================================================================
// Kernel Declarations (implementations in .cu file)
// ============================================================================

__global__ void bspline_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int grid_size,
    int spline_degree
);

__global__ void chebyshev_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int chebyshev_order
);

__global__ void sinc_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int grid_size,
    float bandwidth
);

__global__ void fourier_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs_real,
    const float* __restrict__ coeffs_imag,
    int n_in,
    int n_out,
    int num_modes
);

__global__ void rbf_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    const float* __restrict__ centers,
    int n_in,
    int n_out,
    int num_centers,
    float sigma
);

__global__ void piecewise_linear_kan_forward_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ coeffs,
    int n_in,
    int n_out,
    int grid_size
);

// ============================================================================
// Kernel Launch Wrapper Declarations
// ============================================================================

namespace GPU {
    // Launch B-spline KAN kernel
    void launch_bspline_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size, int spline_degree,
        hipStream_t stream = nullptr
    );
    
    // Launch Chebyshev KAN kernel
    void launch_chebyshev_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int chebyshev_order,
        hipStream_t stream = nullptr
    );
    
    // Launch Sinc KAN kernel
    void launch_sinc_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size, float bandwidth,
        hipStream_t stream = nullptr
    );
    
    // Launch Fourier KAN kernel
    void launch_fourier_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs_real,
        const float* coeffs_imag,
        int n_in, int n_out, int num_modes,
        hipStream_t stream = nullptr
    );
    
    // Launch RBF KAN kernel
    void launch_rbf_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        const float* centers,
        int n_in, int n_out, int num_centers, float sigma,
        hipStream_t stream = nullptr
    );
    
    // Launch Piecewise Linear KAN kernel
    void launch_piecewise_linear_kan(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size,
        hipStream_t stream = nullptr
    );
}

#endif // USE_HIP
