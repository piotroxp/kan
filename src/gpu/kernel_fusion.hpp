#pragma once

#include "../core/tensor.hpp"
#include <vector>

#ifdef USE_HIP
#include <hip/hip_runtime.h>

// Kernel fusion for optimized inference
// Fuses multiple KAN operations into single GPU kernel
namespace KernelFusion {
    
    // Kernel declarations (implementations in .cu file)
    __global__ void fused_bspline_relu_kernel(
        const float* __restrict__ x_in,
        float* __restrict__ x_out,
        const float* __restrict__ coeffs,
        int n_in,
        int n_out,
        int grid_size,
        int spline_degree
    );
    
    __global__ void fused_matmul_bias_relu_kernel(
        const float* __restrict__ A,
        const float* __restrict__ B,
        const float* __restrict__ bias,
        float* __restrict__ C,
        int M, int N, int K
    );
    
    // Launch fused B-spline + ReLU
    void launch_fused_bspline_relu(
        const float* x_in,
        float* x_out,
        const float* coeffs,
        int n_in, int n_out, int grid_size, int spline_degree,
        hipStream_t stream = nullptr
    );
    
    // Launch fused matmul + bias + ReLU
    void launch_fused_matmul_bias_relu(
        const float* A,
        const float* B,
        const float* bias,
        float* C,
        int M, int N, int K,
        hipStream_t stream = nullptr
    );
}

#endif // USE_HIP
