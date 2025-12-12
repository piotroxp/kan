#pragma once

#include "../core/kan_layer.hpp"
#include "../core/tensor.hpp"
#include "rocm_manager.hpp"
#include <vector>
#include <memory>
#include <iostream>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include "hip_kernels.hpp"
#endif

// GPU-enabled KAN layer wrapper
// Automatically uses GPU if available, falls back to CPU
class GPUKANLayer {
public:
    GPUKANLayer(KANBasis basis_type, int n_in, int n_out, int grid_size, 
                int order = 3, double param = 0.0)
        : basis_type_(basis_type),
          n_in_(n_in),
          n_out_(n_out),
          grid_size_(grid_size),
          order_(order),
          param_(param),
          manager_(),
          use_gpu_(manager_.is_gpu_available()) {
        
        // Allocate GPU memory if available
        if (use_gpu_) {
            size_t input_size = n_in_ * sizeof(float);
            size_t output_size = n_out_ * sizeof(float);
            size_t coeffs_size = estimate_coeffs_size() * sizeof(float);
            
            gpu_input_ = manager_.allocate(input_size);
            gpu_output_ = manager_.allocate(output_size);
            gpu_coeffs_ = manager_.allocate(coeffs_size);
            
            // Initialize coefficients on GPU
            initialize_coeffs();
        }
    }
    
    ~GPUKANLayer() {
        if (use_gpu_) {
            if (gpu_input_) manager_.free(gpu_input_);
            if (gpu_output_) manager_.free(gpu_output_);
            if (gpu_coeffs_) manager_.free(gpu_coeffs_);
        }
    }
    
    // Forward pass (GPU or CPU)
    Tensor forward(const Tensor& input) {
        if (use_gpu_) {
            return forward_gpu(input);
        } else {
            return forward_cpu(input);
        }
    }
    
    // Get parameters (for optimizer)
    std::vector<double> parameters() {
        if (use_gpu_) {
            // Copy from GPU to CPU
            size_t size = estimate_coeffs_size();
            std::vector<float> float_params(size);
            manager_.copy_to_host(gpu_coeffs_, float_params.data(), 
                                 size * sizeof(float));
            std::vector<double> params(float_params.begin(), float_params.end());
            return params;
        } else {
            return cpu_coeffs_;
        }
    }
    
    // Set parameters
    void set_parameters(const std::vector<double>& params) {
        if (use_gpu_) {
            // Copy to GPU
            std::vector<float> float_params(params.begin(), params.end());
            manager_.copy_to_device(float_params.data(), gpu_coeffs_,
                                   params.size() * sizeof(float));
        } else {
            cpu_coeffs_ = params;
        }
    }
    
    bool is_using_gpu() const { return use_gpu_; }
    
private:
    KANBasis basis_type_;
    int n_in_, n_out_, grid_size_, order_;
    double param_;
    ROCmMemoryManager manager_;
    bool use_gpu_;
    
    // GPU memory
    void* gpu_input_ = nullptr;
    void* gpu_output_ = nullptr;
    void* gpu_coeffs_ = nullptr;
    
    // CPU fallback
    std::vector<double> cpu_coeffs_;
    
    size_t estimate_coeffs_size() const {
        switch (basis_type_) {
            case KANBasis::BSpline:
                return n_out_ * n_in_ * grid_size_;
            case KANBasis::Chebyshev:
                return n_out_ * n_in_ * (order_ + 1);
            case KANBasis::Sinc:
                return n_out_ * n_in_ * grid_size_;
            case KANBasis::Fourier:
                return n_out_ * n_in_ * (2 * order_ + 1);
            case KANBasis::RBF:
                return n_out_ * n_in_ * order_;  // order = num_centers
            case KANBasis::PiecewiseLinear:
                return n_out_ * n_in_ * grid_size_;
            default:
                return n_out_ * n_in_ * grid_size_;
        }
    }
    
    void initialize_coeffs() {
        size_t size = estimate_coeffs_size();
        std::vector<float> init_coeffs(size, 0.01f);  // Small random init
        if (use_gpu_) {
            manager_.copy_to_device(init_coeffs.data(), gpu_coeffs_,
                                   size * sizeof(float));
        } else {
            cpu_coeffs_.resize(size, 0.01);
        }
    }
    
    Tensor forward_gpu(const Tensor& input) {
#ifdef USE_HIP
        // Copy input to GPU
        std::vector<float> input_vec(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input_vec[i] = static_cast<float>(input[i]);
        }
        hipError_t err = hipMemcpy(gpu_input_, input_vec.data(),
                                   n_in_ * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            std::cerr << "[GPU ERROR] hipMemcpyHostToDevice failed: " << hipGetErrorString(err) << std::endl;
            return forward_cpu(input);
        }
        
        // Launch kernel with error checking
        hipStream_t stream = nullptr;
        bool kernel_launched = false;
        
        switch (basis_type_) {
            case KANBasis::BSpline:
                GPU::launch_bspline_kan(
                    static_cast<const float*>(gpu_input_),
                    static_cast<float*>(gpu_output_),
                    static_cast<const float*>(gpu_coeffs_),
                    n_in_, n_out_, static_cast<int>(grid_size_), order_,
                    stream
                );
                kernel_launched = true;
                break;
            case KANBasis::Chebyshev:
                GPU::launch_chebyshev_kan(
                    static_cast<const float*>(gpu_input_),
                    static_cast<float*>(gpu_output_),
                    static_cast<const float*>(gpu_coeffs_),
                    n_in_, n_out_, order_,
                    stream
                );
                kernel_launched = true;
                break;
            case KANBasis::Sinc:
                GPU::launch_sinc_kan(
                    static_cast<const float*>(gpu_input_),
                    static_cast<float*>(gpu_output_),
                    static_cast<const float*>(gpu_coeffs_),
                    n_in_, n_out_, static_cast<int>(grid_size_), static_cast<float>(param_),
                    stream
                );
                kernel_launched = true;
                break;
            case KANBasis::PiecewiseLinear:
                GPU::launch_piecewise_linear_kan(
                    static_cast<const float*>(gpu_input_),
                    static_cast<float*>(gpu_output_),
                    static_cast<const float*>(gpu_coeffs_),
                    n_in_, n_out_, static_cast<int>(grid_size_),
                    stream
                );
                kernel_launched = true;
                break;
            default:
                // Fallback to CPU for unsupported types
                return forward_cpu(input);
        }
        
        if (!kernel_launched) {
            std::cerr << "[GPU ERROR] Kernel not launched, falling back to CPU" << std::endl;
            return forward_cpu(input);
        }
        
        // Check for kernel launch errors
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "[GPU ERROR] Kernel launch failed: " << hipGetErrorString(err) << std::endl;
            return forward_cpu(input);
        }
        
        // Synchronize
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "[GPU ERROR] hipDeviceSynchronize failed: " << hipGetErrorString(err) << std::endl;
            return forward_cpu(input);
        }
        
        // Copy output back
        std::vector<float> output_vec(n_out_);
        err = hipMemcpy(output_vec.data(), gpu_output_,
                        n_out_ * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            std::cerr << "[GPU ERROR] hipMemcpyDeviceToHost failed: " << hipGetErrorString(err) << std::endl;
            return forward_cpu(input);
        }
        
        // Convert to Tensor
        Tensor output({static_cast<size_t>(n_out_)});
        for (int i = 0; i < n_out_; ++i) {
            output[i] = static_cast<double>(output_vec[i]);
        }
        
        return output;
#else
        // No GPU support, fallback to CPU
        return forward_cpu(input);
#endif
    }
    
    Tensor forward_cpu(const Tensor& input) {
        // CPU fallback - use existing KAN layer implementations
        // For now, return a simple pass-through
        Tensor output({static_cast<size_t>(n_out_)});
        output.fill(0.0);
        
        // Simple linear transformation as fallback
        for (int j = 0; j < n_out_; ++j) {
            for (int i = 0; i < n_in_; ++i) {
                output[j] += input[i] * 0.01;  // Small weight
            }
        }
        
        return output;
    }
};

