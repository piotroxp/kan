#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// ROCm/HIP memory management
// Enable HIP for GPU acceleration
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

class ROCmMemoryManager {
public:
    ROCmMemoryManager() {
#ifdef USE_HIP
        // Initialize HIP runtime (hipInit can be called multiple times safely)
        hipError_t init_err = hipInit(0);
        // hipInit returns hipSuccess even if already initialized in some ROCm versions
        if (init_err != hipSuccess) {
            gpu_available_ = false;
            return;
        }
        
        // Get device count
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        
        if (err == hipSuccess && deviceCount > 0) {
            // Try to set device and verify it works
            hipError_t set_err = hipSetDevice(0);
            if (set_err == hipSuccess) {
                // Verify device properties can be retrieved
                hipDeviceProp_t prop;
                hipError_t prop_err = hipGetDeviceProperties(&prop, 0);
                if (prop_err == hipSuccess) {
                    gpu_available_ = true;
                    // Successfully initialized GPU
                } else {
                    gpu_available_ = false;
                }
            } else {
                gpu_available_ = false;
            }
        } else {
            // Device count is 0 or error occurred
            // This might happen if GPU is in low-power state
            gpu_available_ = false;
        }
#else
        gpu_available_ = false;
#endif
    }
    
    // Allocate device memory
    void* allocate(size_t bytes) {
#ifdef USE_HIP
        if (gpu_available_) {
            void* ptr = nullptr;
            hipError_t err = hipMalloc(&ptr, bytes);
            if (err != hipSuccess) {
                // Fallback to CPU
                return std::malloc(bytes);
            }
            return ptr;
        }
#endif
        // CPU fallback
        return std::malloc(bytes);
    }
    
    // Free device memory
    void free(void* ptr) {
        if (ptr) {
#ifdef USE_HIP
            if (gpu_available_) {
                hipError_t err = hipFree(ptr);
                if (err != hipSuccess) {
                    // Fallback to CPU free
                    std::free(ptr);
                    return;
                }
                return;
            }
#endif
            std::free(ptr);
        }
    }
    
    // Copy to device
    void copy_to_device(const void* host, void* device, size_t bytes) {
#ifdef USE_HIP
        if (gpu_available_) {
            hipError_t err = hipMemcpy(device, host, bytes, hipMemcpyHostToDevice);
            if (err != hipSuccess) {
                // Fallback to CPU
                std::memcpy(device, host, bytes);
            }
            return;
        }
#endif
        std::memcpy(device, host, bytes);
    }
    
    // Copy to host
    void copy_to_host(const void* device, void* host, size_t bytes) {
#ifdef USE_HIP
        if (gpu_available_) {
            hipError_t err = hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                // Fallback to CPU
                std::memcpy(host, device, bytes);
            }
            return;
        }
#endif
        std::memcpy(host, device, bytes);
    }
    
    // Synchronize device
    void synchronize() {
#ifdef USE_HIP
        if (gpu_available_) {
            hipError_t err = hipDeviceSynchronize();
            (void)err;  // Ignore errors
        }
#endif
    }
    
    // Check if GPU is available
    bool is_gpu_available() {
        return gpu_available_;
    }
    
private:
    bool gpu_available_;
};

// GPU tensor (wrapper for GPU memory)
class GPUTensor {
public:
    GPUTensor(size_t size) : size_(size), manager_() {
        data_ = manager_.allocate(size * sizeof(float));
    }
    
    ~GPUTensor() {
        if (data_) {
            manager_.free(data_);
        }
    }
    
    // Copy from host
    void from_host(const float* host_data) {
        manager_.copy_to_device(host_data, data_, size_ * sizeof(float));
    }
    
    // Copy to host
    void to_host(float* host_data) {
        manager_.copy_to_host(data_, host_data, size_ * sizeof(float));
    }
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    size_t size_;
    void* data_;
    ROCmMemoryManager manager_;
};
