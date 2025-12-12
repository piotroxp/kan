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
        // Initialize HIP
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        gpu_available_ = (err == hipSuccess && deviceCount > 0);
        
        if (gpu_available_) {
            hipSetDevice(0);  // Use first device
        } else {
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
            hipDeviceSynchronize();
        }
#endif
    }
    
    // Check if GPU is available
    bool is_gpu_available() const {
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
