#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <iostream>

// ROCm/HIP memory management
// Enable HIP for GPU acceleration
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

class ROCmMemoryManager {
public:
    ROCmMemoryManager() {
        gpu_available_ = detect_gpu();
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
    bool is_gpu_available() const {
        return gpu_available_;
    }
    
private:
    bool detect_gpu() {
#ifdef USE_HIP
        // Force HIP initialization
        hipError_t init_err = hipInit(0);
        if (init_err != hipSuccess) {
            std::cerr << "[GPU DEBUG] hipInit failed: " << init_err << std::endl;
            return false;
        }
        std::cerr << "[GPU DEBUG] hipInit succeeded" << std::endl;
        
        // Get device count (retry once if needed)
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        std::cerr << "[GPU DEBUG] hipGetDeviceCount: err=" << err << " count=" << deviceCount << std::endl;
        
        if (err != hipSuccess || deviceCount == 0) {
            // Retry once after brief delay
            std::cerr << "[GPU DEBUG] Retrying after 50ms delay..." << std::endl;
            usleep(50000);  // 50ms
            err = hipGetDeviceCount(&deviceCount);
            std::cerr << "[GPU DEBUG] Retry hipGetDeviceCount: err=" << err << " count=" << deviceCount << std::endl;
            if (err != hipSuccess || deviceCount == 0) {
                std::cerr << "[GPU DEBUG] GPU detection failed: no devices" << std::endl;
                return false;
            }
        }
        
        // Set device
        hipError_t set_err = hipSetDevice(0);
        std::cerr << "[GPU DEBUG] hipSetDevice(0): " << set_err << std::endl;
        if (set_err != hipSuccess) {
            std::cerr << "[GPU DEBUG] hipSetDevice failed" << std::endl;
            return false;
        }
        
        // Verify device properties
        hipDeviceProp_t prop;
        hipError_t prop_err = hipGetDeviceProperties(&prop, 0);
        std::cerr << "[GPU DEBUG] hipGetDeviceProperties: " << prop_err << std::endl;
        if (prop_err == hipSuccess) {
            std::cerr << "[GPU DEBUG] GPU detected: " << prop.name << " (" << (prop.totalGlobalMem / (1024*1024)) << " MB)" << std::endl;
            return true;
        }
        std::cerr << "[GPU DEBUG] hipGetDeviceProperties failed" << std::endl;
        return false;
#else
        std::cerr << "[GPU DEBUG] USE_HIP not defined" << std::endl;
        return false;
#endif
    }
    
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
