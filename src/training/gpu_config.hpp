#pragma once

#include "../gpu/rocm_manager.hpp"
#include <iostream>
#include <chrono>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

// GPU training configuration
class GPUTrainingConfig {
public:
    static void print_gpu_info() {
        std::cout << "=== GPU Information ===" << std::endl;
        
#ifdef USE_HIP
        // Try direct HIP call first
        hipError_t init_err = hipInit(0);
        int deviceCount = 0;
        hipError_t count_err = hipSuccess;
        
        if (init_err == hipSuccess) {
            count_err = hipGetDeviceCount(&deviceCount);
        }
        
        if (init_err == hipSuccess && count_err == hipSuccess && deviceCount > 0) {
            hipSetDevice(0);
            hipDeviceProp_t prop;
            hipError_t prop_err = hipGetDeviceProperties(&prop, 0);
            if (prop_err == hipSuccess) {
                std::cout << "GPU: Available" << std::endl;
                std::cout << "Device: " << prop.name << std::endl;
                std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
                std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
                std::cout << std::endl;
                return;
            }
        }
#endif
        
        // Fallback to manager check
        ROCmMemoryManager manager;
        if (manager.is_gpu_available()) {
            std::cout << "GPU: Available" << std::endl;
#ifdef USE_HIP
            hipDeviceProp_t prop;
            hipError_t err = hipGetDeviceProperties(&prop, 0);
            if (err == hipSuccess) {
                std::cout << "Device: " << prop.name << std::endl;
                std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
                std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            }
#endif
        } else {
            std::cout << "GPU: Not available (CPU mode)" << std::endl;
        }
        std::cout << std::endl;
    }
    
    static bool use_gpu() {
        ROCmMemoryManager manager;
        bool gpu_available = manager.is_gpu_available();
        
        if (gpu_available) {
            std::cout << "GPU acceleration: ENABLED" << std::endl;
#ifdef USE_HIP
            hipDeviceProp_t prop;
            hipError_t err = hipGetDeviceProperties(&prop, 0);
            if (err == hipSuccess) {
                std::cout << "  Device: " << prop.name << std::endl;
                std::cout << "  Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
            }
#endif
        } else {
            std::cout << "GPU acceleration: DISABLED (CPU fallback)" << std::endl;
        }
        
        return gpu_available;
    }
};

// Performance timer
class PerformanceTimer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time_);
        return duration.count();
    }
    
    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

