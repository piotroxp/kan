#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <atomic>

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
        initialize_memory_limits();
    }
    
    // Allocate device memory (enforces 90% GPU RAM limit)
    void* allocate(size_t bytes) {
#ifdef USE_HIP
        if (gpu_available_) {
            // Check memory limit before allocating (using static tracking)
            std::lock_guard<std::mutex> lock(memory_mutex_);
            size_t current_allocated = allocated_memory_.load();
            if (current_allocated + bytes > memory_limit_) {
                // Would exceed 90% limit, fallback to CPU
                std::cerr << "Warning: GPU memory limit (90%) would be exceeded. "
                          << "Requested: " << (bytes / (1024*1024)) << " MB, "
                          << "Available: " << ((memory_limit_ - current_allocated) / (1024*1024)) << " MB. "
                          << "Falling back to CPU." << std::endl;
                return std::malloc(bytes);
            }
            
            void* ptr = nullptr;
            hipError_t err = hipMalloc(&ptr, bytes);
            if (err != hipSuccess) {
                // Fallback to CPU
                return std::malloc(bytes);
            }
            
            // Track allocation (static, shared across all instances)
            allocated_memory_ += bytes;
            allocation_sizes_[ptr] = bytes;
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
                std::lock_guard<std::mutex> lock(memory_mutex_);
                // Look up allocation size and update tracking (static, shared)
                auto it = allocation_sizes_.find(ptr);
                if (it != allocation_sizes_.end()) {
                    allocated_memory_ -= it->second;
                    allocation_sizes_.erase(it);
                }
                
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
    
    // Get memory statistics
    size_t get_total_memory() const { 
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return total_memory_; 
    }
    size_t get_memory_limit() const { 
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return memory_limit_; 
    }
    size_t get_allocated_memory() const { 
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return allocated_memory_.load(); 
    }
    size_t get_available_memory() const { 
        std::lock_guard<std::mutex> lock(memory_mutex_);
        return memory_limit_ - allocated_memory_.load(); 
    }
    
private:
    void initialize_memory_limits() {
#ifdef USE_HIP
        if (gpu_available_) {
            // Use static initialization to ensure memory limits are set only once
            static std::once_flag init_flag;
            std::call_once(init_flag, []() {
                hipDeviceProp_t prop;
                hipError_t err = hipGetDeviceProperties(&prop, 0);
                if (err == hipSuccess) {
                    total_memory_ = prop.totalGlobalMem;
                    // Set limit to 90% of total GPU memory (10% reserved for system)
                    memory_limit_ = static_cast<size_t>(total_memory_ * 0.90);
                    allocated_memory_.store(0);
                    
                    std::cout << "GPU Memory Configuration:" << std::endl;
                    std::cout << "  Total GPU Memory: " << (total_memory_ / (1024*1024)) << " MB" << std::endl;
                    std::cout << "  Application Limit (90%): " << (memory_limit_ / (1024*1024)) << " MB" << std::endl;
                    std::cout << "  System Reserve (10%): " << ((total_memory_ - memory_limit_) / (1024*1024)) << " MB" << std::endl;
                }
            });
        }
#endif
    }
    
    bool detect_gpu() {
#ifdef USE_HIP
        // Force HIP initialization
        hipError_t init_err = hipInit(0);
        if (init_err != hipSuccess) {
            return false;
        }
        // Get device count (retry once if needed)
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        
        if (err != hipSuccess || deviceCount == 0) {
            // Retry once after brief delay
            usleep(50000);  // 50ms
            err = hipGetDeviceCount(&deviceCount);
            if (err != hipSuccess || deviceCount == 0) {
                return false;
            }
        }
        
        // Set device
        hipError_t set_err = hipSetDevice(0);
        if (set_err != hipSuccess) {
            return false;
        }
        
        // Verify device properties
        hipDeviceProp_t prop;
        hipError_t prop_err = hipGetDeviceProperties(&prop, 0);
        if (prop_err == hipSuccess) {
            return true;
        }
        return false;
#else
        return false;
#endif
    }
    
    bool gpu_available_;
    
    // Static members for global memory tracking across all instances
    static size_t total_memory_;
    static size_t memory_limit_;
    static std::atomic<size_t> allocated_memory_;
    static std::mutex memory_mutex_;
    static std::unordered_map<void*, size_t> allocation_sizes_;
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
