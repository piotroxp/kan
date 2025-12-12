#pragma once

#ifdef USE_HIP
#include <rocfft/rocfft.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>

// Wrapper for rocFFT to simplify FFT operations
class ROCFFTWrapper {
public:
    ROCFFTWrapper(size_t n_fft, size_t batch_size = 1) 
        : n_fft_(n_fft), batch_size_(batch_size), plan_created_(false) {
        
        rocfft_status status = rocfft_setup();
        if (status != rocfft_status_success) {
            throw std::runtime_error("rocFFT setup failed");
        }
        
        // Create FFT plan
        rocfft_plan_description desc = nullptr;
        status = rocfft_plan_description_create(&desc);
        if (status != rocfft_status_success) {
            rocfft_cleanup();
            throw std::runtime_error("rocFFT plan description creation failed");
        }
        
        size_t lengths[1] = {n_fft};
        rocfft_plan plan = nullptr;
        status = rocfft_plan_create(&plan,
                                    rocfft_placement_inplace,
                                    rocfft_transform_type_complex_forward,
                                    rocfft_precision_single,
                                    1,  // dimensions
                                    lengths,
                                    batch_size,
                                    nullptr);
        
        if (status != rocfft_status_success) {
            rocfft_plan_description_destroy(desc);
            rocfft_cleanup();
            throw std::runtime_error("rocFFT plan creation failed");
        }
        
        plan_ = plan;
        desc_ = desc;
        plan_created_ = true;
    }
    
    ~ROCFFTWrapper() {
        if (plan_created_) {
            if (plan_) rocfft_plan_destroy(plan_);
            if (desc_) rocfft_plan_description_destroy(desc_);
            rocfft_cleanup();
        }
    }
    
    // Execute FFT (in-place)
    void execute(void* data, hipStream_t stream = nullptr) {
        if (!plan_created_ || !plan_) {
            throw std::runtime_error("FFT plan not created");
        }
        
        rocfft_execution_info info = nullptr;
        rocfft_execution_info_create(&info);
        rocfft_execution_info_set_stream(info, stream);
        
        rocfft_status status = rocfft_execute(plan_, 
                                             (void**)&data, 
                                             nullptr, 
                                             info);
        
        rocfft_execution_info_destroy(info);
        
        if (status != rocfft_status_success) {
            throw std::runtime_error("rocFFT execution failed");
        }
    }
    
    size_t n_fft() const { return n_fft_; }
    size_t batch_size() const { return batch_size_; }
    
private:
    size_t n_fft_;
    size_t batch_size_;
    rocfft_plan plan_;
    rocfft_plan_description desc_;
    bool plan_created_;
};

#endif // USE_HIP



