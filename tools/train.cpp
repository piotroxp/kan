#include <iostream>
#include "training/training_session.hpp"
#include "training/gpu_config.hpp"
#include "model/speech_model.hpp"

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

int main(int argc, char* argv[]) {
    std::cout << "=== KAN Speech Model Training ===" << std::endl;
    std::cout << std::endl;
    
    // Force HIP initialization early
#ifdef USE_HIP
    std::cout << "[MAIN DEBUG] USE_HIP is defined" << std::endl;
    std::cout << "[MAIN DEBUG] Initializing HIP..." << std::endl;
    hipError_t init = hipInit(0);
    std::cout << "[MAIN DEBUG] hipInit result: " << init << " (0=success)" << std::endl;
    
    int count = 0;
    hipError_t count_err = hipGetDeviceCount(&count);
    std::cout << "[MAIN DEBUG] Device count: " << count << " (err: " << count_err << ")" << std::endl;
    
    if (count > 0) {
        hipDeviceProp_t prop;
        hipError_t prop_err = hipGetDeviceProperties(&prop, 0);
        std::cout << "[MAIN DEBUG] Device properties: " << prop_err << std::endl;
        if (prop_err == 0) {
            std::cout << "[MAIN DEBUG] GPU: " << prop.name << std::endl;
        }
    }
#else
    std::cout << "[MAIN DEBUG] USE_HIP is NOT defined" << std::endl;
#endif
    
    // Print GPU info
    GPUTrainingConfig::print_gpu_info();
    bool use_gpu = GPUTrainingConfig::use_gpu();
    (void)use_gpu;  // Suppress unused warning for now
    
    // Parse arguments (simplified)
    size_t batch_size = 32;
    double learning_rate = 1e-4;
    size_t num_epochs = 10;
    std::string checkpoint_dir = "checkpoints";
    
    if (argc > 1) {
        batch_size = std::stoull(argv[1]);
    }
    if (argc > 2) {
        learning_rate = std::stod(argv[2]);
    }
    if (argc > 3) {
        num_epochs = std::stoull(argv[3]);
    }
    if (argc > 4) {
        checkpoint_dir = argv[4];
    }
    
    // Create training session
    TrainingSession session(checkpoint_dir, batch_size, learning_rate, num_epochs);
    
    // Train
    session.train();
    
    std::cout << std::endl;
    std::cout << "Training completed successfully!" << std::endl;
    std::cout << "Checkpoints saved to: " << checkpoint_dir << std::endl;
    
    return 0;
}

