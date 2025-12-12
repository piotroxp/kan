# GPU Acceleration Setup Guide

## Current Status

The training is currently running on **CPU** (as observed). To enable GPU acceleration on your AMD Radeon RX 7900 XTX, follow these steps:

## Step 1: Enable HIP in ROCm Manager

Edit `src/gpu/rocm_manager.hpp` and uncomment the HIP includes:

```cpp
// Change from:
// #include <hip/hip_runtime.h>

// To:
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
```

Then uncomment the GPU implementations in each function.

## Step 2: Update CMakeLists.txt for HIP

Add HIP support to CMakeLists.txt:

```cmake
# Find HIP
find_package(hip QUIET)

if(hip_FOUND)
    message(STATUS "HIP found - GPU acceleration enabled")
    target_link_libraries(kan_speech_model PUBLIC hip::device)
    target_compile_definitions(kan_speech_manager PRIVATE USE_HIP)
else()
    message(STATUS "HIP not found - using CPU fallback")
endif()
```

## Step 3: Implement GPU Kernels

The placeholder kernels in `src/gpu/hip_kernels.hpp` need to be implemented:

1. **KAN Layer Forward Kernel**: Parallel evaluation of KAN functions
2. **B-spline Evaluation Kernel**: GPU-accelerated B-spline computation
3. **Quantum Wavefunction Kernel**: Parallel wavefunction evaluation

## Step 4: Update Training Session

Modify `src/training/training_session.hpp` to use GPU tensors when available:

```cpp
if (gpu_available) {
    // Use GPUTensor for computations
    GPUTensor gpu_features = ...;
    // Run GPU kernels
} else {
    // CPU fallback
}
```

## Quick Test

After enabling HIP, rebuild and run:

```bash
cd build
cmake .. -DUSE_HIP=ON
cmake --build . -j$(nproc)
./train
```

You should see "GPU acceleration: ENABLED" in the output.

## Performance Expectations

With GPU acceleration:
- **Training speed**: 10-50x faster (depending on batch size)
- **Memory**: 24GB VRAM on 7900 XTX
- **Batch size**: Can increase to 64-128 with GPU

## Current Implementation

The current implementation uses **CPU fallback** mode, which is why training is on CPU. The GPU infrastructure is in place and ready - just needs HIP includes uncommented and kernels implemented.

---

*For now, training works on CPU. GPU acceleration can be enabled by following the steps above.*

