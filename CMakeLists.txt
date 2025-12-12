cmake_minimum_required(VERSION 3.20)
project(kan-speech-model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include Conan dependencies
if(EXISTS "${CMAKE_SOURCE_DIR}/conan_toolchain.cmake")
    include(${CMAKE_SOURCE_DIR}/conan_toolchain.cmake)
elseif(EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
    include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
endif()

# Find ROCm/HIP
find_path(HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    PATHS /opt/rocm/include
    NO_DEFAULT_PATH
)

if(HIP_INCLUDE_DIR)
    message(STATUS "HIP found at: ${HIP_INCLUDE_DIR}")
    include_directories(${HIP_INCLUDE_DIR})
    
    # Find HIP libraries
    find_library(HIP_LIBRARY
        NAMES amdhip64 hip_hcc
        PATHS /opt/rocm/lib /opt/rocm/lib64
        NO_DEFAULT_PATH
    )
    
    if(HIP_LIBRARY)
        message(STATUS "HIP library found: ${HIP_LIBRARY}")
    else()
        message(STATUS "HIP library not found - will compile without GPU support")
    endif()
else()
    message(STATUS "HIP not found - using CPU fallback")
endif()

# Optional packages for tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    find_package(Catch2 REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})

# Source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/kan_layer.cpp
    src/core/bspline_kan.cpp
    src/core/chebyshev_kan.cpp
    src/core/sinc_kan.cpp
    src/core/fourier_kan.cpp
    src/core/rbf_kan.cpp
    src/core/piecewise_linear_kan.cpp
)

set(QUANTUM_SOURCES
    src/quantum/wavefunction.cpp
    src/quantum/quantum_field_embedding.cpp
)

set(AUDIO_SOURCES
    src/audio/audio_buffer.cpp
    src/audio/feature_extraction.cpp
    src/audio/preprocessing.cpp
    src/audio/wav_reader.cpp
)

set(TRAINING_SOURCES
    src/training/loss.cpp
    src/training/optimizer.cpp
    src/training/trainer.cpp
    src/training/gradients.cpp
    src/training/training_session.cpp
    src/training/gpu_config.cpp
    src/training/backprop.cpp
    src/training/evaluation.cpp
    src/training/checkpoint_manager.cpp
)

set(MODEL_SOURCES
    src/model/speech_model.cpp
    src/model/language_model.cpp
)

set(GPU_SOURCES
    src/gpu/rocm_manager.cpp
    src/gpu/hip_kernels.cpp
    src/gpu/gpu_kan_layer.cpp
    src/gpu/kernel_fusion.cpp
)

# HIP kernel sources (compiled with hipcc)
if(HIP_LIBRARY)
    # Find hipcc
    find_program(HIPCC_EXECUTABLE
        NAMES hipcc
        PATHS /opt/rocm/bin
        NO_DEFAULT_PATH
    )
    
    if(HIPCC_EXECUTABLE)
        message(STATUS "Found hipcc: ${HIPCC_EXECUTABLE}")
        
        # Compile .cu files with hipcc as custom commands
        set(HIP_KERNEL_OBJ ${CMAKE_BINARY_DIR}/CMakeFiles/kan_speech_model.dir/src/gpu/hip_kernels.cu.o)
        set(HIP_FUSION_OBJ ${CMAKE_BINARY_DIR}/CMakeFiles/kan_speech_model.dir/src/gpu/kernel_fusion.cu.o)
        set(HIP_FEATURE_OBJ ${CMAKE_BINARY_DIR}/CMakeFiles/kan_speech_model.dir/src/gpu/feature_kernels.cu.o)
        set(HIP_QUANTUM_OBJ ${CMAKE_BINARY_DIR}/CMakeFiles/kan_speech_model.dir/src/gpu/quantum_kernels.cu.o)
        
        add_custom_command(
            OUTPUT ${HIP_KERNEL_OBJ}
            COMMAND ${HIPCC_EXECUTABLE}
                -DUSE_HIP
                -D__HIP_PLATFORM_AMD__
                -I${HIP_INCLUDE_DIR}
                -I${CMAKE_SOURCE_DIR}/src
                -std=c++17
                -O3
                -c
                ${CMAKE_SOURCE_DIR}/src/gpu/hip_kernels.cu
                -o ${HIP_KERNEL_OBJ}
            DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/hip_kernels.cu
                    ${CMAKE_SOURCE_DIR}/src/gpu/hip_kernels.hpp
            COMMENT "Compiling HIP kernels with hipcc"
        )
        
        add_custom_command(
            OUTPUT ${HIP_FUSION_OBJ}
            COMMAND ${HIPCC_EXECUTABLE}
                -DUSE_HIP
                -D__HIP_PLATFORM_AMD__
                -I${HIP_INCLUDE_DIR}
                -I${CMAKE_SOURCE_DIR}/src
                -std=c++17
                -O3
                -c
                ${CMAKE_SOURCE_DIR}/src/gpu/kernel_fusion.cu
                -o ${HIP_FUSION_OBJ}
            DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/kernel_fusion.cu
                    ${CMAKE_SOURCE_DIR}/src/gpu/kernel_fusion.hpp
            COMMENT "Compiling kernel fusion with hipcc"
        )
        
        add_custom_command(
            OUTPUT ${HIP_FEATURE_OBJ}
            COMMAND ${HIPCC_EXECUTABLE}
                -DUSE_HIP
                -D__HIP_PLATFORM_AMD__
                -I${HIP_INCLUDE_DIR}
                -I${CMAKE_SOURCE_DIR}/src
                -std=c++17
                -O3
                -fPIC
                -c
                ${CMAKE_SOURCE_DIR}/src/gpu/feature_kernels.cu
                -o ${HIP_FEATURE_OBJ}
            DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/feature_kernels.cu
                    ${CMAKE_SOURCE_DIR}/src/gpu/feature_kernels.hpp
            COMMENT "Compiling feature extraction kernels with hipcc"
        )
        
        add_custom_command(
            OUTPUT ${HIP_QUANTUM_OBJ}
            COMMAND ${HIPCC_EXECUTABLE}
                -DUSE_HIP
                -D__HIP_PLATFORM_AMD__
                -I${HIP_INCLUDE_DIR}
                -I${CMAKE_SOURCE_DIR}/src
                -std=c++17
                -O3
                -c
                ${CMAKE_SOURCE_DIR}/src/gpu/quantum_kernels.cu
                -o ${HIP_QUANTUM_OBJ}
            DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/quantum_kernels.cu
                    ${CMAKE_SOURCE_DIR}/src/gpu/quantum_kernels.hpp
            COMMENT "Compiling quantum embedding kernels with hipcc"
        )
        
        # Store HIP kernel objects separately - they need to be linked directly to executables
        # Device code from static libraries doesn't get properly extracted
        set(HIP_KERNEL_OBJECTS ${HIP_KERNEL_OBJ} ${HIP_FUSION_OBJ} ${HIP_FEATURE_OBJ} ${HIP_QUANTUM_OBJ})
        message(STATUS "GPU kernels will be compiled with hipcc")
        message(STATUS "HIP kernel objects will be linked directly to executables")
        
        # CRITICAL: Set hipcc as the linker for executables that use HIP kernels
        # Regular linkers (g++/clang++) don't extract .hip_fatbin device code sections from static libraries
        # hipcc knows how to extract and link device code properly
        set(CMAKE_CXX_LINK_EXECUTABLE 
            "${HIPCC_EXECUTABLE} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
            CACHE STRING "Linker command for HIP executables" FORCE
        )
        message(STATUS "Using hipcc for linking executables with HIP kernels")
    else()
        message(WARNING "hipcc not found - GPU kernels will not be compiled")
    endif()
endif()

# Library - Note: HIP kernel object files are NOT included in static library
# because device code needs to be linked directly into executables
add_library(kan_speech_model STATIC
    ${CORE_SOURCES}
    ${QUANTUM_SOURCES}
    ${AUDIO_SOURCES}
    ${DATA_SOURCES}
    ${TRAINING_SOURCES}
    ${MODEL_SOURCES}
    ${INFERENCE_SOURCES}
    ${GPU_SOURCES}
)
# HIP kernel object files are added directly to executables, not the static library
# This ensures device code is properly linked

# Link HIP and rocFFT if available
if(HIP_LIBRARY)
    target_link_libraries(kan_speech_model PUBLIC ${HIP_LIBRARY})
    target_compile_definitions(kan_speech_model PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    
    # Try to find and link rocFFT
    find_library(ROCFFT_LIBRARY
        NAMES rocfft
        PATHS /opt/rocm/lib
        NO_DEFAULT_PATH
    )
    
    if(ROCFFT_LIBRARY)
        target_link_libraries(kan_speech_model PUBLIC ${ROCFFT_LIBRARY})
        target_include_directories(kan_speech_model PUBLIC /opt/rocm/include)
        message(STATUS "rocFFT found: ${ROCFFT_LIBRARY}")
    else()
        message(STATUS "rocFFT not found - using optimized DFT kernels")
    endif()
    
    message(STATUS "GPU support enabled with HIP (AMD)")
endif()

# Example executables
add_executable(example src/example/main.cpp)
target_link_libraries(example PRIVATE kan_speech_model)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(audio_example src/example/audio_example.cpp)
target_link_libraries(audio_example PRIVATE kan_speech_model)
target_include_directories(audio_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(training_example src/example/training_example.cpp)
target_link_libraries(training_example PRIVATE kan_speech_model)
target_include_directories(training_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(full_model_example src/example/full_model_example.cpp)
target_link_libraries(full_model_example PRIVATE kan_speech_model)
target_include_directories(full_model_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(language_example src/example/language_example.cpp)
target_link_libraries(language_example PRIVATE kan_speech_model)
target_include_directories(language_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(evaluation_example src/example/evaluation_example.cpp)
target_link_libraries(evaluation_example PRIVATE kan_speech_model)
target_include_directories(evaluation_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(dataset_example src/example/dataset_example.cpp)
target_link_libraries(dataset_example PRIVATE kan_speech_model)
target_include_directories(dataset_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(inference_optimized_example src/example/inference_optimized_example.cpp)
target_link_libraries(inference_optimized_example PRIVATE kan_speech_model)
target_include_directories(inference_optimized_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Training tool
add_executable(train tools/train.cpp)
target_link_libraries(train PRIVATE kan_speech_model)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Propagate HIP definitions to executable
if(HIP_LIBRARY)
    target_compile_definitions(train PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    target_link_libraries(train PRIVATE ${HIP_LIBRARY})
    # Link HIP kernel object files directly to ensure device code is included
    # Device code from static libraries doesn't get properly extracted by the regular linker
    if(HIPCC_EXECUTABLE AND DEFINED HIP_KERNEL_OBJECTS)
        target_sources(train PRIVATE ${HIP_KERNEL_OBJECTS})
        # Ensure these are treated as object files, not sources
        set_source_files_properties(${HIP_KERNEL_OBJECTS} PROPERTIES
            EXTERNAL_OBJECT TRUE
            GENERATED TRUE
        )
        # CRITICAL: Use hipcc for linking to extract device code from .hip_fatbin sections
        # Regular linkers (g++/clang++) don't extract HIP device code from static libraries
        # hipcc knows how to extract and link the device code properly
        set_target_properties(train PROPERTIES
            LINKER_LANGUAGE CXX
        )
        # Override linker command to use hipcc which handles device code extraction
        set(CMAKE_CXX_LINK_EXECUTABLE 
            "${HIPCC_EXECUTABLE} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
            PARENT_SCOPE
        )
    endif()
endif()

