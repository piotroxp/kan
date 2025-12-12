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

# Find packages
find_package(Eigen3 REQUIRED)

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
)

set(DATA_SOURCES
    src/data/fsd50k_loader.cpp
)

# Library
add_library(kan_speech_model STATIC
    ${CORE_SOURCES}
    ${QUANTUM_SOURCES}
    ${AUDIO_SOURCES}
    ${DATA_SOURCES}
)

target_link_libraries(kan_speech_model PUBLIC
    Eigen3::Eigen
)

# Example executables
add_executable(example src/example/main.cpp)
target_link_libraries(example PRIVATE kan_speech_model)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(audio_example src/example/audio_example.cpp)
target_link_libraries(audio_example PRIVATE kan_speech_model)
target_include_directories(audio_example PRIVATE ${CMAKE_SOURCE_DIR}/src)

