cmake_minimum_required(VERSION 3.20)

# Test executable
add_executable(tests
    ${CMAKE_SOURCE_DIR}/tests/unit/catch2_main.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_tensor.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_kan_layers.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_wavefunction.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_quantum_embedding.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_audio.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_features.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_training.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_bench_datasets.cpp
    ${CMAKE_SOURCE_DIR}/bench/datasets/dataset.cpp
    ${CMAKE_SOURCE_DIR}/bench/datasets/discontinuous.cpp
    ${CMAKE_SOURCE_DIR}/bench/datasets/noisy.cpp
)

target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
target_link_libraries(tests PRIVATE
    kan_speech_model
)

include(CTest)
add_test(NAME UnitTests COMMAND tests)

# Enable testing
enable_testing()
add_test(NAME AllTests COMMAND tests)

