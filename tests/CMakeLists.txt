cmake_minimum_required(VERSION 3.20)

# Add Catch2
find_package(Catch2 REQUIRED)

# Test executable
add_executable(tests
    ${CMAKE_SOURCE_DIR}/tests/unit/test_tensor.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_kan_layers.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_wavefunction.cpp
    ${CMAKE_SOURCE_DIR}/tests/unit/test_quantum_embedding.cpp
)

target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(tests PRIVATE
    kan_speech_model
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
Catch_discover_tests(tests)

# Enable testing
enable_testing()
add_test(NAME AllTests COMMAND tests)

