#!/bin/bash
set -e

echo "=== Building KAN Speech Model ==="

# Create build directory
mkdir -p build
cd build

# Install Conan dependencies if needed
if [ ! -f "conan_toolchain.cmake" ]; then
    echo "Installing Conan dependencies..."
    conan install .. --build=missing
fi

# Configure CMake
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Build
echo "Building..."
cmake --build . -j$(nproc)

echo "=== Build complete! ==="
echo ""
echo "To run tests:"
echo "  cd build && ctest"
echo ""
echo "To run example:"
echo "  cd build && ./example"

