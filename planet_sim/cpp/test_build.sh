#!/bin/bash
# Simple build test script for core components only

set -e  # Exit on error

echo "Building AeonTerraCpp core components for testing..."

# Create build directory
mkdir -p test_build
cd test_build

# Configure minimal build
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF

# Build just the core library and simple test
echo "Building core components..."
cmake --build . --target aeonterracpp simple_test

# If build succeeds, run the simple test
if [ $? -eq 0 ]; then
    echo "Build successful! Running simple test..."
    ./simple_test
else
    echo "Build failed!"
    exit 1
fi

cd ..
echo "Test build completed."