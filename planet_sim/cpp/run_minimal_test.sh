#!/bin/bash
# Simple script to build and run the minimal test

set -e  # Exit on error

echo "Building and running minimal test..."

# Create build directory if it doesn't exist
mkdir -p test_build
cd test_build

# Configure minimal build
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF

# Build just the core library and minimal test
echo "Building minimal test..."
cmake --build . --target aeonterracpp minimal_test

# Run the minimal test with more debug output
echo "Running minimal test..."
echo "-------------------------------------------"
./minimal_test
EXIT_CODE=$?
echo "-------------------------------------------"
echo "Minimal test exited with code: $EXIT_CODE"

cd ..