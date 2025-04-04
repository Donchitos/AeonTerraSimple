#!/bin/bash
# Run the fixed test

set -e  # Exit on error

echo "Building and running fixed test..."

# Create build directory if it doesn't exist
mkdir -p test_build
cd test_build

# Configure minimal build
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF

# Build the fixed test
echo "Building fixed test..."
cmake --build . --target aeonterracpp fixed_test

# Run the fixed test
echo "Running fixed test..."
echo "-------------------------------------------"
./fixed_test
EXIT_CODE=$?
echo "-------------------------------------------"
echo "Fixed test exited with code: $EXIT_CODE"

cd ..