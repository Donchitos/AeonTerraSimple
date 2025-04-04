#!/bin/bash
# Build script for Planet class implementation (C++ only)

set -e  # Exit on error

echo "========== AeonTerra Planet Implementation (C++ Only) =========="
echo "Building C++ implementation without Python bindings..."

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory if it doesn't exist
mkdir -p build_cpp
cd build_cpp

# Configure with Python bindings disabled
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=ON

# Build
echo "Building libraries and tests..."
cmake --build . -- -j$(nproc)

# Run the verification test
echo -e "\n[RUNNING VERIFICATION TEST]"
echo "-------------------------------------------"
./verification_test
VERIFY_EXIT=$?

echo -e "\n========== Build and Test Complete =========="
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "✅ C++ implementation verification passed!"
else
    echo "❌ Verification failed with exit code $VERIFY_EXIT"
fi