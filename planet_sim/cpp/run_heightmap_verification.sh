#!/bin/bash
# Run verification tests for the HeightmapGenerator class

set -e  # Exit on error

echo "========== Running HeightmapGenerator Verification =========="

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory if it doesn't exist
mkdir -p build_cpp
cd build_cpp

# Configure and build
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF

echo "Building heightmap generator tests..."
cmake --build . --target test_heightmap_generator

# Run the heightmap tests
echo -e "\n[RUNNING HEIGHTMAP GENERATOR TESTS]"
echo "-------------------------------------------"
./tests/test_heightmap_generator
VERIFY_EXIT=$?

echo -e "\n========== Verification Complete =========="
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "✅ HeightmapGenerator verification passed!"
else
    echo "❌ Verification failed with exit code $VERIFY_EXIT"
fi