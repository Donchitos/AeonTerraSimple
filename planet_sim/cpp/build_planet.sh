#!/bin/bash
# Build script for Planet class implementation

set -e  # Exit on error

echo "========== AeonTerra Planet Implementation =========="
echo "Building C++ implementation and Python bindings..."

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=ON

# Build
echo "Building libraries and tests..."
cmake --build . -- -j$(nproc)

# Run C++ tests
echo -e "\n[RUNNING C++ TESTS]"
echo "-------------------------------------------"
cd tests
./test_planet
cd ..

# Create Python environment variable for module discovery
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/build/python

# Run Python example
echo -e "\n[RUNNING PYTHON EXAMPLE]"
echo "-------------------------------------------"
cd ..
python3 planet_example.py

echo -e "\n========== Build and Test Complete =========="
echo "Planet implementation is ready for use!"