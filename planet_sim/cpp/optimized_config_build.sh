#!/bin/bash

# Build script for AeonTerra with optimization libraries
# This script builds all components including the performance optimization libraries

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create build directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/build"

# Navigate to build directory
cd "$SCRIPT_DIR/build"

# Find dependencies
if [ -d "/usr/include/eigen3" ]; then
    EIGEN_DIR="/usr/include/eigen3"
elif [ -d "/usr/local/include/eigen3" ]; then
    EIGEN_DIR="/usr/local/include/eigen3"
else
    echo "Eigen3 not found. Using FindEigen3.cmake module."
    EIGEN_DIR=""
fi

# Find pybind11
if [ -d "/usr/include/pybind11" ]; then
    PYBIND11_DIR="/usr/include/pybind11"
elif [ -d "/usr/local/include/pybind11" ]; then
    PYBIND11_DIR="/usr/local/include/pybind11"
else
    echo "pybind11 not found. Using FindPybind11.cmake module."
    PYBIND11_DIR=""
fi

# Determine compiler version and enable C++17
CXX_VERSION=$(c++ --version | head -n 1)
echo "Using C++ compiler: $CXX_VERSION"

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_STANDARD_REQUIRED=ON \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG" \
      -DENABLE_PERFORMANCE_OPTIMIZATION=ON \
      -DBUILD_TESTING=ON \
      ${EIGEN_DIR:+-DEIGEN3_INCLUDE_DIR="$EIGEN_DIR"} \
      ${PYBIND11_DIR:+-DPYBIND11_INCLUDE_DIR="$PYBIND11_DIR"} \
      ..

# Number of processor cores
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    # Linux and others
    NUM_CORES=$(nproc)
fi

echo "Building with $NUM_CORES parallel jobs"

# Build
cmake --build . -j $NUM_CORES

# Run tests
echo "Running tests..."
ctest --output-on-failure

echo "Building Python module..."
cmake --build . --target _aeonterracpp -j $NUM_CORES

echo "Building completed successfully!"
echo ""
echo "Generated libraries:"
ls -la "$SCRIPT_DIR/build/python/*.so" 2>/dev/null || true

# Copy Python module to the python directory
if [ -f "$SCRIPT_DIR/build/python/_aeonterracpp.cpython-*.so" ]; then
    find "$SCRIPT_DIR/build/python" -name "_aeonterracpp.cpython-*.so" -exec cp {} "$SCRIPT_DIR/python/" \;
    echo "Python module copied to $SCRIPT_DIR/python/"
fi

echo "To use the optimized build, run the examples with:"
echo "cd $SCRIPT_DIR && python3 optimized_example.py"