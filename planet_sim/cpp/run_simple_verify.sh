#!/bin/bash
# Simple script to compile and run the simplified verification program

set -e  # Exit on error

echo "========== AeonTerra C++ Component Verification =========="
echo "This script will verify all core components of the C++ implementation"

# Build the core library if it doesn't exist
if [ ! -f "build/src/libaeonterracpp.a" ]; then
    echo "Building the core library..."
    ./build.sh --no-python --no-tests
fi

# Compile the verification program
echo "Compiling simplified verification program..."
g++ -std=c++17 -o simplified_verification simplified_verification.cpp \
    -Iinclude -Lbuild/src -laeonterracpp -pthread -I./build/_deps/eigen-src

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful"

# Run the verification program
echo -e "\nRunning verification tests..."
./simplified_verification

# Check if verification was successful
if [ $? -ne 0 ]; then
    echo "❌ Verification failed!"
    exit 1
fi

echo -e "\n========== Verification Complete =========="
echo "✅ All tests passed!"
echo "The C++ implementation is working correctly."
echo ""
echo "Generated test heightmap: test_heightmap.raw"
echo "You can now use the C++ implementation for heightmap generation."