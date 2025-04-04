#!/bin/bash
# Simple script to run the simplified verification test

set -e  # Exit on error

echo "========== Running AeonTerra Simplified Verification =========="
echo "This script will test the core components of the C++ implementation"

# Create build directory if it doesn't exist
mkdir -p build

# Compile the verification program
echo "Compiling simplified verification program..."
g++ -std=c++17 -o build/simplified_verification simplified_verification.cpp -I./include -L./build -laeonterracpp -pthread

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful"

# Run the verification program
echo -e "\nRunning verification tests..."
LD_LIBRARY_PATH=./build ./build/simplified_verification

# Check if verification was successful
if [ $? -ne 0 ]; then
    echo "❌ Verification failed!"
    exit 1
fi

echo -e "\n========== Verification Complete =========="
echo "✅ All tests passed!"
echo "The C++ implementation is working correctly."