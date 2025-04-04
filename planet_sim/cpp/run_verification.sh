#!/bin/bash
# Run verification test to ensure code is ready for the next phase

set -e  # Exit on error

echo "========== AeonTerra C++ Implementation Verification =========="
echo "Building and running verification test..."

# Create build directory if it doesn't exist
mkdir -p test_build
cd test_build

# Configure
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF

# Build the verification test
echo "Building verification test..."
cmake --build . --target aeonterracpp verification_test

# Run the verification test
echo -e "\n[RUNNING VERIFICATION TEST]"
echo "-------------------------------------------"
./verification_test
EXIT_CODE=$?
echo "-------------------------------------------"
echo "Verification test exited with code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✓ VERIFICATION PASSED: The C++ implementation is ready for the next phase!"
else
    echo -e "\n✗ VERIFICATION FAILED: Issues need to be fixed before proceeding."
fi

cd ..