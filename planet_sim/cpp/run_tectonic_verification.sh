#!/bin/bash
# Run the tectonic simulation verification

set -e  # Exit on error

echo "========== Running Tectonic Simulation Verification =========="

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory if it doesn't exist
mkdir -p build_cpp
cd build_cpp

# Configure and build
echo "Configuring with CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=OFF

echo "Building tectonic verification..."
cmake --build . --target tectonic_verification

# Run the verification test
echo -e "\n[RUNNING TECTONIC VERIFICATION]"
echo "-------------------------------------------"
./tectonic_verification
VERIFY_EXIT=$?

echo -e "\n========== Verification Complete =========="
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "✅ Tectonic simulation verification passed!"
else
    echo "❌ Verification failed with exit code $VERIFY_EXIT"
fi