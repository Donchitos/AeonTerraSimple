#!/bin/bash
# Run just the verification test for the Planet class

set -e  # Exit on error

echo "========== Running Verification Test =========="

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Make sure build_cpp_only was already run
if [ ! -f "build_cpp/verification_test" ]; then
    echo "Building verification test first..."
    ./build_cpp_only.sh
fi

# Run the verification test
echo -e "\n[RUNNING VERIFICATION TEST]"
echo "-------------------------------------------"
./build_cpp/verification_test
VERIFY_EXIT=$?

echo -e "\n========== Test Complete =========="
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "✅ Planet implementation verification passed!"
else
    echo "❌ Verification failed with exit code $VERIFY_EXIT"
fi