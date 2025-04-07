#!/bin/bash

# Exit on error
set -e

# Display commands being executed
set -x

# Make sure we're in the correct directory
cd "$(dirname "$0")"

# Create build directory if it doesn't exist
mkdir -p build_climate

# Navigate to build directory
cd build_climate

# Configure with CMake (enable everything including climate system)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the complete library and Python bindings
make -j4

# Run the climate system tests
./tests/test_climate_system

# Return to original directory
cd ..

echo "Build completed successfully."
echo "Climate system tests have been run."

# Create a run script for the climate example
cat > run_climate_example.sh << 'EOF'
#!/bin/bash

# Exit on error
set -e

# Make sure Python can find the module
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the climate example
python3 climate_example.py

echo ""
echo "======================================================="
echo "Climate Example Completed!"
echo "======================================================="
echo ""
EOF

# Make the run script executable
chmod +x ./run_climate_example.sh

echo "You can now run the climate example with: ./run_climate_example.sh"