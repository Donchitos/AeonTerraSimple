#!/bin/bash

# Exit on error
set -e

# Display commands being executed
set -x

# Make sure we're in the correct directory
cd "$(dirname "$0")"

# Create build directory if it doesn't exist
mkdir -p build_simple

# Navigate to build directory
cd build_simple

# Configure with CMake (disable tectonics and Python bindings)
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TECTONICS=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF

# Build the core library (without tectonics)
make -j4 aeonterracpp_core

# Get the Eigen include path - we need the directory that contains the Eigen/ folder
# Try different search strategies to find it
EIGEN_INCLUDE_DIR="_deps/eigen-src"

if [ ! -d "$EIGEN_INCLUDE_DIR/Eigen" ]; then
    echo "Searching for Eigen directory..."
    
    # Try to find the correct path
    for dir in $(find _deps -type d -name "eigen*"); do
        if [ -d "$dir/Eigen" ]; then
            EIGEN_INCLUDE_DIR="$dir"
            break
        fi
    done
fi

echo "Found Eigen at: $EIGEN_INCLUDE_DIR"

# Verify the Eigen path is correct
if [ ! -f "$EIGEN_INCLUDE_DIR/Eigen/Core" ]; then
    echo "Error: Could not find Eigen/Core header"
    exit 1
fi

# Find library location
if [ -f "src/libaeonterracpp_core.a" ]; then
    LIB_PATH="src"
elif [ -f "libaeonterracpp_core.a" ]; then
    LIB_PATH="."
else
    echo "Error: Could not find libaeonterracpp_core.a"
    find . -name "libaeonterracpp_core.a"
    exit 1
fi

echo "Found library at: $LIB_PATH"

# Compile the simplified heightmap demo directly
g++ -o heightmap_demo ../simplified_heightmap_demo.cpp \
    -I"$EIGEN_INCLUDE_DIR" \
    -I../include \
    -L"$LIB_PATH" -laeonterracpp_core \
    -std=c++17 -O3 \
    -Wl,-rpath,.

# Return to original directory
cd ..

echo "Build completed successfully. Run with: ./run_simplified_demo.sh"

# Create the run script
cat > run_simplified_demo.sh << 'EOF'
#!/bin/bash

# Exit on error
set -e

# Directory for output
OUTPUT_DIR="./heightmap_output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Change to the build directory
cd build_simple

# Run the heightmap demo
echo "Running Simplified Heightmap Demo..."
./heightmap_demo

# Move output files to output directory
echo "Moving output files to $OUTPUT_DIR"
mv *.pgm ../$OUTPUT_DIR/

# Return to original directory
cd ..

echo ""
echo "======================================================="
echo "Heightmap Demo Completed!"
echo "PGM files saved in: $OUTPUT_DIR/"
echo "======================================================="
echo ""
echo "To view these PGM files:"
echo "1. In a local environment, most image viewers can open PGM files"
echo "2. The ASCII visualization shown in the terminal gives a preview"
echo "3. If you have ImageMagick installed, you can convert them with:"
echo "   convert $OUTPUT_DIR/final_heightmap.pgm $OUTPUT_DIR/final_heightmap.png"
echo ""
EOF

# Make the run script executable
chmod +x ./run_simplified_demo.sh