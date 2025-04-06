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
