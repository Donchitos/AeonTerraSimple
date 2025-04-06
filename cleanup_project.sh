#!/bin/bash

# Exit on error
set -e

echo "Cleaning up redundant files in AeonTerra C++ implementation..."

# Remove build directories
echo "Removing build directories..."
rm -rf build/ build_cpp/ build_simple/ 

# Remove old test files
echo "Removing unnecessary test files..."
rm -f minimal_test.cpp simple_test.cpp fixed_test.cpp verification_test.cpp

# Remove old test scripts
echo "Removing old test scripts..."
rm -f run_minimal_test.sh run_fixed_test.sh run_simple_verify.sh 
rm -f run_verification.sh run_verification_test.sh run_tectonic_verification.sh
rm -f verify_full_pipeline.sh test_build.sh run_heightmap_verification.sh

# Remove redundant build scripts
echo "Removing redundant build scripts..."
rm -f build_cpp_only.sh build_planet.sh cleanup.sh build_heightmap_demo.sh

# Remove old example files (we now have simplified version)
echo "Removing old example files..."
rm -f example.py heightmap_example.py planet_example.py tectonics_example.py

# Remove redundant heightmap demo (we use simplified version)
rm -f minimal_heightmap_demo.cpp

# Remove test raw file
rm -f test_heightmap.raw

# Remove verification files
rm -f simplified_verification simplified_verification.cpp run_simplified_verification.sh

# We only need one build script, keep build_simplified_demo.sh and rename it
mv build_simplified_demo.sh build.sh

# Clean up tectonic verification files (replaced by our new implementation)
rm -f tectonic_verification.cpp

# Keep only the essential files for our clean implementation
echo "Keeping essential files:"
echo " - include/ (C++ headers)"
echo " - src/ (C++ implementation)"
echo " - python/ (Python bindings)"
echo " - cmake/ (CMake modules)"
echo " - CMakeLists.txt (Build system)"
echo " - README.md (Documentation)"
echo " - simplified_heightmap_demo.cpp (Main demo)"
echo " - build.sh (Build script)"
echo " - run_simplified_demo.sh (Run script)"
echo " - heightmap_output/ (Generated terrain images)"

# Rename the demo file to make it the main one
mv simplified_heightmap_demo.cpp heightmap_demo.cpp
mv run_simplified_demo.sh run_demo.sh

echo "Cleanup complete!"