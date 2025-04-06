#!/bin/bash

# Exit on error
set -e

echo "Performing final manual cleanup..."

# Go to the cpp directory
cd planet_sim/cpp

# Rename the main demo file
echo "Renaming simplified_heightmap_demo.cpp to heightmap_demo.cpp"
mv simplified_heightmap_demo.cpp heightmap_demo.cpp

# Rename the run script
echo "Renaming run_simplified_demo.sh to run_demo.sh"
mv run_simplified_demo.sh run_demo.sh

# Remove redundant build scripts
echo "Removing redundant build scripts"
rm -f build_heightmap_demo.sh run_heightmap_demo.sh

# Remove verification files
echo "Removing verification files"
rm -f simplified_verification simplified_verification.cpp run_simplified_verification.sh

# Update build script (optional, if we want to keep it updated)
echo "Updating build scripts"
mv build_simplified_demo.sh build_new.sh
rm -f build.sh  # Remove the old build.sh
mv build_new.sh build.sh  # Rename the new one to build.sh

# Also remove the cleanup scripts since they've done their job
rm -f cleanup_project.sh

echo "Cleanup complete!"