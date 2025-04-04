#!/bin/bash
# Clean up temporary build files and directories

echo "Cleaning up build directories and temporary files..."

# Remove build directories
rm -rf build/
rm -rf build_cpp/
rm -rf test_build/

# Remove any generated binary files
rm -f simple_test
rm -f minimal_test
rm -f fixed_test
rm -f verification_test
rm -f test_planet
rm -f test_spherical_grid

# Remove any generated test outputs
rm -f example_planet.bin
rm -f *.png
rm -f *.bin

echo "Cleanup complete!"
echo "Note: These directories are now added to .gitignore and won't be committed to the repository."