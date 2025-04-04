#!/bin/bash
# Build script for AeonTerraCpp

set -e  # Exit on error

# Parse arguments
BUILD_TYPE="Release"
BUILD_TESTS=true
BUILD_PYTHON=true

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --debug)
      BUILD_TYPE="Debug"
      ;;
    --no-tests)
      BUILD_TESTS=false
      ;;
    --no-python)
      BUILD_PYTHON=false
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --debug     Build in debug mode"
      echo "  --no-tests  Skip building tests"
      echo "  --no-python Skip building Python bindings"
      echo "  --help      Show this help message"
      exit 0
      ;;
  esac
done

echo "Building AeonTerraCpp in $BUILD_TYPE mode"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake_args="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

if [ "$BUILD_TESTS" = false ]; then
  cmake_args="$cmake_args -DBUILD_TESTS=OFF"
fi

if [ "$BUILD_PYTHON" = false ]; then
  cmake_args="$cmake_args -DBUILD_PYTHON_BINDINGS=OFF"
fi

cmake $cmake_args ..

# Build
echo "Building..."
cmake --build . -j$(nproc)

# Run tests if enabled
if [ "$BUILD_TESTS" = true ]; then
  echo "Running tests..."
  ctest --output-on-failure
fi

# Show success message
echo "Build completed successfully!"

# Information about Python module
if [ "$BUILD_PYTHON" = true ]; then
  echo ""
  echo "Python module information:"
  echo "---------------------------"
  python_path=$(find . -name "_aeonterracpp*.so" | head -1)
  if [ -n "$python_path" ]; then
    echo "Python module built at: $python_path"
    echo ""
    echo "To use the Python module, you can:"
    echo "1. Add the following to your PYTHONPATH:"
    echo "   export PYTHONPATH=\$PYTHONPATH:$(pwd)/python"
    echo "2. Or copy the module to your Python site-packages directory"
    echo ""
    echo "Example usage (from the cpp directory):"
    echo "  PYTHONPATH=\$PYTHONPATH:$(pwd)/python python3 ../example.py"
  else
    echo "Python module not found. There might have been an issue with the build."
  fi
fi

# Return to original directory
cd ..