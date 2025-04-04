# AeonTerra C++ Implementation

This directory contains the C++ implementation of the AeonTerra planet generation system, focusing on high-performance heightmap generation and terrain simulation.

## Structure

- `include/` - Header files
- `src/` - Implementation files
- `python/` - Python bindings (using pybind11)
- `tests/` - C++ unit tests
- `cmake/` - CMake modules and utilities

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Requirements

- C++17 compatible compiler
- CMake 3.14+
- Python 3.7+
- pybind11
- Eigen 3
- OpenGL 4.3+ (for compute shaders)