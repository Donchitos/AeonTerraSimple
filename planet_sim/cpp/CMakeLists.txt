cmake_minimum_required(VERSION 3.14)
project(AeonTerraCpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++17 as the standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_OPENGL "Build with OpenGL support for GPU acceleration" ON)
option(USE_CUDA "Build with CUDA support for GPU acceleration" OFF)
option(BUILD_TECTONICS "Build with tectonic simulation support" ON)

# Dependencies
find_package(Eigen3 3.3 REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Core library
add_subdirectory(src)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPybind11.cmake)
    endif()
    add_subdirectory(python)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
# Test executables
add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test PRIVATE aeonterracpp)

add_executable(minimal_test minimal_test.cpp)
target_link_libraries(minimal_test PRIVATE aeonterracpp)

add_executable(fixed_test fixed_test.cpp)
target_link_libraries(fixed_test PRIVATE aeonterracpp)
add_executable(verification_test verification_test.cpp)
target_link_libraries(verification_test PRIVATE aeonterracpp)

add_executable(tectonic_verification tectonic_verification.cpp)
target_link_libraries(tectonic_verification PRIVATE aeonterracpp)



# Installation
install(DIRECTORY include/ DESTINATION include/aeonterracpp)