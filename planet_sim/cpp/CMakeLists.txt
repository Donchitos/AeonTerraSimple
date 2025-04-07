cmake_minimum_required(VERSION 3.14)
project(AeonTerraCpp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Eigen3 REQUIRED)
find_package(Pybind11 REQUIRED)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
endif()

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Create build target list
set(AEONTERRACPP_SOURCES
    src/core/vector_types.cpp
    src/core/spatial_index.cpp
    src/core/spherical_grid.cpp
    src/core/planet.cpp
    src/core/tectonics.cpp
    src/core/heightmap_generator.cpp
    src/core/enhanced_planet.cpp
    src/core/atmospheric_circulation.cpp
    src/core/oceanic_circulation.cpp
    src/core/hydrological_system.cpp
    src/core/climate_system.cpp
    src/core/biome_system.cpp
    src/core/enhanced_erosion.cpp
    src/core/biome_climate_feedback.cpp
    src/core/river_ocean_chemistry.cpp
)

# Compile library
add_library(aeonterracpp STATIC ${AEONTERRACPP_SOURCES})
target_link_libraries(aeonterracpp PRIVATE Eigen3::Eigen)

# Create Python module
pybind11_add_module(_aeonterracpp 
    python/bindings.cpp
    python/spherical_grid_bindings.cpp
    python/planet_bindings.cpp
    python/tectonics_bindings.cpp
    python/heightmap_bindings.cpp
    python/climate_bindings.cpp
    python/biome_bindings.cpp
    python/erosion_bindings.cpp
    python/feedback_bindings.cpp
    python/chemistry_bindings.cpp
)
target_link_libraries(_aeonterracpp PRIVATE aeonterracpp)

# Install targets
install(TARGETS aeonterracpp DESTINATION lib)
install(TARGETS _aeonterracpp DESTINATION python/planet_sim/cpp/python)
install(DIRECTORY include/ DESTINATION include)

# Add tests
# Comment out if not building tests
add_subdirectory(tests)