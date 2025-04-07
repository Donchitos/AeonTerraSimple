cmake_minimum_required(VERSION 3.14)

# Add the include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Define test executables
set(TEST_EXECUTABLES
    test_spherical_grid
    test_spatial_index
    test_vector_types
    test_climate_system
    test_biome_system
)

# Create test targets
foreach(test_name ${TEST_EXECUTABLES})
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE aeonterracpp)
endforeach()