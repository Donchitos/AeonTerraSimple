# AeonTerra C++ Core Library

# Add subdirectories
add_subdirectory(heightmap)
add_subdirectory(optimization)

# Core source files (without tectonics)
set(CORE_SOURCES
    core/spherical_grid.cpp
    core/vector_types.cpp
    core/spatial_index.cpp
    core/planet.cpp
)

# Create core library (without tectonics)
add_library(aeonterracpp_core STATIC ${CORE_SOURCES})

# Source files for full library
set(SOURCES ${CORE_SOURCES})

# Add tectonics if enabled
if(BUILD_TECTONICS)
    list(APPEND SOURCES core/tectonics.cpp)
    add_definitions(-DBUILD_TECTONICS)
endif()

# Add heightmap sources
list(APPEND SOURCES ${HEIGHTMAP_SOURCES})

# Add optimization sources if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/optimization/CMakeLists.txt")
    add_subdirectory(optimization)
    # Variable OPTIMIZATION_SOURCES is defined in optimization/CMakeLists.txt
    list(APPEND SOURCES ${OPTIMIZATION_SOURCES})
endif()

# Create full library
add_library(aeonterracpp STATIC ${SOURCES})

# Include directories for core library
target_include_directories(aeonterracpp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link dependencies for core library
target_link_libraries(aeonterracpp_core PUBLIC
    Eigen3::Eigen
    Threads::Threads
)

# Include directories for full library
target_include_directories(aeonterracpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link dependencies for full library
target_link_libraries(aeonterracpp PUBLIC
    Eigen3::Eigen
    Threads::Threads
)

# Compiler options for core library
target_compile_options(aeonterracpp_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Compiler options for full library
target_compile_options(aeonterracpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Installation for core library
install(TARGETS aeonterracpp_core
    EXPORT aeonterracpp-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Installation for full library
install(TARGETS aeonterracpp
    EXPORT aeonterracpp-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Add simplified heightmap demo executable
add_executable(simplified_heightmap_demo ../simplified_heightmap_demo.cpp)
target_link_libraries(simplified_heightmap_demo PRIVATE aeonterracpp_core)