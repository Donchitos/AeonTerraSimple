# AeonTerra C++ Core Library

# Add subdirectories
add_subdirectory(heightmap)

# Source files
set(SOURCES
    core/spherical_grid.cpp
    core/vector_types.cpp
    core/spatial_index.cpp
    core/planet.cpp
    core/tectonics.cpp
    ${HEIGHTMAP_SOURCES}
)

# Create library
add_library(aeonterracpp STATIC ${SOURCES})

# Include directories
target_include_directories(aeonterracpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link dependencies
target_link_libraries(aeonterracpp PUBLIC
    Eigen3::Eigen
    Threads::Threads
)

# Compiler options
target_compile_options(aeonterracpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Installation
install(TARGETS aeonterracpp
    EXPORT aeonterracpp-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)