# Heightmap generation source files
set(HEIGHTMAP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/heightmap_generator.cpp
)

# Add source files to parent scope
set(HEIGHTMAP_SOURCES ${HEIGHTMAP_SOURCES} PARENT_SCOPE)