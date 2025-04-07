set(OPTIMIZATION_SOURCES
    spatial_acceleration.cpp
    parallel_processing.cpp
    memory_optimization.cpp
)

# Add these sources to the main library
target_sources(aeonterracpp PRIVATE ${OPTIMIZATION_SOURCES})