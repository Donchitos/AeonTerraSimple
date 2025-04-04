# Tests for AeonTerraCpp

# Find or fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.3.2  # Adjust version as needed
)
FetchContent_MakeAvailable(Catch2)

# Add tests
add_executable(test_spherical_grid test_spherical_grid.cpp)
target_link_libraries(test_spherical_grid PRIVATE aeonterracpp Catch2::Catch2WithMain)

add_executable(test_planet test_planet.cpp)
target_link_libraries(test_planet PRIVATE aeonterracpp Catch2::Catch2WithMain)

# Register tests
include(CTest)
include(Catch)
catch_discover_tests(test_spherical_grid)
catch_discover_tests(test_planet)