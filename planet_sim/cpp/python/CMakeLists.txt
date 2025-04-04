# Python bindings for AeonTerraCpp

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11 should be found in the main CMakeLists.txt

# Python module
pybind11_add_module(aeonterracpp_python
    bindings.cpp
    spherical_grid_bindings.cpp
    planet_bindings.cpp
)

# Rename the target (to avoid the default name being 'python')
set_target_properties(aeonterracpp_python PROPERTIES
    OUTPUT_NAME _aeonterracpp
)

# Link with the C++ library
target_link_libraries(aeonterracpp_python PRIVATE
    aeonterracpp
    pybind11::module
)

# Installation
install(TARGETS aeonterracpp_python
    LIBRARY DESTINATION ${Python_SITEARCH}/aeonterracpp
)

# Add the Python package __init__.py
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
    COPYONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
    DESTINATION ${Python_SITEARCH}/aeonterracpp
)