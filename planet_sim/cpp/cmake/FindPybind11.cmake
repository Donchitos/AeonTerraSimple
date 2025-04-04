# FindPybind11.cmake - Find pybind11 library
#
# This module defines:
#  pybind11_FOUND - True if pybind11 was found
#  pybind11_INCLUDE_DIRS - Include directories for pybind11
#  pybind11::module - Imported target for pybind11

# Try to find pybind11 in standard paths
find_path(PYBIND11_INCLUDE_DIR
  NAMES pybind11/pybind11.h
  HINTS
  ${CMAKE_INSTALL_PREFIX}/include
  ${CMAKE_PREFIX_PATH}/include
)

# If not found, try to use FetchContent to download pybind11
if(NOT PYBIND11_INCLUDE_DIR)
  message(STATUS "pybind11 not found in standard paths, will try to fetch it")
  
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
  )
  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Fetching pybind11...")
    FetchContent_Populate(pybind11)
    set(PYBIND11_INCLUDE_DIR ${pybind11_SOURCE_DIR}/include)
    
    # Add pybind11 subdirectory
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    
    # Set up imported target if not already defined
    if(NOT TARGET pybind11::module)
      add_library(pybind11::module INTERFACE IMPORTED)
      set_target_properties(pybind11::module PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

# Set FOUND flag based on include dir
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(pybind11
  REQUIRED_VARS PYBIND11_INCLUDE_DIR
)

# Create imported target if found and not already defined
if(pybind11_FOUND AND NOT TARGET pybind11::module)
  add_library(pybind11::module INTERFACE IMPORTED)
  set_target_properties(pybind11::module PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(PYBIND11_INCLUDE_DIR)