# FindEigen3.cmake - Find Eigen3 library
#
# This module defines:
#  EIGEN3_INCLUDE_DIR - Directory containing Eigen headers
#  EIGEN3_FOUND - True if Eigen was found
#
# Based on the official Eigen3 FindEigen3.cmake with modifications

if(NOT Eigen3_FIND_VERSION)
  if(NOT Eigen3_FIND_VERSION_MAJOR)
    set(Eigen3_FIND_VERSION_MAJOR 3)
  endif()
  if(NOT Eigen3_FIND_VERSION_MINOR)
    set(Eigen3_FIND_VERSION_MINOR 3)
  endif()
  if(NOT Eigen3_FIND_VERSION_PATCH)
    set(Eigen3_FIND_VERSION_PATCH 0)
  endif()
  set(Eigen3_FIND_VERSION "${Eigen3_FIND_VERSION_MAJOR}.${Eigen3_FIND_VERSION_MINOR}.${Eigen3_FIND_VERSION_PATCH}")
endif()

macro(_eigen3_check_version)
  file(READ "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen3_version_header)

  string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen3_world_version_match "${_eigen3_version_header}")
  set(EIGEN3_WORLD_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen3_major_version_match "${_eigen3_version_header}")
  set(EIGEN3_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen3_minor_version_match "${_eigen3_version_header}")
  set(EIGEN3_MINOR_VERSION "${CMAKE_MATCH_1}")

  set(EIGEN3_VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})
  if(${EIGEN3_VERSION} VERSION_LESS ${Eigen3_FIND_VERSION})
    set(EIGEN3_VERSION_OK FALSE)
  else()
    set(EIGEN3_VERSION_OK TRUE)
  endif()
endmacro()

# Try to find Eigen3 in standard paths
find_path(EIGEN3_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library
  HINTS
  ${CMAKE_INSTALL_PREFIX}/include
  ${KDE4_INCLUDE_DIR}
  PATH_SUFFIXES eigen3 eigen
)

# If not found, try to use FetchContent to download Eigen
if(NOT EIGEN3_INCLUDE_DIR)
  message(STATUS "Eigen3 not found in standard paths, will try to fetch it")
  
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    message(STATUS "Fetching Eigen3...")
    FetchContent_Populate(eigen)
    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
    set(EIGEN3_VERSION "3.4.0")
    set(EIGEN3_VERSION_OK TRUE)
  endif()
else()
  # Check version of found Eigen
  if(EIGEN3_INCLUDE_DIR)
    _eigen3_check_version()
  endif()
endif()

# Set FOUND flag based on include dir and version
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3
  REQUIRED_VARS EIGEN3_INCLUDE_DIR EIGEN3_VERSION_OK
  VERSION_VAR EIGEN3_VERSION
)

# Create imported target if not already defined
if(EIGEN3_FOUND AND NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(EIGEN3_INCLUDE_DIR)