# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SphericalHarmonicsIO)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ReadYlmCoefficients.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ReadYlmCoefficients.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  Boost::boost
  DataStructures
  ErrorHandling
  SPHEREPACK
  Utilities
  PRIVATE
  Interpolation
  )
