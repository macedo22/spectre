# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ConfSpatialMetric.cpp
  Phi.cpp
  PhiSquared.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConfSpatialMetric.hpp
  Phi.hpp
  PhiSquared.hpp
)
