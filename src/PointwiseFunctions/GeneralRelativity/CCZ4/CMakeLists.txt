# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ATilde.cpp
  ConfSpatialMetric.cpp
  ExtrinsicCurvature.cpp
  Phi.cpp
  PhiSquared.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ATilde.hpp
  ConfSpatialMetric.hpp
  ExtrinsicCurvature.hpp
  Phi.hpp
  PhiSquared.hpp
)
