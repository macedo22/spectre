# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DgPackageDataQuantities.cpp
  RegisterDerived.cpp
  UpwindPenalty.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  DgPackageDataQuantities.hpp
  Factory.hpp
  RegisterDerived.hpp
  UpwindPenalty.hpp
  )
