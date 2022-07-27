# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EvolveGhBinaryBlackHoleFactored)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FactoryCreation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FactoryCreation.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainStructure
  ErrorHandling
  InitialDataUtilities
  Options
  Spectral
  Time
  Utilities
  PRIVATE
  LinearOperators
  INTERFACE
  SystemUtilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
