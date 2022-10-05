# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EvolveGhBinaryBlackHoleOptions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Options.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Options
  Parallel
  Utilities
  )
