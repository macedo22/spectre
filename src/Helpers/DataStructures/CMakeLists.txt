# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SrcDataStructuresHelpers")

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  )
