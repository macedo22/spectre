# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ExpressionsHelpers")

set(LIBRARY_SOURCES
  EvaluateRankN.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  )
