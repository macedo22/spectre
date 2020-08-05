# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "TensorExpressionsHelpers")

set(LIBRARY_SOURCES
  ComputeRhsTensorIndexRank2TestHelpers.cpp
  #ComputeRhsTensorIndexRank3TestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  )
