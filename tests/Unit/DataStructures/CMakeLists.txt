# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataStructures")

set(LIBRARY_SOURCES
  Test_DataVector.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  )
