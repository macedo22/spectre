# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_OuterProduct0x0to0Dim3Double")

set(LIBRARY_SOURCES
  Test_OuterProduct0x0to0Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/OuterProduct/0x0to0/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
