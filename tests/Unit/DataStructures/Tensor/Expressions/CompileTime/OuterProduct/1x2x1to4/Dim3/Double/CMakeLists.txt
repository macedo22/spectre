# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_OuterProduct1x2x1to4Dim3Double")

set(LIBRARY_SOURCES
  Test_OuterProduct1x2x1to4Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/OuterProduct/1x2x1to4/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
