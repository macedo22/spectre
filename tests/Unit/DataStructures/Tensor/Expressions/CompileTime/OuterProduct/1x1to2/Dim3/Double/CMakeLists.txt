# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_OuterProduct1x1to2Dim3Double")

set(LIBRARY_SOURCES
  Test_OuterProduct1x1to2Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/OuterProduct/1x1to2/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
