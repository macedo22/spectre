# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_OuterProduct1x2x1to4Dim3DataVector5")

set(LIBRARY_SOURCES
  Test_OuterProduct1x2x1to4Dim3DataVector5.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/OuterProduct/1x2x1to4/Dim3/DataVector5/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
