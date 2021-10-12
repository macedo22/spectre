# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AddSubtractOp1Dim3Double")

set(LIBRARY_SOURCES
  Test_AddSubtractOp1Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/AddSubtract/Op1/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
