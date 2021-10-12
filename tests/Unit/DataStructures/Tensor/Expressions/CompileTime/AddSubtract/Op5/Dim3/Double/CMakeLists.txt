# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AddSubtractOp5Dim3Double")

set(LIBRARY_SOURCES
  Test_AddSubtractOp5Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/AddSubtract/Op5/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
