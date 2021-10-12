# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_InnerProduct2x2to0Dim3Double")

set(LIBRARY_SOURCES
  Test_InnerProduct2x2to0Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/InnerProduct/2x2to0/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
