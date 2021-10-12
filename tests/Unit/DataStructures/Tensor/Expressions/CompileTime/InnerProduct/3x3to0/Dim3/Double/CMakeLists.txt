# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_InnerProduct3x3to0Dim3Double")

set(LIBRARY_SOURCES
  Test_InnerProduct3x3to0Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/InnerProduct/3x3to0/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
