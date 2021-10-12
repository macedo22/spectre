# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_InnerProduct4x4to0Dim3DataVector5")

set(LIBRARY_SOURCES
  Test_InnerProduct4x4to0Dim3DataVector5.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/InnerProduct/4x4to0/Dim3/DataVector5/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
