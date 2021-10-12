# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ContractRank4to2Dim3Double")

set(LIBRARY_SOURCES
  Test_ContractRank4to2Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/Contract/Rank4to2/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
