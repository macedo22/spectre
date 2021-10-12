# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ContractRank2to0Dim3DataVector5")

set(LIBRARY_SOURCES
  Test_ContractRank2to0Dim3DataVector5.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/Contract/Rank2to0/Dim3/DataVector5/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
