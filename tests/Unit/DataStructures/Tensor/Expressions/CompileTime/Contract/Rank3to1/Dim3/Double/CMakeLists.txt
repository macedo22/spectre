# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ContractRank3to1Dim3Double")

set(LIBRARY_SOURCES
  Test_ContractRank3to1Dim3Double.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/CompileTime/Contract/Rank3to1/Dim3/Double/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
