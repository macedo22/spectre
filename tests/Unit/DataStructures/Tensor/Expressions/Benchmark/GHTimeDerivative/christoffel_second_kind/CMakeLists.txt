# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions_Benchmark_GH_christoffel_second_kind")

set(LIBRARY_SOURCES
  Test_Benchmark.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/Benchmark/GHTimeDerivative/christoffel_second_kind"
  "${LIBRARY_SOURCES}"
  "DataStructures;GeneralRelativity;Utilities"
  )
