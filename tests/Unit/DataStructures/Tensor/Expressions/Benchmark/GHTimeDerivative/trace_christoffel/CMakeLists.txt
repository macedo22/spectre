# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions_Benchmark_GH_trace_christoffel")

set(LIBRARY_SOURCES
  Test_Benchmark.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/Benchmark/GHTimeDerivative/trace_christoffel"
  "${LIBRARY_SOURCES}"
  "DataStructures;GeneralRelativity;Utilities"
  )
