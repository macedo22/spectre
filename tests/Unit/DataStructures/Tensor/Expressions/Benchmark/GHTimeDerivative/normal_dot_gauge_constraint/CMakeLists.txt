# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions_Benchmark_GH_normal_dot_gauge_constraint")

set(LIBRARY_SOURCES
  Test_Benchmark.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/Benchmark/GHTimeDerivative/normal_dot_gauge_constraint"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
