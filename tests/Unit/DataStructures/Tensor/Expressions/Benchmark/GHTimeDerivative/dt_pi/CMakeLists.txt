# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions_Benchmark_GH_dt_pi")

set(LIBRARY_SOURCES
  Test_Benchmark.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/Benchmark/GHTimeDerivative/dt_pi"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
