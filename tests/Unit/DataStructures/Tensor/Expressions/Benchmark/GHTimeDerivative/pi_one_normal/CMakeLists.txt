# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions_Benchmark_GH_pi_one_normal")

set(LIBRARY_SOURCES
  Test_Benchmark.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/Benchmark/GHTimeDerivative/pi_one_normal"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
