# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Expressions_Benchmark_General_inner_product_spacetime_2x2")

set(LIBRARY_SOURCES
  Test_Benchmark.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/Expressions/Benchmark/General/inner_product_spacetime_2x2"
  "${LIBRARY_SOURCES}"
  "DataStructures;Utilities"
  )
