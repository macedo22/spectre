# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(GaugeSourceFunctions)
add_subdirectory(ConstraintDamping)

set(LIBRARY "Test_CCZ4")

set(LIBRARY_SOURCES
  Test_TimeDerivative.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/CCZ4/"
  "${LIBRARY_SOURCES}"
  "CCZ4;DataStructures;GeneralRelativityHelpers;GeneralizedHarmonic;Spectral;Utilities;Test_GeneralRelativity"
  )
