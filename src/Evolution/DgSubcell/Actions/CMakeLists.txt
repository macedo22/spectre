# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Actions.hpp
  Initialize.hpp
  Labels.hpp
  ReconstructionCommunication.hpp
  SelectNumericalMethod.hpp
  TakeTimeStep.hpp
  TciAndRollback.hpp
  TciAndSwitchToDg.hpp
  )
