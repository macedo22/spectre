# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DomainCreators
  Informer
  IO
  Utilities
  )

add_spectre_parallel_executable(
  PrintExportCoordinates1D
  PrintExportCoordinates
  Executables/PrintExportCoordinates
  Metavariables<1>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  PrintExportCoordinates2D
  PrintExportCoordinates
  Executables/PrintExportCoordinates
  Metavariables<2>
  "${LIBS_TO_LINK}"
  )

add_spectre_parallel_executable(
  PrintExportCoordinates3D
  PrintExportCoordinates
  Executables/PrintExportCoordinates
  Metavariables<3>
  "${LIBS_TO_LINK}"
  )
