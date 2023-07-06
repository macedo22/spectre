# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Informer  # tells us things like we're on proc X and on branch Y
  Options
  Parallel
  Utilities
  )

add_spectre_parallel_executable(
    PiMonteCarlo  # name of executable
    PiMonteCarlo  # name of hpp source file
    Executables/Examples/PiMonteCarlo  # dir of executable?
    Metavariables  # could be named something else
    "${LIBS_TO_LINK}"
  )
