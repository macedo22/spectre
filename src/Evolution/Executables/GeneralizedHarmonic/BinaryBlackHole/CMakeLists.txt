# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  Domain
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  GeneralRelativity
  GeneralizedHarmonic
  GeneralizedHarmonicGaugeSourceFunctions
  GhConstraintDamping
  IO
  Informer
  Interpolation
  LinearOperators
  MathFunctions
  Options
  Parallel
  ParallelInterpolation
  PhaseControl
  Time
  Utilities
  )

add_spectre_parallel_executable(
  "EvolveGhBinaryBlackHole"
  EvolveGhBinaryBlackHole
  Evolution/Executables/GeneralizedHarmonic/BinaryBlackHole
  "EvolutionMetavars"
  "${LIBS_TO_LINK};ControlSystem;GeneralRelativitySolutions;Importers;EvolveGhBinaryBlackHoleOptions"
)

add_subdirectory(Options)
