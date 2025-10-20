// Distributed under the MIT License.
// See LICENSE.txt for details.

// Rank 3 test cases for tenex::evaluate are split into this file and
// Test_EvaluateRank3Symmetric.cpp in order to reduce compile time memory usage
// per cpp file.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3NonSymmetric",
    "[DataStructures][Unit]") {
  // nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::d, ti::A, ti::i, Symmetry<3, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<1, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::b, ti::a, ti::C, Symmetry<2, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::i, ti::f, ti::j, Symmetry<1, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  // second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::J, ti::M, ti::I, Symmetry<2, 1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  // symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::f, ti::d, ti::a, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  // TODO : remove
  // compilation stats as of: e4dda597aa
  // User time (seconds): 21.45
  // System time (seconds): 1.19
  // Percent of CPU this job got: 99%
  // Elapsed (wall clock) time (h:mm:ss or m:ss): 0:22.70
  // Maximum resident set size (kbytes): 1081912
}
