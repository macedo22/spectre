// Distributed under the MIT License.
// See LICENSE.txt for details.

// Rank 3 test cases for tenex::evaluate are split into this file and
// Test_EvaluateRank3NonSymmetric.cpp in order to reduce compile time memory
// usage per cpp file.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3Symmetric",
    "[DataStructures][Unit]") {
  // Rank 3: double; first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::b, ti::a, ti::C, double, Symmetry<2, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // Rank 3: double; first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::i, ti::f, ti::j, double, Symmetry<1, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  // Rank 3: double; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::D, ti::J, ti::I, double, Symmetry<2, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  // Rank 3: double; symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::f, ti::d, ti::a, double, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 3: DataVector; first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::b, ti::a, ti::C, DataVector, Symmetry<2, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // Rank 3: DataVector; first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::i, ti::f, ti::j, DataVector, Symmetry<1, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  // Rank 3: DataVector; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::D, ti::J, ti::I, DataVector, Symmetry<2, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  // Rank 3: DataVector; symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, ti::f, ti::d, ti::a, DataVector, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
}
