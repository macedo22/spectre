// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank4",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  // Rank 4: double; nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, double, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::b, ti::A, ti::k, ti::l>(make_not_null(&generator));

  // Rank 4: double; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, double, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      ti::G, ti::d, ti::a, ti::j>(make_not_null(&generator));

  // Rank 4: double; first, second, and fourth indices symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, double, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::i, ti::k, ti::l>(make_not_null(&generator));

  // Rank 4: double; symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, double, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::F, ti::A, ti::C, ti::D>(make_not_null(&generator));

  // Rank 4: DataVector; nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataVector, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::b, ti::A, ti::k, ti::l>(make_not_null(&generator));

  // Rank 4: DataVector; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataVector, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      ti::G, ti::d, ti::a, ti::j>(make_not_null(&generator));

  // Rank 4: DataVector; first, second, and fourth indices symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataVector, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::i, ti::k, ti::l>(make_not_null(&generator));

  // Rank 4: DataVector; symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataVector, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::F, ti::A, ti::C, ti::D>(make_not_null(&generator));
}
