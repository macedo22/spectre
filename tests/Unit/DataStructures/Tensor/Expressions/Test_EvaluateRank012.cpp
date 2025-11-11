// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
// #include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0.hpp"
// #include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1.hpp"
// #include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank012",
                  "[DataStructures][Unit]") {
  // TODO : remove this
  TestHelpers::tenex::test_evaluate<true, double>();
  TestHelpers::tenex::test_evaluate<
      true, double, Symmetry<1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>, ti::a>();
//   TestHelpers::tenex::test_evaluate<
//       true, double, Symmetry<1>,
//       index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>(ti::a);

  //   // Rank 0: double
  //   TestHelpers::tenex::test_evaluate_rank_0<true, double>(-7.31);
  //   // Rank 0: DataVector
  //   TestHelpers::tenex::test_evaluate_rank_0<true, DataVector>(
  //       DataVector{-3.1, 9.4, 0.0, -3.1, 2.4, 9.8});

  // TODO : replace these with calls to generic rank test_evaluate
  // Rank 0: double
  TestHelpers::tenex::test_evaluate_rank_0<true, double>();
  // Rank 0: DataVector
  TestHelpers::tenex::test_evaluate_rank_0<true, DataVector>();

  // Rank 1: spacetime
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::a, index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::b, index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::A, index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::B, index_list<SpacetimeIndex<1, UpLo::Up, Frame::Grid>>>();

  // Rank 1: spatial
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::i, index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::j, index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::I, index_list<SpatialIndex<1, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_1<
      true, ti::J, index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spacetime only
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::a, ti::b, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::D, ti::C, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::e, ti::F, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::G, ti::b, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spatial only
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::j, ti::i, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::ElementLogical>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::I, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::k, ti::M, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::M, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::c, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::A, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::J, ti::C, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::e, ti::m, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: symmetric, spacetime
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::a, ti::d, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::G, ti::B, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>();

  // Rank 2: symmetric, spatial
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::j, ti::i, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2<
      true, ti::I, ti::J, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
}
