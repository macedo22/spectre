// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank012",
                  "[DataStructures][Unit]") {
  // Rank 0
  TestHelpers::tenex::test_evaluate<true>();

  // Rank 1: spacetime
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>, ti::a>();
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>, ti::b>();
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>, ti::A>();
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpacetimeIndex<1, UpLo::Up, Frame::Grid>>, ti::B>();

  // Rank 1: spatial
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>>, ti::i>();
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>, ti::j>();
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpatialIndex<1, UpLo::Up, Frame::Inertial>>, ti::I>();
  TestHelpers::tenex::test_evaluate<
      true, index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>>, ti::J>();

  // Rank 2: nonsymmetric, spacetime only
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::b>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>,
      ti::D, ti::C>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>,
      ti::e, ti::F>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::G, ti::b>();

  // Rank 2: nonsymmetric, spatial only
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::ElementLogical>>,
      ti::j, ti::i>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Up, Frame::Grid>>,
      ti::I, ti::J>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>,
      ti::k, ti::M>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::M, ti::k>();

  // Rank 2: nonsymmetric, spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::c, ti::I>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::A, ti::i>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::J, ti::C>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::e, ti::m>();

  // Rank 2: symmetric, spacetime
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::d>();
  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::d, Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>,
      ti::G, ti::B>();
   TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>,
      ti::G, ti::B, Symmetry<2, 1>>();

  // Rank 2: symmetric, spatial
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::i>();
  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::i, Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate<
      true, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      ti::I, ti::J>();
  TestHelpers::tenex::test_evaluate<
      false, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      ti::I, ti::J, Symmetry<2, 1>>();
}
