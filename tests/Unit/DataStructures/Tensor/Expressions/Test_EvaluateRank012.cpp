// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"

namespace {
template <IndexType... Is>
using indextype_list = tmpl::integral_list<IndexType, Is...>;
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank012",
                  "[DataStructures][Unit]") {
  // Rank 0
  TestHelpers::tenex::test_evaluate();

  // Rank 1: spacetime
  TestHelpers::tenex::test_evaluate<
      true, ti::a, index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::b, index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::A, index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::B, index_list<SpacetimeIndex<1, UpLo::Up, Frame::Grid>>>();

  // Rank 1: spatial
  TestHelpers::tenex::test_evaluate<
      true, ti::i, index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::j, index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::I, index_list<SpatialIndex<1, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::J, index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spacetime only
  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::b, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::D, ti::C, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::e, ti::F, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::G, ti::b, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spatial only
  TestHelpers::tenex::test_evaluate<
      true, ti::j, ti::i, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::ElementLogical>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::k, ti::M, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::M, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate<
      true, ti::c, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::A, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::C, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::e, ti::m, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: symmetric, spacetime
  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::d, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::d, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::G, ti::B, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::G, ti::B, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>>();

  // Rank 2: symmetric, spatial
  TestHelpers::tenex::test_evaluate<
      true, ti::j, ti::i, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::j, ti::i, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::J, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::J, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>>();

  // TODO : put in some section that makes sense
  const IndexType spatial_index = IndexType::Spatial;
  const IndexType spacetime_index = IndexType::Spacetime;

  // TODOTODOTODO: now run the suite on a few instances that make sense
  // and not just these random two
  // note: going to keep that test_evaluate_suite does all
  // indices Inertial and then all Grid because the point of this
  // test suite is more so for mixing dims not mixing dims and frames.
  // if we want to test mixing frames, then we can do a dedicated test that
  // targets that or just sprinkle it in throughout other tests, which is
  // basically what I have now already. So, just need to call
  // test_evaluate_suite for a handful of cases, but then use
  // the regular test_evaluate to exhaustively test other things?
  // maybe use test suite in regular Test_EvaluateRankX.cpp tests but
  // then in the spatial-spacetime test files, use the suite a
  // handful of times but mostly the regular one since the mixed
  // dim functionality has already been tested hard? or maybe just
  // use the regular one exclusively in spatial-spacetime test files,
  // like I do already, which would mean the only change to make would
  // be to have the Test_EvaluateRankX.cpp files run the suite
  // for all cases (or some)
  TestHelpers::tenex::test_evaluate_suite<
      true, ti::J, ti::C, Symmetry<2, 1>,
      indextype_list<spatial_index, spacetime_index>>();
  TestHelpers::tenex::test_evaluate_suite<
      true, ti::J, ti::I, Symmetry<1, 1>,
      indextype_list<spatial_index, spatial_index>>();
}
