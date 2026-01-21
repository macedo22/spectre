// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// template <IndexType... Is>
// using indextype_list = tmpl::integral_list<IndexType, Is...>;

// template <IndexType Index, typename Fr = Frame::Inertial>
// using indextype = tmpl::pair<tmpl::integral_constant<IndexType, Index>, Fr>;

template <IndexType Index, typename Fr = Frame::Inertial>
struct indextype_and_frame {
  static constexpr IndexType indextype = Index;
  using frame = Fr;
};

template <typename Fr = Frame::Inertial>
using spatial_index = indextype_and_frame<IndexType::Spatial, Fr>;

template <typename Fr = Frame::Inertial>
using spacetime_index = indextype_and_frame<IndexType::Spacetime, Fr>;

template <typename... Indices>
using indextype_list = tmpl::list<Indices...>;
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

  //   // TODO : put in some section that makes sense
  //   const IndexType spatial_index = IndexType::Spatial;
  //   const IndexType spacetime_index = IndexType::Spacetime;

  using test_index = spatial_index<>;

  using indexlist_1 = indextype_list<spatial_index<>>;

  TestHelpers::tenex::test_evaluate_suite<
      true, ti::J, ti::C, Symmetry<2, 1>,
      indextype_list<spatial_index, spacetime_index>>();
  TestHelpers::tenex::test_evaluate_suite<
      true, ti::J, ti::I, Symmetry<1, 1>,
      indextype_list<spatial_index, spatial_index>>();
}
