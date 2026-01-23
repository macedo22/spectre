// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"

namespace {
template <IndexType... Is>
using indextype_list = tmpl::integral_list<IndexType, Is...>;

const IndexType spatial_index = IndexType::Spatial;
const IndexType spacetime_index = IndexType::Spacetime;

// \brief Test evaluation of rank 0, rank 1, and rank 2 tensors
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_012() {
  // Rank 0
  TestHelpers::tenex::test_evaluate<DataType>();

  // Rank 1: spacetime
  TestHelpers::tenex::test_evaluate<true, ti::a, DataType,
                                    indextype_list<spacetime_index>,
                                    Frame::Inertial>();
  TestHelpers::tenex::test_evaluate<
      true, ti::b, DataType, indextype_list<spacetime_index>, Frame::Grid>();
  TestHelpers::tenex::test_evaluate<true, ti::A, DataType,
                                    indextype_list<spacetime_index>,
                                    Frame::Inertial>();
  TestHelpers::tenex::test_evaluate<
      true, ti::B, DataType, indextype_list<spacetime_index>, Frame::Grid>();

  // Rank 1: spatial
  TestHelpers::tenex::test_evaluate<
      true, ti::i, DataType, indextype_list<spatial_index>, Frame::Grid>();
  TestHelpers::tenex::test_evaluate<
      true, ti::j, DataType, indextype_list<spatial_index>, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate<
      true, ti::I, DataType, indextype_list<spatial_index>, Frame::Grid>();
  TestHelpers::tenex::test_evaluate<
      true, ti::J, DataType, indextype_list<spatial_index>, Frame::Inertial>();

  // Rank 2: nonsymmetric, spacetime only
  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::b, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::D, ti::C, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::e, ti::F, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::G, ti::b, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spatial only
  TestHelpers::tenex::test_evaluate<
      true, ti::j, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::ElementLogical>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::J, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::k, ti::M, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::M, ti::k, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: nonsymmetric, spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate<
      true, ti::c, ti::I, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::A, ti::i, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::C, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::e, ti::m, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // Rank 2: symmetric, spacetime
  TestHelpers::tenex::test_evaluate<
      true, ti::a, ti::d, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::a, ti::d, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::G, ti::B, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::G, ti::B, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>>();

  // Rank 2: symmetric, spatial
  TestHelpers::tenex::test_evaluate<
      true, ti::j, ti::i, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::j, ti::i, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      Symmetry<2, 1>>();
  TestHelpers::tenex::test_evaluate<
      true, ti::I, ti::J, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::I, ti::J, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      Symmetry<2, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank012",
                  "[DataStructures][Unit]") {
  test_evaluate_rank_012<double>();
  test_evaluate_rank_012<DataVector>();
}
