// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"

namespace {
// \brief Test evaluation of rank 4 tensors
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_4() {
  // nonsymmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::b, ti::A, ti::k, ti::l, DataType, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();

  // second and third indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::G, ti::d, ti::a, ti::j, DataType, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>>();

  // first, second, and fourth indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::j, ti::i, ti::k, ti::l, DataType, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::F, ti::A, ti::C, ti::D, DataType, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  // different LHS symmetry
  TestHelpers::tenex::test_evaluate<
      false, ti::j, ti::i, ti::k, ti::l, DataType, Symmetry<2, 1, 2, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      Symmetry<3, 2, 1, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank4",
                  "[DataStructures][Unit]") {
  test_evaluate_rank_4<double>();
  test_evaluate_rank_4<DataVector>();
}
