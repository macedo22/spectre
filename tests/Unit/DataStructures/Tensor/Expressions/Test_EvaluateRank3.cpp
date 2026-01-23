// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRankN.hpp"

namespace {
// \brief Test evaluation of rank 3 tensors
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_3() {
  // nonsymmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::d, ti::A, ti::i, DataType, Symmetry<3, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<1, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // first and second indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::b, ti::a, ti::C, DataType, Symmetry<2, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::b, ti::a, ti::C, DataType, Symmetry<2, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<3, 2, 1>>();

  // first and third indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::f, ti::j, DataType, Symmetry<1, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::f, ti::j, DataType, Symmetry<1, 2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>,
      Symmetry<3, 2, 1>>();

  // second and third indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::M, ti::I, DataType, Symmetry<2, 1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::M, ti::I, DataType, Symmetry<2, 1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      Symmetry<3, 2, 1>>();

  // symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 2, 1>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<1, 2, 1>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<2, 1, 1>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      Symmetry<3, 2, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank3",
                  "[DataStructures][Unit]") {
  test_evaluate_rank_3<double>();
  test_evaluate_rank_3<DataVector>();
}
