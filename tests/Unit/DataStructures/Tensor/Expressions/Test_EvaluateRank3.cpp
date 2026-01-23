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

// \brief Test evaluation of rank 3 tensors
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_3() {
  // nonsymmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::d, ti::A, ti::i, DataType, Symmetry<3, 2, 1>,
      indextype_list<spacetime_index, spacetime_index, spatial_index>,
      Frame::Inertial>();

  // first and second indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::b, ti::a, ti::C, DataType, Symmetry<2, 2, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate<
      false, ti::b, ti::a, ti::C, DataType, Symmetry<2, 2, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Grid, Symmetry<3, 2, 1>>();

  // first and third indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::i, ti::f, ti::j, DataType, Symmetry<1, 2, 1>,
      indextype_list<spatial_index, spacetime_index, spatial_index>,
      Frame::Grid>();
  TestHelpers::tenex::test_evaluate<
      false, ti::i, ti::f, ti::j, DataType, Symmetry<1, 2, 1>,
      indextype_list<spatial_index, spacetime_index, spatial_index>,
      Frame::Inertial, Symmetry<3, 2, 1>>();

  // second and third indices symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::J, ti::M, ti::I, DataType, Symmetry<2, 1, 1>,
      indextype_list<spatial_index, spacetime_index, spatial_index>,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate<
      false, ti::J, ti::M, ti::I, DataType, Symmetry<2, 1, 1>,
      indextype_list<spacetime_index, spatial_index, spacetime_index>,
      Frame::Grid, Symmetry<3, 2, 1>>();

  // symmetric
  TestHelpers::tenex::test_evaluate<
      true, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Inertial>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Grid, Symmetry<2, 2, 1>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Inertial, Symmetry<1, 2, 1>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Grid, Symmetry<2, 1, 1>>();
  TestHelpers::tenex::test_evaluate<
      false, ti::f, ti::d, ti::a, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>,
      Frame::Distorted, Symmetry<3, 2, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank3",
                  "[DataStructures][Unit]") {
  test_evaluate_rank_3<double>();
  test_evaluate_rank_3<DataVector>();
}
