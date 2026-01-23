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
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3NonSymmetric",
    "[DataStructures][Unit]") {
  // \brief Test evaluation of rank 3 tensors with no symmetry

  // double
  TestHelpers::tenex::test_evaluate<
      true, ti::d, ti::A, ti::i, double, Symmetry<3, 2, 1>,
      indextype_list<spacetime_index, spacetime_index, spatial_index>,
      Frame::Inertial>();

  // DataVector
  TestHelpers::tenex::test_evaluate<
      true, ti::K, ti::f, ti::m, DataVector, Symmetry<3, 2, 1>,
      indextype_list<spatial_index, spacetime_index, spatial_index>,
      Frame::Grid>();
}
