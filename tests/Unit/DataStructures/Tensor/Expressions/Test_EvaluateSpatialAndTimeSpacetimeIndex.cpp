// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression."
    "EvaluateSpatialAndTimeSpacetimeIndex",
    "[DataStructures][Unit]") {
  using FrameType = Frame::Inertial;
  using symm_1111 = Symmetry<1, 1, 1, 1>;
  using index_list_abcd = index_list<SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>>;

  TestHelpers::tenex::test_evaluate_rank_4<false, ti::t, ti::a, ti::j, ti::i,
                                           DataVector, symm_1111,
                                           index_list_abcd>();
}
