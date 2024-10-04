// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression."
    "EvaluateSpatialAndTimeSpacetimeIndex",
    "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  using DataType = DataVector;
  using FrameType = Frame::Inertial;
  const DataType used_for_size(3, std::numeric_limits<double>::signaling_NaN());

  using symm_1111 = Symmetry<1, 1, 1, 1>;
  using index_list_abcd = index_list<SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>,
                                     SpacetimeIndex<3, UpLo::Lo, FrameType>>;

  TestHelpers::tenex::test_evaluate_rank_4<false, ti::t, ti::a, ti::j, ti::i,
                                           DataVector, symm_1111,
                                           index_list_abcd>();
}
