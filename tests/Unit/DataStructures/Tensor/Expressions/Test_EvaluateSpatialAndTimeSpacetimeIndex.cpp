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

  const auto R_abcd =
      make_with_random_values<Tensor<DataType, symm_1111, index_list_abcd>>(
          make_not_null(&generator), distribution, used_for_size);
  auto expected_L_abcd =
      make_with_value<Tensor<DataType, symm_1111, index_list_abcd>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  for (size_t a = 0; a < tmpl::at_c<index_list_abcd, 0>::dim; a++) {
    for (size_t j = 1; j < tmpl::at_c<index_list_abcd, 1>::dim; j++) {
      for (size_t i = 1; i < tmpl::at_c<index_list_abcd, 2>::dim; i++) {
        expected_L_abcd.get(0, a, j, i) = R_abcd.get(0, a, j, i);
      }
    }
  }
  TestHelpers::tenex::test_evaluate_rank_4_impl<false, ti::t, ti::a, ti::j,
                                                ti::i>(expected_L_abcd, R_abcd);
}
