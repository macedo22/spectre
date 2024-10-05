// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <random>
#include <type_traits>
#include <utility>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"  // TODO : remove after factoring out
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {
// TODO : update testing func docs

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 1 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam TensorIndexTypeList the Tensors' typelist containing their
/// \ref SpacetimeIndex "TensorIndexType"
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti::a`
template <bool ReturnLhsTensor, auto& TensorIndex, typename DataType,
          typename LhsTensorIndexTypeList,
          typename RhsTensorIndexTypeList = LhsTensorIndexTypeList>
void test_evaluate_rank_1() {
  using symmetry = Symmetry<1>;
  using L_a_type = Tensor<DataType, symmetry, LhsTensorIndexTypeList>;
  using R_a_type = Tensor<DataType, symmetry, RhsTensorIndexTypeList>;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  const auto R_a = make_with_random_values<R_a_type>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_a =
      ReturnLhsTensor
          ? L_a_type{}
          : make_with_value<L_a_type>(
                used_for_size, component_placeholder_value<DataType>::value);

  using lhs_tensorindextype = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype = tmpl::at_c<RhsTensorIndexTypeList, 0>;

  const std::pair<size_t, size_t> lhs_index_value_range =
      get_index_value_range<lhs_tensorindextype, TensorIndex>();
  const std::pair<size_t, size_t> rhs_index_value_range =
      get_index_value_range<rhs_tensorindextype, TensorIndex>();

  for (size_t lhs_a = lhs_index_value_range.first,
              rhs_a = rhs_index_value_range.first;
       lhs_a <= lhs_index_value_range.second; lhs_a++, rhs_a++) {
    expected_L_a.get(lhs_a) = R_a.get(rhs_a);
  }

  // L_a = R_a
  // Use explicit type (vs auto) so the compiler checks return type of
  // `evaluate`
  L_a_type L_a(used_for_size);
  std::fill(L_a.begin(), L_a.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndex>(make_not_null(&L_a),
                                              R_a(TensorIndex));

  const size_t dim = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;

  for (size_t lhs_a = 0; lhs_a < dim; ++lhs_a) {
    CHECK(L_a.get(lhs_a) == expected_L_a.get(lhs_a));
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    // L_a = R_a
    Variables<tmpl::list<::Tags::TempTensor<1, L_a_type>>> L_a_var{
        used_for_size};
    L_a_type& L_a_temp = get<::Tags::TempTensor<1, L_a_type>>(L_a_var);
    std::fill(L_a_temp.begin(), L_a_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<false, TensorIndex>(make_not_null(&L_a_temp),
                                      R_a(TensorIndex));

    for (size_t lhs_a = 0; lhs_a < dim; ++lhs_a) {
      CHECK(L_a_temp.get(lhs_a) == expected_L_a.get(lhs_a));
    }
  }
}
}  // namespace TestHelpers::tenex
