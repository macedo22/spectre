// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"
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
/// If `ReturnLhsTensor == true`, the `tenex::evaluate` overload that returns
/// the LHS tensor will be tested. This, in turn, includes testing whether
/// `tenex::evaluate` is deducing the correct LHS tensor return type, where
/// `LhsTensorIndexType` is its expected list of indices.
///
/// If `ReturnLhsTensor == false`, the `tenex::evaluate` overload that takes a
/// preallocated LHS tensor will be tested. In this case, `LhsSymmetry` and
/// `LhsTensorIndexList` can be different from and will override what would be
/// automatically deduced from the RHS tensor expression. This is useful for
/// testing evaluations where the desired LHS tensor type would not
/// automatically be deduced from the RHS expression. For example, given some
/// tensor \f$R_{a}\f$ with one spacetime index, one can test whether
/// \f$R_{i} = ...\f$ correctly only assigns to the spatial components of the
/// tensor.
///
/// \param ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// preallocating the result tensor and filling it)
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti::a`
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndex, typename DataType,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_1_core() {
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
  // TODO : don't size this in case ReturnLhsTensor == false, correct in other
  // files too
  L_a_type L_a(used_for_size);
  std::fill(L_a.begin(), L_a.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndex>(make_not_null(&L_a),
                                              R_a(TensorIndex));

  CHECK(L_a == expected_L_a);  // check LHS evaluated correctly

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<0, R_a_type>,
                         ::Tags::TempTensor<1, L_a_type>>>
        vars(used_for_size, std::numeric_limits<double>::signaling_NaN());

    R_a_type& R_a_temp = get<::Tags::TempTensor<0, R_a_type>>(vars);
    R_a_temp = R_a;

    // L_a = R_a
    L_a_type& L_a_temp = get<::Tags::TempTensor<1, L_a_type>>(vars);
    std::fill(L_a_temp.begin(), L_a_temp.end(),
              component_placeholder_value<DataType>::value);
    // TODO : replace false here and other files
    call_evaluate<false, TensorIndex>(make_not_null(&L_a_temp),
                                      R_a(TensorIndex));

    CHECK(R_a_temp == R_a);           // check RHS wasn't modified
    CHECK(L_a_temp == expected_L_a);  // check LHS evaluated correctly
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 1 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details This simply runs `test_evaluate_rank_1_impl` but for different
/// data types for the tensor components
///
/// \param ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// preallocating the result tensor and filling it)
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti::a`
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndex,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_1() {
  TestHelpers::tenex::test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndex,
                                                double, RhsTensorIndexTypeList,
                                                LhsTensorIndexTypeList>();
  TestHelpers::tenex::test_evaluate_rank_1_core<
      ReturnLhsTensor, TensorIndex, DataVector, RhsTensorIndexTypeList,
      LhsTensorIndexTypeList>();
}
}  // namespace TestHelpers::tenex
