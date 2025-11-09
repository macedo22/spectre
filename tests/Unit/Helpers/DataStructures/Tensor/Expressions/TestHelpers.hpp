// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::tenex {
template <bool ReturnLhsTensor, auto&... LhsTensorIndices, typename LhsTensor,
          typename RhsExpression>
void call_evaluate(const gsl::not_null<LhsTensor*> lhs_tensor,
                   const RhsExpression& rhs_expression) {
  if constexpr (ReturnLhsTensor) {
    *lhs_tensor = ::tenex::evaluate<LhsTensorIndices...>(rhs_expression);
  } else {
    ::tenex::evaluate<LhsTensorIndices...>(lhs_tensor, rhs_expression);
  }
}

template <typename Index, auto& TensorIndex>
constexpr std::pair<size_t, size_t> get_index_value_range() {
  constexpr bool tensorindex_is_time =
      ::tenex::detail::is_time_index_value(TensorIndex.value);
  static_assert(
      not(Index::index_type == IndexType::Spatial and tensorindex_is_time),
      "Cannot use a concrete time TensorIndex with a SpatialIndex.");
  std::pair<size_t, size_t> range{};
  range.first =
      Index::index_type == IndexType::Spacetime and not TensorIndex.is_spacetime
          ? 1
          : 0;
  range.second = tensorindex_is_time ? 0 : Index::dim - 1;
  return range;
}

// TODO : add in docs that this doesn't support different ranks on either side
// of the eq
template <bool ReturnLhsTensor, typename DataType,
          typename RhsSymmetry = tmpl::list<>,
          typename RhsTensorIndexTypeList = tmpl::list<>,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList,
          typename... TensorIndices>
void test_evaluate(const TensorIndices&... /*meta*/) {
  constexpr size_t num_indices = tmpl::size<RhsSymmetry>::value;
  static_assert(tmpl::size<LhsSymmetry>::value == num_indices,
                "LHS and RHS symmetry lists are not the same length");
  static_assert(tmpl::size<RhsTensorIndexTypeList>::value == num_indices,
                "RHS index list is not the same length as the RHS symmetry");
  static_assert(tmpl::size<LhsTensorIndexTypeList>::value ==
                    tmpl::size<RhsTensorIndexTypeList>::value,
                "LHS index list is not the same length as the RHS index list");
  static_assert(num_indices <= 4,
                "`test_evaluate` is only implemented for rank <= 4");

  if constexpr (num_indices == 0) {
  } else if constexpr (num_indices == 1) {
  } else if constexpr (num_indices == 2) {
  } else if constexpr (num_indices == 3) {
  } else if constexpr (num_indices == 4) {
  } else {
    ERROR("Unsupported rank");
  }
}
}  // namespace TestHelpers::tenex
