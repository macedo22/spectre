// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::tenex {
// TODO : add in docs that this doesn't support different ranks on either side
// of the eq
template <bool ReturnLhsTensor, typename DataType,
          typename RhsSymmetry = tmpl::list<>,
          typename RhsTensorIndexTypeList = tmpl::list<>,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList,
          typename... TensorIndices>
void test_evaluate(const TensorIndices&... tensorindices) {
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
    test_evaluate_rank_0<ReturnLhsTensor, DataType>();
  } else if constexpr (num_indices == 1) {
    // TODO : doesn't compile because can't pass the references as tparams
    // test_evaluate_rank_1_core<ReturnLhsTensor, tensorindices..., DataType,
    //                           RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
  } else if constexpr (num_indices == 2) {
  } else if constexpr (num_indices == 3) {
  } else if constexpr (num_indices == 4) {
  } else {
    ERROR("Unsupported rank");
  }
}
}  // namespace TestHelpers::tenex
