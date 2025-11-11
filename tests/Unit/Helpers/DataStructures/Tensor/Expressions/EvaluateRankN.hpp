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
          auto&... TensorIndices, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  constexpr size_t rank = sizeof...(TensorIndices);
  static_assert(rank <= 4, "`test_evaluate` is only implemented for rank <= 4");

  if constexpr (rank == 0) {
    test_evaluate_rank_0<ReturnLhsTensor, DataType>();
  } else if constexpr (rank == 1) {
    test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
  } else if constexpr (rank == 2) {
    test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsTensorIndexTypeList, LhsSymmetry,
                              LhsTensorIndexTypeList>();
  } else if constexpr (rank == 3) {
    test_evaluate_rank_3_core<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsTensorIndexTypeList, LhsSymmetry,
                              LhsTensorIndexTypeList>();
  } else if constexpr (rank == 4) {
    test_evaluate_rank_4_core<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsTensorIndexTypeList, LhsSymmetry,
                              LhsTensorIndexTypeList>();
  } else {
    ERROR("Unsupported rank");
  }
}
}  // namespace TestHelpers::tenex
