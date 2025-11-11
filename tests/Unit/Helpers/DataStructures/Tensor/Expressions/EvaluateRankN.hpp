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
template <bool ReturnLhsTensor, typename DataType>
void test_evaluate() {
  test_evaluate_rank_0_core<ReturnLhsTensor, DataType>();
}

template <bool ReturnLhsTensor, typename DataType,
          typename RhsTensorIndexTypeList, auto& TensorIndex,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndex, DataType,
                            RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
}

template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto&... TensorIndices,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  constexpr size_t rank = 2 + sizeof...(TensorIndices);
  static_assert(rank <= 4, "`test_evaluate` is only implemented for rank <= 4");

  // rank == 0 and rank == 1 handled by other test_evaluate overloads
  if constexpr (rank == 2) {
    test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              TensorIndices..., DataType, RhsSymmetry,
                              RhsTensorIndexTypeList, LhsSymmetry,
                              LhsTensorIndexTypeList>();
  } else if constexpr (rank == 3) {
    test_evaluate_rank_3_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              TensorIndices..., DataType, RhsSymmetry,
                              RhsTensorIndexTypeList, LhsSymmetry,
                              LhsTensorIndexTypeList>();
  } else if constexpr (rank == 4) {
    test_evaluate_rank_4_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              TensorIndices..., DataType, RhsSymmetry,
                              RhsTensorIndexTypeList, LhsSymmetry,
                              LhsTensorIndexTypeList>();
  } else {
    ERROR("Unsupported rank");
  }
}

template <bool ReturnLhsTensor, typename DataType,
          typename RhsSymmetry = tmpl::list<>,
          typename RhsTensorIndexTypeList = tmpl::list<>,
          auto&... TensorIndices,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_symmetry_cases() {
  constexpr size_t rank = sizeof...(TensorIndices);
  static_assert(rank <= 3, "`test_evaluate` is only implemented for rank <= 3");
  // Checking input tparam rank is consistent at this point since below logic
  // for rank 0 ignores symmetry and tensor index type tparams and rank 1
  // ignores the symmetry tparam. This way, someone can
  // static_assert(tmpl::size<RhsSymmetry>::value == rank,
  //               "RHS symmetry list length not equal to rank");
  // static_assert(tmpl::size<RhsTensorIndexTypeList>::value == rank,
  //               "RHS index list length not equal to rank");
  // static_assert(tmpl::size<LhsTensorIndexTypeList>::value == rank,
  //               "LHS index list length not equal to rank");

  // rank = 0 and rank == 1 don't have different symmetry cases
  // TODO maybe move this above comment into function docs because it should be
  // there for the user anyway?
  if constexpr (rank == 0) {
    // input validation since these tparams are not forwarded to the core test
    // function
    static_assert(tmpl::size<RhsSymmetry>::value == rank,
                  "Symmetry for RHS scalar should be an empty list");
    static_assert(
        tmpl::size<RhsTensorIndexTypeList>::value == rank,
        "TensorIndexType list for RHS scalar should be an empty list");
    static_assert(
        tmpl::size<LhsTensorIndexTypeList>::value == rank,
        "TensorIndexType list for LHS scalar should be an empty list");
    test_evaluate_rank_0_core<ReturnLhsTensor, DataType>();
  } else if constexpr (rank == 1) {
    // input validation since these tparams are not forwarded to the core test
    // function
    static_assert(tmpl::size<RhsSymmetry>::value == rank,
                  "Symmetry for RHS vector should be a list of length 1");
    test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
  } else if constexpr (rank == 2) {
    test_evaluate_rank_2_impl<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsSymmetry, RhsTensorIndexTypeList,
                              LhsTensorIndexTypeList>();
  } else if constexpr (rank == 3) {
    test_evaluate_rank_3_impl<ReturnLhsTensor, TensorIndices..., DataType,
                              RhsSymmetry, RhsTensorIndexTypeList,
                              LhsTensorIndexTypeList>();
  } else {
    ERROR("Unsupported rank");
  }
}

// // TODO : add in docs that this doesn't support different ranks on either
// side
// // of the eq
// template <bool ReturnLhsTensor, typename DataType,
//           typename RhsSymmetry = tmpl::list<>,
//           typename RhsTensorIndexTypeList = tmpl::list<>,
//           auto&... TensorIndices, typename LhsSymmetry = RhsSymmetry,
//           typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
// void test_evaluate() {
//   constexpr size_t rank = sizeof...(TensorIndices);
//   static_assert(rank <= 4, "`test_evaluate` is only implemented for rank <=
//   4");

//   if constexpr (rank == 0) {
//     test_evaluate_rank_0<ReturnLhsTensor, DataType>();
//   } else if constexpr (rank == 1) {
//     test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList,
//                               LhsTensorIndexTypeList>();
//   } else if constexpr (rank == 2) {
//     test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList, LhsSymmetry,
//                               LhsTensorIndexTypeList>();
//   } else if constexpr (rank == 3) {
//     test_evaluate_rank_3_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList, LhsSymmetry,
//                               LhsTensorIndexTypeList>();
//   } else if constexpr (rank == 4) {
//     test_evaluate_rank_4_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList, LhsSymmetry,
//                               LhsTensorIndexTypeList>();
//   } else {
//     ERROR("Unsupported rank");
//   }
// }

// template <bool ReturnLhsTensor, typename DataType,
//           typename RhsSymmetry = tmpl::list<>,
//           typename RhsTensorIndexTypeList = tmpl::list<>,
//           auto&... TensorIndices,
//           typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
// void test_evaluate() {
//   constexpr size_t rank = sizeof...(TensorIndices);
//   static_assert(rank <= 4, "`test_evaluate` is only implemented for rank <=
//   4");

//   if constexpr (rank == 0) {
//     test_evaluate_rank_0<ReturnLhsTensor, DataType>();
//   } else if constexpr (rank == 1) {
//     test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList,
//                               LhsTensorIndexTypeList>();
//   } else if constexpr (rank == 2) {
//     test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList, LhsSymmetry,
//                               LhsTensorIndexTypeList>();
//   } else if constexpr (rank == 3) {
//     test_evaluate_rank_3_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList, LhsSymmetry,
//                               LhsTensorIndexTypeList>();
//   } else if constexpr (rank == 4) {
//     test_evaluate_rank_4_core<ReturnLhsTensor, TensorIndices..., DataType,
//                               RhsTensorIndexTypeList, LhsSymmetry,
//                               LhsTensorIndexTypeList>();
//   } else {
//     ERROR("Unsupported rank");
//   }
// }
}  // namespace TestHelpers::tenex
