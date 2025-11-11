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
// TODO : add in docs that this doesn't support different ranks on either
// side

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

// TODO : consider including default params for symm and tensor index type
// (same for the rank 1 overload below) to have a common interface.
// TODO : consider moving these to RankX.hpp files or those functions here
template <bool ReturnLhsTensor, typename DataType>
void test_evaluate_symmetry_cases() {
  test_evaluate_rank_0_core<ReturnLhsTensor, TensorIndex>();
}

template <bool ReturnLhsTensor, typename DataType,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_symmetry_cases() {
  test_evaluate_rank_1_core<ReturnLhsTensor, TensorIndex, DataType,
                            RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
}

template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_symmetry_cases() {
  test_evaluate_rank_2_impl<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                            DataType, RhsSymmetry, RhsTensorIndexTypeList,
                            LhsTensorIndexTypeList>();
}

template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_symmetry_cases() {
  test_evaluate_rank_3_impl<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                            TensorIndexC, DataType, RhsSymmetry,
                            RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
}

// TODO : should we have symm and tensor index type list tparams for rank 0 and
// rank 1 to have an consistent interface or nah? should we even have these
// definitions if they don't make sense to have? this rank 0 interface is
// maybe misleading or confusing because it's not clearly rank 0, so maybe
// we should add the tparams and assert that they are correct or use
// SFINAE/Requires for restricting template matching
template <bool ReturnLhsTensor>
void test_evaluate_datatype_and_symmetry_cases() {
  test_evaluate_rank_0<ReturnLhsTensor>();
}

template <bool ReturnLhsTensor, auto& TensorIndex,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_datatype_and_symmetry_cases() {
  test_evaluate_rank_1<ReturnLhsTensor, TensorIndex, RhsTensorIndexTypeList,
                       LhsTensorIndexTypeList>();
}

template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_datatype_and_symmetry_cases() {
  test_evaluate_rank_2<ReturnLhsTensor, TensorIndexA, TensorIndexB, RhsSymmetry,
                       RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
}

template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename RhsSymmetry,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_datatype_and_symmetry_cases() {
  test_evaluate_rank_3<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                       TensorIndexC, RhsSymmetry, RhsTensorIndexTypeList,
                       LhsTensorIndexTypeList>();
}

// TODO : same interface question lol
template <bool ReturnLhsTensor>
void test_evaluate_datatype_cases() {
  test_evaluate_rank_0<ReturnLhsTensor>();
}

template <bool ReturnLhsTensor, auto& TensorIndex,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_datatype_cases() {
  test_evaluate_rank_1<ReturnLhsTensor, TensorIndex, RhsTensorIndexTypeList,
                       LhsTensorIndexTypeList>();
}

template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, auto& TensorIndexD, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_datatype_cases() {
  test_evaluate_rank_4<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                       TensorIndexC, TensorIndexD, double, RhsSymmetry,
                       RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
}
}  // namespace TestHelpers::tenex
