// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

namespace TestHelpers::tenex {
// test evaluation of rank 0 tensor
void test_evaluate();

// test evaluation of rank 1 tensor
template <bool ReturnLhsTensor, auto& TensorIndex,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  test_evaluate_rank_1<ReturnLhsTensor, TensorIndex, double,
                       RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
  test_evaluate_rank_1<ReturnLhsTensor, TensorIndex, DataVector,
                       RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
}

// test evaluation of rank 2 tensor
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  test_evaluate_rank_2<ReturnLhsTensor, TensorIndexA, TensorIndexB, double,
                       RhsSymmetry, RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
  test_evaluate_rank_2<ReturnLhsTensor, TensorIndexA, TensorIndexB, DataVector,
                       RhsSymmetry, RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
}

// test evaluation of rank 3 tensor
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  test_evaluate_rank_3<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                       TensorIndexC, double, RhsSymmetry,
                       RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
  test_evaluate_rank_3<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                       TensorIndexC, DataVector, RhsSymmetry,
                       RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
}

// test evaluation of rank 4 tensor
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, auto& TensorIndexD, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  test_evaluate_rank_4<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                       TensorIndexC, TensorIndexD, double, RhsSymmetry,
                       RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
  test_evaluate_rank_4<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                       TensorIndexC, TensorIndexD, DataVector, RhsSymmetry,
                       RhsTensorIndexTypeList, LhsSymmetry,
                       LhsTensorIndexTypeList>();
}
}  // namespace TestHelpers::tenex
