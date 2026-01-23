// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

namespace TestHelpers::tenex {
/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 0 tensor correctly assigns the data to the evaluated left hand
/// side tensor
void test_evaluate();

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 1 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details See `test_evaluate` rank 4 function template for general details
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// assigning to the result tensor passed in as an argument)
/// \tparam TensorIndex the TensorIndex used in the the TensorExpression,
/// e.g. `ti::a`
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndex,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  test_evaluate_rank_1<ReturnLhsTensor, TensorIndex, double,
                       RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
  test_evaluate_rank_1<ReturnLhsTensor, TensorIndex, DataVector,
                       RhsTensorIndexTypeList, LhsTensorIndexTypeList>();
}

// TODO : update docs for TensorIndexTypeList tparams since now they can also be
// a list of IndexType
/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 2 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details See `test_evaluate` rank 4 function template for general details
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// assigning to the result tensor passed in as an argument)
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate() {
  if constexpr (std::is_same_v<
                    typename tmpl::at_c<RhsTensorIndexTypeList, 0>::value_type,
                    IndexType>) {
    test_evaluate_rank_2<ReturnLhsTensor, TensorIndexA, TensorIndexB, double,
                         RhsSymmetry, RhsTensorIndexTypeList, LhsSymmetry,
                         LhsTensorIndexTypeList>();
    test_evaluate_rank_2<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                         DataVector, RhsSymmetry, RhsTensorIndexTypeList,
                         LhsSymmetry, LhsTensorIndexTypeList>();
  } else {
    test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              double, RhsSymmetry, RhsTensorIndexTypeList,
                              LhsSymmetry, LhsTensorIndexTypeList>();
    test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              DataVector, RhsSymmetry, RhsTensorIndexTypeList,
                              LhsSymmetry, LhsTensorIndexTypeList>();
    test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              double, RhsSymmetry, RhsTensorIndexTypeList,
                              LhsSymmetry, LhsTensorIndexTypeList>();
    test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                              DataVector, RhsSymmetry, RhsTensorIndexTypeList,
                              LhsSymmetry, LhsTensorIndexTypeList>();
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 3 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details See `test_evaluate` rank 4 function template for general details
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// assigning to the result tensor passed in as an argument)
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
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

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 4 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, `TensorIndexC`, and  `TensorIndexD`
/// can be any type of TensorIndex and are not necessarily `ti::a`, `ti::b`,
/// `ti::c`, and `ti::d`. The "A", "B", "C", and "D" suffixes just denote the
/// ordering of the generic indices of the RHS tensor expression. In the RHS
/// tensor expression, it means `TensorIndexA` is the first index used,
/// `TensorIndexB` is the second index used, `TensorIndexC` is the third index
/// used, and `TensorIndexD` is the fourth index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b, c, d), then
/// this test checks that the data in the evaluated LHS tensor is correct
/// according to the index orders of the LHS and RHS. The possible cases that
/// are checked are when the LHS tensor is evaluated with index orders of all 24
/// permutations of (a, b, c, d), e.g. (a, b, d, c), (a, c, b, d), ...
///
/// If `ReturnLhsTensor == true`, the `tenex::evaluate` overload that returns
/// the LHS tensor will be tested. This, in turn, includes testing whether
/// `tenex::evaluate` is deducing the correct LHS tensor return type, where
/// `LhsSymmetry` is its expected symmetry and `LhsTensorIndexType` is its
/// expected list of indices.
///
/// If `ReturnLhsTensor == false`, the `tenex::evaluate` overload that takes a
/// LHS tensor as an argument will be tested. In this case, `LhsSymmetry` and
/// `LhsTensorIndexList` can be different from and will override what would be
/// automatically deduced from the RHS tensor expression. This is useful for
/// testing evaluations where the desired LHS tensor type would not
/// automatically be deduced from the RHS expression. For example, given some
/// tensor \f$R_{abcd}\f$ with four spacetime indices, one can test whether
/// \f$R_{ijkl} = ...\f$ correctly only assigns to the spatial-spatial
/// components of the tensor. Likewise, `ReturnLhsTensor == false` is
/// necessary to test cases where the LHS symmetry is different from what
/// would be deduced.
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// assigning to the result tensor passed in as an argument)
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
/// \tparam TensorIndexD the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::D`
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
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
