// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 4 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, `TensorIndexC`, and  `TensorIndexD`
/// can be any type of TensorIndex and are not necessarily `ti_a_t`, `ti_b_t`,
/// `ti_c_t`, and `ti_d_t`. The "A", "B", "C", and "D" suffixes just denote the
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
/// \tparam Datatype the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the type of the first TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_a_t`
/// \tparam TensorIndexB the type of the second TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_B_t`
/// \tparam TensorIndexC the type of the third TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_c_t`
/// \tparam TensorIndexD the type of the fourth TensorIndex used on the RHS of
/// the TensorExpression, e.g. `ti_D_t`
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \param tensorindex_c the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
/// \param tensorindex_d the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_D`
template <typename Datatype, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename TensorIndexA,
          typename TensorIndexB, typename TensorIndexC, typename TensorIndexD>
void test_evaluate_rank_4(
    const TensorIndexA& tensorindex_a, const TensorIndexB& tensorindex_b,
    const TensorIndexC& tensorindex_c, const TensorIndexD& tensorindex_d) {
  Tensor<Datatype, RhsSymmetry, RhsTensorIndexTypeList> rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);

  size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;
  size_t dim_d = tmpl::at_c<RhsTensorIndexTypeList, 3>::dim;

  // L_{abcd} = R_{abcd}
  auto abcd_to_abcd =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{abdc} = R_{abcd}
  auto abcd_to_abdc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexD,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{acbd} = R_{abcd}
  auto abcd_to_acbd =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{acdb} = R_{abcd}
  auto abcd_to_acdb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexD,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{adbc} = R_{abcd}
  auto abcd_to_adbc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexB,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{adcb} = R_{abcd}
  auto abcd_to_adcb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexC,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{bacd} = R_{abcd}
  auto abcd_to_bacd =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{badc} = R_{abcd}
  auto abcd_to_badc =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexD,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{bcad} = R_{abcd}
  auto abcd_to_bcad =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{bcda} = R_{abcd}
  auto abcd_to_bcda =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexD,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{bdac} = R_{abcd}
  auto abcd_to_bdac =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexA,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{bdca} = R_{abcd}
  auto abcd_to_bdca =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexC,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{cabd} = R_{abcd}
  auto abcd_to_cabd =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{cadb} = R_{abcd}
  auto abcd_to_cadb =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexD,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{cbad} = R_{abcd}
  auto abcd_to_cbad =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA,
                                  TensorIndexD>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{cbda} = R_{abcd}
  auto abcd_to_cbda =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexD,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{cdab} = R_{abcd}
  auto abcd_to_cdab =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexA,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{cdba} = R_{abcd}
  auto abcd_to_cdba =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexB,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{dabc} = R_{abcd}
  auto abcd_to_dabc =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexB,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{dacb} = R_{abcd}
  auto abcd_to_dacb =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexC,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{dbac} = R_{abcd}
  auto abcd_to_dbac =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexA,
                                  TensorIndexC>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{dbca} = R_{abcd}
  auto abcd_to_dbca =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexC,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{dcab} = R_{abcd}
  auto abcd_to_dcab =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexA,
                                  TensorIndexB>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  // L_{dcba} = R_{abcd}
  auto abcd_to_dcba =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexB,
                                  TensorIndexA>(
          rhs_tensor(tensorindex_a, tensorindex_b, tensorindex_c,
                     tensorindex_d));

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
          for (size_t l = 0; l < dim_d; ++l) {
            CHECK(abcd_to_abcd.get(i, j, k, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_abdc.get(i, j, l, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_acbd.get(i, k, j, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_acdb.get(i, k, l, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_adbc.get(i, l, j, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_adcb.get(i, l, k, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bacd.get(j, i, k, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_badc.get(j, i, l, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bcad.get(j, k, i, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bcda.get(j, k, l, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bdac.get(j, l, i, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_bdca.get(j, l, k, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cabd.get(k, i, j, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cadb.get(k, i, l, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cbad.get(k, j, i, l) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cbda.get(k, j, l, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cdab.get(k, l, i, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_cdba.get(k, l, j, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dabc.get(l, i, j, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dacb.get(l, i, k, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dbac.get(l, j, i, k) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dbca.get(l, j, k, i) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dcab.get(l, k, i, j) == rhs_tensor.get(i, j, k, l));
            CHECK(abcd_to_dcba.get(l, k, j, i) == rhs_tensor.get(i, j, k, l));
        }
      }
    }
  }
}
