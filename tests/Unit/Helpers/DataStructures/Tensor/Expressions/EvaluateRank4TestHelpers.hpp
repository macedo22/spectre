// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
  Tensor<Datatype, RhsSymmetry, RhsTensorIndexTypeList> R_abcd{};
  std::iota(R_abcd.begin(), R_abcd.end(), 0.0);

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;
  const size_t dim_d = tmpl::at_c<RhsTensorIndexTypeList, 3>::dim;

  // L_{abcd} = R_{abcd}
  const auto L_abcd =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC,
                                  TensorIndexD>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{abdc} = R_{abcd}
  const auto L_abdc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexD,
                                  TensorIndexC>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{acbd} = R_{abcd}
  const auto L_acbd =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB,
                                  TensorIndexD>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{acdb} = R_{abcd}
  const auto L_acdb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexD,
                                  TensorIndexB>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{adbc} = R_{abcd}
  const auto L_adbc =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexB,
                                  TensorIndexC>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{adcb} = R_{abcd}
  const auto L_adcb =
      TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexC,
                                  TensorIndexB>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{bacd} = R_{abcd}
  const auto L_bacd =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC,
                                  TensorIndexD>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{badc} = R_{abcd}
  const auto L_badc =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexD,
                                  TensorIndexC>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{bcad} = R_{abcd}
  const auto L_bcad =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA,
                                  TensorIndexD>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{bcda} = R_{abcd}
  const auto L_bcda =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexD,
                                  TensorIndexA>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{bdac} = R_{abcd}
  const auto L_bdac =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexA,
                                  TensorIndexC>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{bdca} = R_{abcd}
  const auto L_bdca =
      TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexC,
                                  TensorIndexA>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{cabd} = R_{abcd}
  const auto L_cabd =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB,
                                  TensorIndexD>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{cadb} = R_{abcd}
  const auto L_cadb =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexD,
                                  TensorIndexB>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{cbad} = R_{abcd}
  const auto L_cbad =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA,
                                  TensorIndexD>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{cbda} = R_{abcd}
  const auto L_cbda =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexD,
                                  TensorIndexA>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{cdab} = R_{abcd}
  const auto L_cdab =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexA,
                                  TensorIndexB>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{cdba} = R_{abcd}
  const auto L_cdba =
      TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexB,
                                  TensorIndexA>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{dabc} = R_{abcd}
  const auto L_dabc =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexB,
                                  TensorIndexC>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{dacb} = R_{abcd}
  const auto L_dacb =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexC,
                                  TensorIndexB>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{dbac} = R_{abcd}
  const auto L_dbac =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexA,
                                  TensorIndexC>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{dbca} = R_{abcd}
  const auto L_dbca =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexC,
                                  TensorIndexA>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{dcab} = R_{abcd}
  const auto L_dcab =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexA,
                                  TensorIndexB>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  // L_{dcba} = R_{abcd}
  const auto L_dcba =
      TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexB,
                                  TensorIndexA>(
          R_abcd(tensorindex_a, tensorindex_b, tensorindex_c, tensorindex_d));

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
          for (size_t l = 0; l < dim_d; ++l) {
            // For L_{abcd} = R_{abcd}, check that L_{ijkl} == R_{ijkl}
            CHECK(L_abcd.get(i, j, k, l) == R_abcd.get(i, j, k, l));
            // For L_{abdc} = R_{abcd}, check that L_{ijlk} == R_{ijkl}
            CHECK(L_abdc.get(i, j, l, k) == R_abcd.get(i, j, k, l));
            // For L_{acbd} = R_{abcd}, check that L_{ikjl} == R_{ijkl}
            CHECK(L_acbd.get(i, k, j, l) == R_abcd.get(i, j, k, l));
            // For L_{acdb} = R_{abcd}, check that L_{iklj} == R_{ijkl}
            CHECK(L_acdb.get(i, k, l, j) == R_abcd.get(i, j, k, l));
            // For L_{adbc} = R_{abcd}, check that L_{iljk} == R_{ijkl}
            CHECK(L_adbc.get(i, l, j, k) == R_abcd.get(i, j, k, l));
            // For L_{adcb} = R_{abcd}, check that L_{ilkj} == R_{ijkl}
            CHECK(L_adcb.get(i, l, k, j) == R_abcd.get(i, j, k, l));
            // For L_{bacd} = R_{abcd}, check that L_{jikl} == R_{ijkl}
            CHECK(L_bacd.get(j, i, k, l) == R_abcd.get(i, j, k, l));
            // For L_{badc} = R_{abcd}, check that L_{jilk} == R_{ijkl}
            CHECK(L_badc.get(j, i, l, k) == R_abcd.get(i, j, k, l));
            // For L_{bcad} = R_{abcd}, check that L_{jkil} == R_{ijkl}
            CHECK(L_bcad.get(j, k, i, l) == R_abcd.get(i, j, k, l));
            // For L_{bcda} = R_{abcd}, check that L_{jkli} == R_{ijkl}
            CHECK(L_bcda.get(j, k, l, i) == R_abcd.get(i, j, k, l));
            // For L_{bdac} = R_{abcd}, check that L_{jlik} == R_{ijkl}
            CHECK(L_bdac.get(j, l, i, k) == R_abcd.get(i, j, k, l));
            // For L_{bdca} = R_{abcd}, check that L_{jlki} == R_{ijkl}
            CHECK(L_bdca.get(j, l, k, i) == R_abcd.get(i, j, k, l));
            // For L_{cabd} = R_{abcd}, check that L_{kijl} == R_{ijkl}
            CHECK(L_cabd.get(k, i, j, l) == R_abcd.get(i, j, k, l));
            // For L_{cadb} = R_{abcd}, check that L_{kilj} == R_{ijkl}
            CHECK(L_cadb.get(k, i, l, j) == R_abcd.get(i, j, k, l));
            // For L_{cbad} = R_{abcd}, check that L_{kjil} == R_{ijkl}
            CHECK(L_cbad.get(k, j, i, l) == R_abcd.get(i, j, k, l));
            // For L_{cbda} = R_{abcd}, check that L_{kjli} == R_{ijkl}
            CHECK(L_cbda.get(k, j, l, i) == R_abcd.get(i, j, k, l));
            // For L_{cdab} = R_{abcd}, check that L_{klij} == R_{ijkl}
            CHECK(L_cdab.get(k, l, i, j) == R_abcd.get(i, j, k, l));
            // For L_{cdba} = R_{abcd}, check that L_{klji} == R_{ijkl}
            CHECK(L_cdba.get(k, l, j, i) == R_abcd.get(i, j, k, l));
            // For L_{dabc} = R_{abcd}, check that L_{lijk} == R_{ijkl}
            CHECK(L_dabc.get(l, i, j, k) == R_abcd.get(i, j, k, l));
            // For L_{dacb} = R_{abcd}, check that L_{likj} == R_{ijkl}
            CHECK(L_dacb.get(l, i, k, j) == R_abcd.get(i, j, k, l));
            // For L_{dbac} = R_{abcd}, check that L_{ljik} == R_{ijkl}
            CHECK(L_dbac.get(l, j, i, k) == R_abcd.get(i, j, k, l));
            // For L_{dbca} = R_{abcd}, check that L_{ljki} == R_{ijkl}
            CHECK(L_dbca.get(l, j, k, i) == R_abcd.get(i, j, k, l));
            // For L_{dcab} = R_{abcd}, check that L_{lkij} == R_{ijkl}
            CHECK(L_dcab.get(l, k, i, j) == R_abcd.get(i, j, k, l));
            // For L_{dcba} = R_{abcd}, check that L_{lkji} == R_{ijkl}
            CHECK(L_dcba.get(l, k, j, i) == R_abcd.get(i, j, k, l));
        }
      }
    }
  }
}
