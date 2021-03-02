// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 4 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, `TensorIndexC`, and  `TensorIndexD`
/// can be any type of TensorIndex and are not necessarily `ti_a`, `ti_b`,
/// `ti_c`, and `ti_d`. The "A", "B", "C", and "D" suffixes just denote the
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
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_c`
/// \tparam TensorIndexD the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_D`
template <typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC, auto& TensorIndexD>
void test_evaluate_rank_4() noexcept {
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> R_abcd(5_st);
  std::iota(R_abcd.begin(), R_abcd.end(), 0.0);

  // Used for enforcing the ordering of the symmetry and TensorIndexTypes of the
  // LHS Tensor returned by `evaluate`
  const std::int32_t rhs_symmetry_element_a = tmpl::at_c<RhsSymmetry, 0>::value;
  const std::int32_t rhs_symmetry_element_b = tmpl::at_c<RhsSymmetry, 1>::value;
  const std::int32_t rhs_symmetry_element_c = tmpl::at_c<RhsSymmetry, 2>::value;
  const std::int32_t rhs_symmetry_element_d = tmpl::at_c<RhsSymmetry, 3>::value;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;
  using rhs_tensorindextype_d = tmpl::at_c<RhsTensorIndexTypeList, 3>;

  // L_{abcd} = R_{abcd}
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  const Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> L_abcd_returned =
      ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC,
                                    TensorIndexD>(
          R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> L_abcd_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexC,
                                TensorIndexD>(
      make_not_null(&L_abcd_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{abdc} = R_{abcd}
  using L_abdc_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_b,
               rhs_symmetry_element_d, rhs_symmetry_element_c>;
  using L_abdc_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_b,
                 rhs_tensorindextype_d, rhs_tensorindextype_c>;
  const Tensor<DataType, L_abdc_symmetry, L_abdc_tensorindextype_list>
      L_abdc_returned =
          ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB,
                                        TensorIndexD, TensorIndexC>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_abdc_symmetry, L_abdc_tensorindextype_list>
      L_abdc_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexB, TensorIndexD,
                                TensorIndexC>(
      make_not_null(&L_abdc_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{acbd} = R_{abcd}
  using L_acbd_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_c,
               rhs_symmetry_element_b, rhs_symmetry_element_d>;
  using L_acbd_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_c,
                 rhs_tensorindextype_b, rhs_tensorindextype_d>;
  const Tensor<DataType, L_acbd_symmetry, L_acbd_tensorindextype_list>
      L_acbd_returned =
          ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC,
                                        TensorIndexB, TensorIndexD>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_acbd_symmetry, L_acbd_tensorindextype_list>
      L_acbd_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexB,
                                TensorIndexD>(
      make_not_null(&L_acbd_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{acdb} = R_{abcd}
  using L_acdb_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_c,
               rhs_symmetry_element_d, rhs_symmetry_element_b>;
  using L_acdb_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_c,
                 rhs_tensorindextype_d, rhs_tensorindextype_b>;
  const Tensor<DataType, L_acdb_symmetry, L_acdb_tensorindextype_list>
      L_acdb_returned =
          ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC,
                                        TensorIndexD, TensorIndexB>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_acdb_symmetry, L_acdb_tensorindextype_list>
      L_acdb_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexC, TensorIndexD,
                                TensorIndexB>(
      make_not_null(&L_acdb_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{adbc} = R_{abcd}
  using L_adbc_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_d,
               rhs_symmetry_element_b, rhs_symmetry_element_c>;
  using L_adbc_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_d,
                 rhs_tensorindextype_b, rhs_tensorindextype_c>;
  const Tensor<DataType, L_adbc_symmetry, L_adbc_tensorindextype_list>
      L_adbc_returned =
          ::TensorExpressions::evaluate<TensorIndexA, TensorIndexD,
                                        TensorIndexB, TensorIndexC>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_adbc_symmetry, L_adbc_tensorindextype_list>
      L_adbc_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexB,
                                TensorIndexC>(
      make_not_null(&L_adbc_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{adcb} = R_{abcd}
  using L_adcb_symmetry =
      Symmetry<rhs_symmetry_element_a, rhs_symmetry_element_d,
               rhs_symmetry_element_c, rhs_symmetry_element_b>;
  using L_adcb_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_a, rhs_tensorindextype_d,
                 rhs_tensorindextype_c, rhs_tensorindextype_b>;
  const Tensor<DataType, L_adcb_symmetry, L_adcb_tensorindextype_list>
      L_adcb_returned =
          ::TensorExpressions::evaluate<TensorIndexA, TensorIndexD,
                                        TensorIndexC, TensorIndexB>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_adcb_symmetry, L_adcb_tensorindextype_list>
      L_adcb_filled{};
  ::TensorExpressions::evaluate<TensorIndexA, TensorIndexD, TensorIndexC,
                                TensorIndexB>(
      make_not_null(&L_adcb_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{bacd} = R_{abcd}
  using L_bacd_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_a,
               rhs_symmetry_element_c, rhs_symmetry_element_d>;
  using L_bacd_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_a,
                 rhs_tensorindextype_c, rhs_tensorindextype_d>;
  const Tensor<DataType, L_bacd_symmetry, L_bacd_tensorindextype_list>
      L_bacd_returned =
          ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA,
                                        TensorIndexC, TensorIndexD>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_bacd_symmetry, L_bacd_tensorindextype_list>
      L_bacd_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexC,
                                TensorIndexD>(
      make_not_null(&L_bacd_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{badc} = R_{abcd}
  using L_badc_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_a,
               rhs_symmetry_element_d, rhs_symmetry_element_c>;
  using L_badc_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_a,
                 rhs_tensorindextype_d, rhs_tensorindextype_c>;
  const Tensor<DataType, L_badc_symmetry, L_badc_tensorindextype_list>
      L_badc_returned =
          ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA,
                                        TensorIndexD, TensorIndexC>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_badc_symmetry, L_badc_tensorindextype_list>
      L_badc_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexA, TensorIndexD,
                                TensorIndexC>(
      make_not_null(&L_badc_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{bcad} = R_{abcd}
  using L_bcad_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_c,
               rhs_symmetry_element_a, rhs_symmetry_element_d>;
  using L_bcad_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_c,
                 rhs_tensorindextype_a, rhs_tensorindextype_d>;
  const Tensor<DataType, L_bcad_symmetry, L_bcad_tensorindextype_list>
      L_bcad_returned =
          ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC,
                                        TensorIndexA, TensorIndexD>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_bcad_symmetry, L_bcad_tensorindextype_list>
      L_bcad_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexA,
                                TensorIndexD>(
      make_not_null(&L_bcad_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{bcda} = R_{abcd}
  using L_bcda_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_c,
               rhs_symmetry_element_d, rhs_symmetry_element_a>;
  using L_bcda_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_c,
                 rhs_tensorindextype_d, rhs_tensorindextype_a>;
  const Tensor<DataType, L_bcda_symmetry, L_bcda_tensorindextype_list>
      L_bcda_returned =
          ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC,
                                        TensorIndexD, TensorIndexA>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_bcda_symmetry, L_bcda_tensorindextype_list>
      L_bcda_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexC, TensorIndexD,
                                TensorIndexA>(
      make_not_null(&L_bcda_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{bdac} = R_{abcd}
  using L_bdac_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_d,
               rhs_symmetry_element_a, rhs_symmetry_element_c>;
  using L_bdac_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_d,
                 rhs_tensorindextype_a, rhs_tensorindextype_c>;
  const Tensor<DataType, L_bdac_symmetry, L_bdac_tensorindextype_list>
      L_bdac_returned =
          ::TensorExpressions::evaluate<TensorIndexB, TensorIndexD,
                                        TensorIndexA, TensorIndexC>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_bdac_symmetry, L_bdac_tensorindextype_list>
      L_bdac_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexA,
                                TensorIndexC>(
      make_not_null(&L_bdac_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{bdca} = R_{abcd}
  using L_bdca_symmetry =
      Symmetry<rhs_symmetry_element_b, rhs_symmetry_element_d,
               rhs_symmetry_element_c, rhs_symmetry_element_a>;
  using L_bdca_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_b, rhs_tensorindextype_d,
                 rhs_tensorindextype_c, rhs_tensorindextype_a>;
  const Tensor<DataType, L_bdca_symmetry, L_bdca_tensorindextype_list>
      L_bdca_returned =
          ::TensorExpressions::evaluate<TensorIndexB, TensorIndexD,
                                        TensorIndexC, TensorIndexA>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_bdca_symmetry, L_bdca_tensorindextype_list>
      L_bdca_filled{};
  ::TensorExpressions::evaluate<TensorIndexB, TensorIndexD, TensorIndexC,
                                TensorIndexA>(
      make_not_null(&L_bdca_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{cabd} = R_{abcd}
  using L_cabd_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_a,
               rhs_symmetry_element_b, rhs_symmetry_element_d>;
  using L_cabd_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_a,
                 rhs_tensorindextype_b, rhs_tensorindextype_d>;
  const Tensor<DataType, L_cabd_symmetry, L_cabd_tensorindextype_list>
      L_cabd_returned =
          ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA,
                                        TensorIndexB, TensorIndexD>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_cabd_symmetry, L_cabd_tensorindextype_list>
      L_cabd_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexB,
                                TensorIndexD>(
      make_not_null(&L_cabd_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{cadb} = R_{abcd}
  using L_cadb_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_a,
               rhs_symmetry_element_d, rhs_symmetry_element_b>;
  using L_cadb_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_a,
                 rhs_tensorindextype_d, rhs_tensorindextype_b>;
  const Tensor<DataType, L_cadb_symmetry, L_cadb_tensorindextype_list>
      L_cadb_returned =
          ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA,
                                        TensorIndexD, TensorIndexB>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_cadb_symmetry, L_cadb_tensorindextype_list>
      L_cadb_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexA, TensorIndexD,
                                TensorIndexB>(
      make_not_null(&L_cadb_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{cbad} = R_{abcd}
  using L_cbad_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_b,
               rhs_symmetry_element_a, rhs_symmetry_element_d>;
  using L_cbad_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_b,
                 rhs_tensorindextype_a, rhs_tensorindextype_d>;
  const Tensor<DataType, L_cbad_symmetry, L_cbad_tensorindextype_list>
      L_cbad_returned =
          ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB,
                                        TensorIndexA, TensorIndexD>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_cbad_symmetry, L_cbad_tensorindextype_list>
      L_cbad_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexA,
                                TensorIndexD>(
      make_not_null(&L_cbad_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{cbda} = R_{abcd}
  using L_cbda_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_b,
               rhs_symmetry_element_d, rhs_symmetry_element_a>;
  using L_cbda_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_b,
                 rhs_tensorindextype_d, rhs_tensorindextype_a>;
  const Tensor<DataType, L_cbda_symmetry, L_cbda_tensorindextype_list>
      L_cbda_returned =
          ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB,
                                        TensorIndexD, TensorIndexA>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_cbda_symmetry, L_cbda_tensorindextype_list>
      L_cbda_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexB, TensorIndexD,
                                TensorIndexA>(
      make_not_null(&L_cbda_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{cdab} = R_{abcd}
  using L_cdab_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_d,
               rhs_symmetry_element_a, rhs_symmetry_element_b>;
  using L_cdab_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_d,
                 rhs_tensorindextype_a, rhs_tensorindextype_b>;
  const Tensor<DataType, L_cdab_symmetry, L_cdab_tensorindextype_list>
      L_cdab_returned =
          ::TensorExpressions::evaluate<TensorIndexC, TensorIndexD,
                                        TensorIndexA, TensorIndexB>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_cdab_symmetry, L_cdab_tensorindextype_list>
      L_cdab_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexA,
                                TensorIndexB>(
      make_not_null(&L_cdab_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{cdba} = R_{abcd}
  using L_cdba_symmetry =
      Symmetry<rhs_symmetry_element_c, rhs_symmetry_element_d,
               rhs_symmetry_element_b, rhs_symmetry_element_a>;
  using L_cdba_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_c, rhs_tensorindextype_d,
                 rhs_tensorindextype_b, rhs_tensorindextype_a>;
  const Tensor<DataType, L_cdba_symmetry, L_cdba_tensorindextype_list>
      L_cdba_returned =
          ::TensorExpressions::evaluate<TensorIndexC, TensorIndexD,
                                        TensorIndexB, TensorIndexA>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_cdba_symmetry, L_cdba_tensorindextype_list>
      L_cdba_filled{};
  ::TensorExpressions::evaluate<TensorIndexC, TensorIndexD, TensorIndexB,
                                TensorIndexA>(
      make_not_null(&L_cdba_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{dabc} = R_{abcd}
  using L_dabc_symmetry =
      Symmetry<rhs_symmetry_element_d, rhs_symmetry_element_a,
               rhs_symmetry_element_b, rhs_symmetry_element_c>;
  using L_dabc_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_d, rhs_tensorindextype_a,
                 rhs_tensorindextype_b, rhs_tensorindextype_c>;
  const Tensor<DataType, L_dabc_symmetry, L_dabc_tensorindextype_list>
      L_dabc_returned =
          ::TensorExpressions::evaluate<TensorIndexD, TensorIndexA,
                                        TensorIndexB, TensorIndexC>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_dabc_symmetry, L_dabc_tensorindextype_list>
      L_dabc_filled{};
  ::TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexB,
                                TensorIndexC>(
      make_not_null(&L_dabc_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{dacb} = R_{abcd}
  using L_dacb_symmetry =
      Symmetry<rhs_symmetry_element_d, rhs_symmetry_element_a,
               rhs_symmetry_element_c, rhs_symmetry_element_b>;
  using L_dacb_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_d, rhs_tensorindextype_a,
                 rhs_tensorindextype_c, rhs_tensorindextype_b>;
  const Tensor<DataType, L_dacb_symmetry, L_dacb_tensorindextype_list>
      L_dacb_returned =
          ::TensorExpressions::evaluate<TensorIndexD, TensorIndexA,
                                        TensorIndexC, TensorIndexB>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_dacb_symmetry, L_dacb_tensorindextype_list>
      L_dacb_filled{};
  ::TensorExpressions::evaluate<TensorIndexD, TensorIndexA, TensorIndexC,
                                TensorIndexB>(
      make_not_null(&L_dacb_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{dbac} = R_{abcd}
  using L_dbac_symmetry =
      Symmetry<rhs_symmetry_element_d, rhs_symmetry_element_b,
               rhs_symmetry_element_a, rhs_symmetry_element_c>;
  using L_dbac_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_d, rhs_tensorindextype_b,
                 rhs_tensorindextype_a, rhs_tensorindextype_c>;
  const Tensor<DataType, L_dbac_symmetry, L_dbac_tensorindextype_list>
      L_dbac_returned =
          ::TensorExpressions::evaluate<TensorIndexD, TensorIndexB,
                                        TensorIndexA, TensorIndexC>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_dbac_symmetry, L_dbac_tensorindextype_list>
      L_dbac_filled{};
  ::TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexA,
                                TensorIndexC>(
      make_not_null(&L_dbac_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{dbca} = R_{abcd}
  using L_dbca_symmetry =
      Symmetry<rhs_symmetry_element_d, rhs_symmetry_element_b,
               rhs_symmetry_element_c, rhs_symmetry_element_a>;
  using L_dbca_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_d, rhs_tensorindextype_b,
                 rhs_tensorindextype_c, rhs_tensorindextype_a>;
  const Tensor<DataType, L_dbca_symmetry, L_dbca_tensorindextype_list>
      L_dbca_returned =
          ::TensorExpressions::evaluate<TensorIndexD, TensorIndexB,
                                        TensorIndexC, TensorIndexA>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_dbca_symmetry, L_dbca_tensorindextype_list>
      L_dbca_filled{};
  ::TensorExpressions::evaluate<TensorIndexD, TensorIndexB, TensorIndexC,
                                TensorIndexA>(
      make_not_null(&L_dbca_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{dcab} = R_{abcd}
  using L_dcab_symmetry =
      Symmetry<rhs_symmetry_element_d, rhs_symmetry_element_c,
               rhs_symmetry_element_a, rhs_symmetry_element_b>;
  using L_dcab_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_d, rhs_tensorindextype_c,
                 rhs_tensorindextype_a, rhs_tensorindextype_b>;
  const Tensor<DataType, L_dcab_symmetry, L_dcab_tensorindextype_list>
      L_dcab_returned =
          ::TensorExpressions::evaluate<TensorIndexD, TensorIndexC,
                                        TensorIndexA, TensorIndexB>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_dcab_symmetry, L_dcab_tensorindextype_list>
      L_dcab_filled{};
  ::TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexA,
                                TensorIndexB>(
      make_not_null(&L_dcab_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  // L_{dcba} = R_{abcd}
  using L_dcba_symmetry =
      Symmetry<rhs_symmetry_element_d, rhs_symmetry_element_c,
               rhs_symmetry_element_b, rhs_symmetry_element_a>;
  using L_dcba_tensorindextype_list =
      tmpl::list<rhs_tensorindextype_d, rhs_tensorindextype_c,
                 rhs_tensorindextype_b, rhs_tensorindextype_a>;
  const Tensor<DataType, L_dcba_symmetry, L_dcba_tensorindextype_list>
      L_dcba_returned =
          ::TensorExpressions::evaluate<TensorIndexD, TensorIndexC,
                                        TensorIndexB, TensorIndexA>(
              R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));
  Tensor<DataType, L_dcba_symmetry, L_dcba_tensorindextype_list>
      L_dcba_filled{};
  ::TensorExpressions::evaluate<TensorIndexD, TensorIndexC, TensorIndexB,
                                TensorIndexA>(
      make_not_null(&L_dcba_filled),
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD));

  const size_t dim_a = tmpl::at_c<RhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<RhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<RhsTensorIndexTypeList, 2>::dim;
  const size_t dim_d = tmpl::at_c<RhsTensorIndexTypeList, 3>::dim;

  for (size_t i = 0; i < dim_a; ++i) {
    for (size_t j = 0; j < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
          for (size_t l = 0; l < dim_d; ++l) {
            // For L_{abcd} = R_{abcd}, check that L_{ijkl} == R_{ijkl}
            CHECK(L_abcd_returned.get(i, j, k, l) == R_abcd.get(i, j, k, l));
            CHECK(L_abcd_filled.get(i, j, k, l) == R_abcd.get(i, j, k, l));
            // For L_{abdc} = R_{abcd}, check that L_{ijlk} == R_{ijkl}
            CHECK(L_abdc_returned.get(i, j, l, k) == R_abcd.get(i, j, k, l));
            CHECK(L_abdc_filled.get(i, j, l, k) == R_abcd.get(i, j, k, l));
            // For L_{acbd} = R_{abcd}, check that L_{ikjl} == R_{ijkl}
            CHECK(L_acbd_returned.get(i, k, j, l) == R_abcd.get(i, j, k, l));
            CHECK(L_acbd_filled.get(i, k, j, l) == R_abcd.get(i, j, k, l));
            // For L_{acdb} = R_{abcd}, check that L_{iklj} == R_{ijkl}
            CHECK(L_acdb_returned.get(i, k, l, j) == R_abcd.get(i, j, k, l));
            CHECK(L_acdb_filled.get(i, k, l, j) == R_abcd.get(i, j, k, l));
            // For L_{adbc} = R_{abcd}, check that L_{iljk} == R_{ijkl}
            CHECK(L_adbc_returned.get(i, l, j, k) == R_abcd.get(i, j, k, l));
            CHECK(L_adbc_filled.get(i, l, j, k) == R_abcd.get(i, j, k, l));
            // For L_{adcb} = R_{abcd}, check that L_{ilkj} == R_{ijkl}
            CHECK(L_adcb_returned.get(i, l, k, j) == R_abcd.get(i, j, k, l));
            CHECK(L_adcb_filled.get(i, l, k, j) == R_abcd.get(i, j, k, l));
            // For L_{bacd} = R_{abcd}, check that L_{jikl} == R_{ijkl}
            CHECK(L_bacd_returned.get(j, i, k, l) == R_abcd.get(i, j, k, l));
            CHECK(L_bacd_filled.get(j, i, k, l) == R_abcd.get(i, j, k, l));
            // For L_{badc} = R_{abcd}, check that L_{jilk} == R_{ijkl}
            CHECK(L_badc_returned.get(j, i, l, k) == R_abcd.get(i, j, k, l));
            CHECK(L_badc_filled.get(j, i, l, k) == R_abcd.get(i, j, k, l));
            // For L_{bcad} = R_{abcd}, check that L_{jkil} == R_{ijkl}
            CHECK(L_bcad_returned.get(j, k, i, l) == R_abcd.get(i, j, k, l));
            CHECK(L_bcad_filled.get(j, k, i, l) == R_abcd.get(i, j, k, l));
            // For L_{bcda} = R_{abcd}, check that L_{jkli} == R_{ijkl}
            CHECK(L_bcda_returned.get(j, k, l, i) == R_abcd.get(i, j, k, l));
            CHECK(L_bcda_filled.get(j, k, l, i) == R_abcd.get(i, j, k, l));
            // For L_{bdac} = R_{abcd}, check that L_{jlik} == R_{ijkl}
            CHECK(L_bdac_returned.get(j, l, i, k) == R_abcd.get(i, j, k, l));
            CHECK(L_bdac_filled.get(j, l, i, k) == R_abcd.get(i, j, k, l));
            // For L_{bdca} = R_{abcd}, check that L_{jlki} == R_{ijkl}
            CHECK(L_bdca_returned.get(j, l, k, i) == R_abcd.get(i, j, k, l));
            CHECK(L_bdca_filled.get(j, l, k, i) == R_abcd.get(i, j, k, l));
            // For L_{cabd} = R_{abcd}, check that L_{kijl} == R_{ijkl}
            CHECK(L_cabd_returned.get(k, i, j, l) == R_abcd.get(i, j, k, l));
            CHECK(L_cabd_filled.get(k, i, j, l) == R_abcd.get(i, j, k, l));
            // For L_{cadb} = R_{abcd}, check that L_{kilj} == R_{ijkl}
            CHECK(L_cadb_returned.get(k, i, l, j) == R_abcd.get(i, j, k, l));
            CHECK(L_cadb_filled.get(k, i, l, j) == R_abcd.get(i, j, k, l));
            // For L_{cbad} = R_{abcd}, check that L_{kjil} == R_{ijkl}
            CHECK(L_cbad_returned.get(k, j, i, l) == R_abcd.get(i, j, k, l));
            CHECK(L_cbad_filled.get(k, j, i, l) == R_abcd.get(i, j, k, l));
            // For L_{cbda} = R_{abcd}, check that L_{kjli} == R_{ijkl}
            CHECK(L_cbda_returned.get(k, j, l, i) == R_abcd.get(i, j, k, l));
            CHECK(L_cbda_filled.get(k, j, l, i) == R_abcd.get(i, j, k, l));
            // For L_{cdab} = R_{abcd}, check that L_{klij} == R_{ijkl}
            CHECK(L_cdab_returned.get(k, l, i, j) == R_abcd.get(i, j, k, l));
            CHECK(L_cdab_filled.get(k, l, i, j) == R_abcd.get(i, j, k, l));
            // For L_{cdba} = R_{abcd}, check that L_{klji} == R_{ijkl}
            CHECK(L_cdba_returned.get(k, l, j, i) == R_abcd.get(i, j, k, l));
            CHECK(L_cdba_filled.get(k, l, j, i) == R_abcd.get(i, j, k, l));
            // For L_{dabc} = R_{abcd}, check that L_{lijk} == R_{ijkl}
            CHECK(L_dabc_returned.get(l, i, j, k) == R_abcd.get(i, j, k, l));
            CHECK(L_dabc_filled.get(l, i, j, k) == R_abcd.get(i, j, k, l));
            // For L_{dacb} = R_{abcd}, check that L_{likj} == R_{ijkl}
            CHECK(L_dacb_returned.get(l, i, k, j) == R_abcd.get(i, j, k, l));
            CHECK(L_dacb_filled.get(l, i, k, j) == R_abcd.get(i, j, k, l));
            // For L_{dbac} = R_{abcd}, check that L_{ljik} == R_{ijkl}
            CHECK(L_dbac_returned.get(l, j, i, k) == R_abcd.get(i, j, k, l));
            CHECK(L_dbac_filled.get(l, j, i, k) == R_abcd.get(i, j, k, l));
            // For L_{dbca} = R_{abcd}, check that L_{ljki} == R_{ijkl}
            CHECK(L_dbca_returned.get(l, j, k, i) == R_abcd.get(i, j, k, l));
            CHECK(L_dbca_filled.get(l, j, k, i) == R_abcd.get(i, j, k, l));
            // For L_{dcab} = R_{abcd}, check that L_{lkij} == R_{ijkl}
            CHECK(L_dcab_returned.get(l, k, i, j) == R_abcd.get(i, j, k, l));
            CHECK(L_dcab_filled.get(l, k, i, j) == R_abcd.get(i, j, k, l));
            // For L_{dcba} = R_{abcd}, check that L_{lkji} == R_{ijkl}
            CHECK(L_dcba_returned.get(l, k, j, i) == R_abcd.get(i, j, k, l));
            CHECK(L_dcba_filled.get(l, k, j, i) == R_abcd.get(i, j, k, l));
        }
      }
    }
  }
}

}  // namespace TestHelpers::TensorExpressions
