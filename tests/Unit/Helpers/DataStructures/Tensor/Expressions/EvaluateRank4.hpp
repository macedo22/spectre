// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {
// TODO : update testing func docs

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
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
/// \tparam TensorIndexD the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::D`
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, auto& TensorIndexD, typename DataType,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_4_core() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  const auto R_abcd = make_with_random_values<
      Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_abcd =
      make_with_value<Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>>(
          used_for_size, component_placeholder_value<DataType>::value);

  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<LhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<LhsSymmetry, 1>::value;
  const std::int32_t lhs_symmetry_element_c = tmpl::at_c<LhsSymmetry, 2>::value;
  const std::int32_t lhs_symmetry_element_d = tmpl::at_c<LhsSymmetry, 3>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using lhs_tensorindextype_c = tmpl::at_c<LhsTensorIndexTypeList, 2>;
  using lhs_tensorindextype_d = tmpl::at_c<LhsTensorIndexTypeList, 3>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;
  using rhs_tensorindextype_d = tmpl::at_c<RhsTensorIndexTypeList, 3>;

  std::array<std::pair<size_t, size_t>, 4> lhs_index_value_ranges{};
  lhs_index_value_ranges[0] =
      get_index_value_range<lhs_tensorindextype_a, TensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, TensorIndexB>();
  lhs_index_value_ranges[2] =
      get_index_value_range<lhs_tensorindextype_c, TensorIndexC>();
  lhs_index_value_ranges[3] =
      get_index_value_range<lhs_tensorindextype_d, TensorIndexD>();
  std::array<std::pair<size_t, size_t>, 4> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, TensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, TensorIndexB>();
  rhs_index_value_ranges[2] =
      get_index_value_range<rhs_tensorindextype_c, TensorIndexC>();
  rhs_index_value_ranges[3] =
      get_index_value_range<rhs_tensorindextype_d, TensorIndexD>();

  for (size_t lhs_a = lhs_index_value_ranges[0].first,
              rhs_a = rhs_index_value_ranges[0].first;
       lhs_a <= lhs_index_value_ranges[0].second; lhs_a++, rhs_a++) {
    for (size_t lhs_b = lhs_index_value_ranges[1].first,
                rhs_b = rhs_index_value_ranges[1].first;
         lhs_b <= lhs_index_value_ranges[1].second; lhs_b++, rhs_b++) {
      for (size_t lhs_c = lhs_index_value_ranges[2].first,
                  rhs_c = rhs_index_value_ranges[2].first;
           lhs_c <= lhs_index_value_ranges[2].second; lhs_c++, rhs_c++) {
        for (size_t lhs_d = lhs_index_value_ranges[3].first,
                    rhs_d = rhs_index_value_ranges[3].first;
             lhs_d <= lhs_index_value_ranges[3].second; lhs_d++, rhs_d++) {
          expected_L_abcd.get(lhs_a, lhs_b, lhs_c, lhs_d) =
              R_abcd.get(rhs_a, rhs_b, rhs_c, rhs_d);
        }
      }
    }
  }

  const auto rhs_expression =
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD);

  // L_{abcd} = R_{abcd}
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  using L_abcd_type = Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>;
  L_abcd_type L_abcd;
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC,
                TensorIndexD>(make_not_null(&L_abcd), rhs_expression);

  // L_{abdc} = R_{abcd}
  using L_abdc_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_b,
               lhs_symmetry_element_d, lhs_symmetry_element_c>;
  using L_abdc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_b,
                 lhs_tensorindextype_d, lhs_tensorindextype_c>;
  using L_abdc_type =
      Tensor<DataType, L_abdc_symmetry, L_abdc_tensorindextype_list>;
  L_abdc_type L_abdc;
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexD,
                TensorIndexC>(make_not_null(&L_abdc), rhs_expression);

  // L_{acbd} = R_{abcd}
  using L_acbd_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_b, lhs_symmetry_element_d>;
  using L_acbd_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_b, lhs_tensorindextype_d>;
  using L_acbd_type =
      Tensor<DataType, L_acbd_symmetry, L_acbd_tensorindextype_list>;
  L_acbd_type L_acbd;
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexB,
                TensorIndexD>(make_not_null(&L_acbd), rhs_expression);

  // L_{acdb} = R_{abcd}
  using L_acdb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_d, lhs_symmetry_element_b>;
  using L_acdb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_d, lhs_tensorindextype_b>;
  using L_acdb_type =
      Tensor<DataType, L_acdb_symmetry, L_acdb_tensorindextype_list>;
  L_acdb_type L_acdb;
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexD,
                TensorIndexB>(make_not_null(&L_acdb), rhs_expression);

  // L_{adbc} = R_{abcd}
  using L_adbc_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_d,
               lhs_symmetry_element_b, lhs_symmetry_element_c>;
  using L_adbc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_d,
                 lhs_tensorindextype_b, lhs_tensorindextype_c>;
  using L_adbc_type =
      Tensor<DataType, L_adbc_symmetry, L_adbc_tensorindextype_list>;
  L_adbc_type L_adbc;
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexD, TensorIndexB,
                TensorIndexC>(make_not_null(&L_adbc), rhs_expression);

  // L_{adcb} = R_{abcd}
  using L_adcb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_d,
               lhs_symmetry_element_c, lhs_symmetry_element_b>;
  using L_adcb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_d,
                 lhs_tensorindextype_c, lhs_tensorindextype_b>;
  using L_adcb_type =
      Tensor<DataType, L_adcb_symmetry, L_adcb_tensorindextype_list>;
  L_adcb_type L_adcb;
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexD, TensorIndexC,
                TensorIndexB>(make_not_null(&L_adcb), rhs_expression);

  // L_{bacd} = R_{abcd}
  using L_bacd_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_c, lhs_symmetry_element_d>;
  using L_bacd_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_c, lhs_tensorindextype_d>;
  using L_bacd_type =
      Tensor<DataType, L_bacd_symmetry, L_bacd_tensorindextype_list>;
  L_bacd_type L_bacd;
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexC,
                TensorIndexD>(make_not_null(&L_bacd), rhs_expression);

  // L_{badc} = R_{abcd}
  using L_badc_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_d, lhs_symmetry_element_c>;
  using L_badc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_d, lhs_tensorindextype_c>;
  using L_badc_type =
      Tensor<DataType, L_badc_symmetry, L_badc_tensorindextype_list>;
  L_badc_type L_badc;
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexD,
                TensorIndexC>(make_not_null(&L_badc), rhs_expression);

  // L_{bcad} = R_{abcd}
  using L_bcad_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_a, lhs_symmetry_element_d>;
  using L_bcad_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_a, lhs_tensorindextype_d>;
  using L_bcad_type =
      Tensor<DataType, L_bcad_symmetry, L_bcad_tensorindextype_list>;
  L_bcad_type L_bcad;
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexA,
                TensorIndexD>(make_not_null(&L_bcad), rhs_expression);

  // L_{bcda} = R_{abcd}
  using L_bcda_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_d, lhs_symmetry_element_a>;
  using L_bcda_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_d, lhs_tensorindextype_a>;
  using L_bcda_type =
      Tensor<DataType, L_bcda_symmetry, L_bcda_tensorindextype_list>;
  L_bcda_type L_bcda;
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexD,
                TensorIndexA>(make_not_null(&L_bcda), rhs_expression);

  // L_{bdac} = R_{abcd}
  using L_bdac_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_d,
               lhs_symmetry_element_a, lhs_symmetry_element_c>;
  using L_bdac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_d,
                 lhs_tensorindextype_a, lhs_tensorindextype_c>;
  using L_bdac_type =
      Tensor<DataType, L_bdac_symmetry, L_bdac_tensorindextype_list>;
  L_bdac_type L_bdac;
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexD, TensorIndexA,
                TensorIndexC>(make_not_null(&L_bdac), rhs_expression);

  // L_{bdca} = R_{abcd}
  using L_bdca_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_d,
               lhs_symmetry_element_c, lhs_symmetry_element_a>;
  using L_bdca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_d,
                 lhs_tensorindextype_c, lhs_tensorindextype_a>;
  using L_bdca_type =
      Tensor<DataType, L_bdca_symmetry, L_bdca_tensorindextype_list>;
  L_bdca_type L_bdca;
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexD, TensorIndexC,
                TensorIndexA>(make_not_null(&L_bdca), rhs_expression);

  // L_{cabd} = R_{abcd}
  using L_cabd_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_b, lhs_symmetry_element_d>;
  using L_cabd_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_b, lhs_tensorindextype_d>;
  using L_cabd_type =
      Tensor<DataType, L_cabd_symmetry, L_cabd_tensorindextype_list>;
  L_cabd_type L_cabd;
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexB,
                TensorIndexD>(make_not_null(&L_cabd), rhs_expression);

  // L_{cadb} = R_{abcd}
  using L_cadb_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_d, lhs_symmetry_element_b>;
  using L_cadb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_d, lhs_tensorindextype_b>;
  using L_cadb_type =
      Tensor<DataType, L_cadb_symmetry, L_cadb_tensorindextype_list>;
  L_cadb_type L_cadb;
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexD,
                TensorIndexB>(make_not_null(&L_cadb), rhs_expression);

  // L_{cbad} = R_{abcd}
  using L_cbad_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_a, lhs_symmetry_element_d>;
  using L_cbad_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_a, lhs_tensorindextype_d>;
  using L_cbad_type =
      Tensor<DataType, L_cbad_symmetry, L_cbad_tensorindextype_list>;
  L_cbad_type L_cbad;
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexA,
                TensorIndexD>(make_not_null(&L_cbad), rhs_expression);

  // L_{cbda} = R_{abcd}
  using L_cbda_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_d, lhs_symmetry_element_a>;
  using L_cbda_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_d, lhs_tensorindextype_a>;
  using L_cbda_type =
      Tensor<DataType, L_cbda_symmetry, L_cbda_tensorindextype_list>;
  L_cbda_type L_cbda;
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexD,
                TensorIndexA>(make_not_null(&L_cbda), rhs_expression);

  // L_{cdab} = R_{abcd}
  using L_cdab_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_d,
               lhs_symmetry_element_a, lhs_symmetry_element_b>;
  using L_cdab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_d,
                 lhs_tensorindextype_a, lhs_tensorindextype_b>;
  using L_cdab_type =
      Tensor<DataType, L_cdab_symmetry, L_cdab_tensorindextype_list>;
  L_cdab_type L_cdab;
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexD, TensorIndexA,
                TensorIndexB>(make_not_null(&L_cdab), rhs_expression);

  // L_{cdba} = R_{abcd}
  using L_cdba_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_d,
               lhs_symmetry_element_b, lhs_symmetry_element_a>;
  using L_cdba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_d,
                 lhs_tensorindextype_b, lhs_tensorindextype_a>;
  using L_cdba_type =
      Tensor<DataType, L_cdba_symmetry, L_cdba_tensorindextype_list>;
  L_cdba_type L_cdba;
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexD, TensorIndexB,
                TensorIndexA>(make_not_null(&L_cdba), rhs_expression);

  // L_{dabc} = R_{abcd}
  using L_dabc_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_a,
               lhs_symmetry_element_b, lhs_symmetry_element_c>;
  using L_dabc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_a,
                 lhs_tensorindextype_b, lhs_tensorindextype_c>;
  using L_dabc_type =
      Tensor<DataType, L_dabc_symmetry, L_dabc_tensorindextype_list>;
  L_dabc_type L_dabc;
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexA, TensorIndexB,
                TensorIndexC>(make_not_null(&L_dabc), rhs_expression);

  // L_{dacb} = R_{abcd}
  using L_dacb_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_a,
               lhs_symmetry_element_c, lhs_symmetry_element_b>;
  using L_dacb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_a,
                 lhs_tensorindextype_c, lhs_tensorindextype_b>;
  using L_dacb_type =
      Tensor<DataType, L_dacb_symmetry, L_dacb_tensorindextype_list>;
  L_dacb_type L_dacb;
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexA, TensorIndexC,
                TensorIndexB>(make_not_null(&L_dacb), rhs_expression);

  // L_{dbac} = R_{abcd}
  using L_dbac_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_b,
               lhs_symmetry_element_a, lhs_symmetry_element_c>;
  using L_dbac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_b,
                 lhs_tensorindextype_a, lhs_tensorindextype_c>;
  using L_dbac_type =
      Tensor<DataType, L_dbac_symmetry, L_dbac_tensorindextype_list>;
  L_dbac_type L_dbac;
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexB, TensorIndexA,
                TensorIndexC>(make_not_null(&L_dbac), rhs_expression);

  // L_{dbca} = R_{abcd}
  using L_dbca_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_b,
               lhs_symmetry_element_c, lhs_symmetry_element_a>;
  using L_dbca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_b,
                 lhs_tensorindextype_c, lhs_tensorindextype_a>;
  using L_dbca_type =
      Tensor<DataType, L_dbca_symmetry, L_dbca_tensorindextype_list>;
  L_dbca_type L_dbca;
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexB, TensorIndexC,
                TensorIndexA>(make_not_null(&L_dbca), rhs_expression);

  // L_{dcab} = R_{abcd}
  using L_dcab_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_c,
               lhs_symmetry_element_a, lhs_symmetry_element_b>;
  using L_dcab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_c,
                 lhs_tensorindextype_a, lhs_tensorindextype_b>;
  using L_dcab_type =
      Tensor<DataType, L_dcab_symmetry, L_dcab_tensorindextype_list>;
  L_dcab_type L_dcab;
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexC, TensorIndexA,
                TensorIndexB>(make_not_null(&L_dcab), rhs_expression);

  // L_{dcba} = R_{abcd}
  using L_dcba_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_c,
               lhs_symmetry_element_b, lhs_symmetry_element_a>;
  using L_dcba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_c,
                 lhs_tensorindextype_b, lhs_tensorindextype_a>;
  using L_dcba_type =
      Tensor<DataType, L_dcba_symmetry, L_dcba_tensorindextype_list>;
  L_dcba_type L_dcba;
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexC, TensorIndexB,
                TensorIndexA>(make_not_null(&L_dcba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<LhsTensorIndexTypeList, 2>::dim;
  const size_t dim_d = tmpl::at_c<LhsTensorIndexTypeList, 3>::dim;

  // check LHS evaluated correctly
  for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
    for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
      for (size_t lhs_c = 0; lhs_c < dim_c; ++lhs_c) {
        for (size_t lhs_d = 0; lhs_d < dim_d; ++lhs_d) {
          const auto& expected_result =
              expected_L_abcd.get(lhs_a, lhs_b, lhs_c, lhs_d);

          CHECK(L_abcd.get(lhs_a, lhs_b, lhs_c, lhs_d) == expected_result);
          CHECK(L_abdc.get(lhs_a, lhs_b, lhs_d, lhs_c) == expected_result);
          CHECK(L_acbd.get(lhs_a, lhs_c, lhs_b, lhs_d) == expected_result);
          CHECK(L_acdb.get(lhs_a, lhs_c, lhs_d, lhs_b) == expected_result);
          CHECK(L_adbc.get(lhs_a, lhs_d, lhs_b, lhs_c) == expected_result);
          CHECK(L_adcb.get(lhs_a, lhs_d, lhs_c, lhs_b) == expected_result);
          CHECK(L_bacd.get(lhs_b, lhs_a, lhs_c, lhs_d) == expected_result);
          CHECK(L_badc.get(lhs_b, lhs_a, lhs_d, lhs_c) == expected_result);
          CHECK(L_bcad.get(lhs_b, lhs_c, lhs_a, lhs_d) == expected_result);
          CHECK(L_bcda.get(lhs_b, lhs_c, lhs_d, lhs_a) == expected_result);
          CHECK(L_bdac.get(lhs_b, lhs_d, lhs_a, lhs_c) == expected_result);
          CHECK(L_bdca.get(lhs_b, lhs_d, lhs_c, lhs_a) == expected_result);
          CHECK(L_cabd.get(lhs_c, lhs_a, lhs_b, lhs_d) == expected_result);
          CHECK(L_cadb.get(lhs_c, lhs_a, lhs_d, lhs_b) == expected_result);
          CHECK(L_cbad.get(lhs_c, lhs_b, lhs_a, lhs_d) == expected_result);
          CHECK(L_cbda.get(lhs_c, lhs_b, lhs_d, lhs_a) == expected_result);
          CHECK(L_cdab.get(lhs_c, lhs_d, lhs_a, lhs_b) == expected_result);
          CHECK(L_cdba.get(lhs_c, lhs_d, lhs_b, lhs_a) == expected_result);
          CHECK(L_dabc.get(lhs_d, lhs_a, lhs_b, lhs_c) == expected_result);
          CHECK(L_dacb.get(lhs_d, lhs_a, lhs_c, lhs_b) == expected_result);
          CHECK(L_dbac.get(lhs_d, lhs_b, lhs_a, lhs_c) == expected_result);
          CHECK(L_dbca.get(lhs_d, lhs_b, lhs_c, lhs_a) == expected_result);
          CHECK(L_dcab.get(lhs_d, lhs_c, lhs_a, lhs_b) == expected_result);
          CHECK(L_dcba.get(lhs_d, lhs_c, lhs_b, lhs_a) == expected_result);
        }
      }
    }
  }
}

template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, auto& TensorIndexD, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_4() {
  TestHelpers::tenex::test_evaluate_rank_4_core<
      ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD,
      double, RhsSymmetry, RhsTensorIndexTypeList, RhsSymmetry,
      LhsTensorIndexTypeList>();
  TestHelpers::tenex::test_evaluate_rank_4_core<
      ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD,
      DataVector, RhsSymmetry, RhsTensorIndexTypeList, RhsSymmetry,
      LhsTensorIndexTypeList>();
}
}  // namespace TestHelpers::tenex
