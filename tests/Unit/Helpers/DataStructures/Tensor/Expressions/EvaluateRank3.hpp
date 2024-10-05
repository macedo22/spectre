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

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {
template <bool ReturnLhsTensor, auto&... LhsTensorIndices, typename LhsTensor,
          typename RhsExpression>
void call_evaluate(const gsl::not_null<LhsTensor*> lhs_tensor,
                   const RhsExpression& rhs_expression) {
  if constexpr (ReturnLhsTensor) {
    *lhs_tensor = ::tenex::evaluate<LhsTensorIndices...>(rhs_expression);
  } else {
    ::tenex::evaluate<LhsTensorIndices...>(lhs_tensor, rhs_expression);
  }
}

template <typename Index, auto& TensorIndex>
constexpr std::pair<size_t, size_t> get_index_value_range() {
  constexpr bool tensorindex_is_time =
      ::tenex::detail::is_time_index_value(TensorIndex.value);
  static_assert(
      not(Index::index_type == IndexType::Spatial and tensorindex_is_time),
      "Cannot use a concrete time TensorIndex with a SpatialIndex.");
  std::pair<size_t, size_t> range{};
  range.first =
      Index::index_type == IndexType::Spacetime and not TensorIndex.is_spacetime
          ? 1
          : 0;
  range.second = tensorindex_is_time ? 0 : Index::dim - 1;
  return range;
}

// TODO : update testing func docs

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 3 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti::a`, `ti::b`, and `ti::c`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b, c), then this
/// test checks that the data in the evaluated LHS tensor is correct according
/// to the index orders of the LHS and RHS. The possible cases that are checked
/// are when the LHS tensor is evaluated with index orders: (a, b, c),
/// (a, c, b), (b, a, c), (b, c, a), (c, a, b), and (c, b, a).
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
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          auto& TensorIndexC, typename DataType, typename LhsSymmetry,
          typename LhsTensorIndexTypeList, typename RhsSymmetry = LhsSymmetry,
          typename RhsTensorIndexTypeList = LhsTensorIndexTypeList>
void test_evaluate_rank_3() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  const auto R_abc = make_with_random_values<
      Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_abc =
      ReturnLhsTensor
          ? Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>{}
          : make_with_value<
                Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>>(
                used_for_size, component_placeholder_value<DataType>::value);

  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<LhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<LhsSymmetry, 1>::value;
  const std::int32_t lhs_symmetry_element_c = tmpl::at_c<LhsSymmetry, 2>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using lhs_tensorindextype_c = tmpl::at_c<LhsTensorIndexTypeList, 2>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;

  std::array<std::pair<size_t, size_t>, 3> lhs_index_value_ranges{};
  lhs_index_value_ranges[0] =
      get_index_value_range<lhs_tensorindextype_a, TensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, TensorIndexB>();
  lhs_index_value_ranges[2] =
      get_index_value_range<lhs_tensorindextype_c, TensorIndexC>();
  std::array<std::pair<size_t, size_t>, 3> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, TensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, TensorIndexB>();
  rhs_index_value_ranges[2] =
      get_index_value_range<rhs_tensorindextype_c, TensorIndexC>();

  for (size_t lhs_a = lhs_index_value_ranges[0].first,
              rhs_a = rhs_index_value_ranges[0].first;
       lhs_a <= lhs_index_value_ranges[0].second; lhs_a++, rhs_a++) {
    for (size_t lhs_b = lhs_index_value_ranges[1].first,
                rhs_b = rhs_index_value_ranges[1].first;
         lhs_b <= lhs_index_value_ranges[1].second; lhs_b++, rhs_b++) {
      for (size_t lhs_c = lhs_index_value_ranges[2].first,
                  rhs_c = rhs_index_value_ranges[2].first;
           lhs_c <= lhs_index_value_ranges[2].second; lhs_c++, rhs_c++) {
        expected_L_abc.get(lhs_a, lhs_b, lhs_c) =
            R_abc.get(rhs_a, rhs_b, rhs_c);
      }
    }
  }

  const auto rhs_expression = R_abc(TensorIndexA, TensorIndexB, TensorIndexC);

  // L_{abc} = R_{abc}
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  using L_abc_type = Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>;
  L_abc_type L_abc(used_for_size);
  std::fill(L_abc.begin(), L_abc.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC>(
      make_not_null(&L_abc), rhs_expression);

  // L_{acb} = R_{abc}
  using L_acb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_b>;
  using L_acb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_b>;
  using L_acb_type =
      Tensor<DataType, L_acb_symmetry, L_acb_tensorindextype_list>;
  L_acb_type L_acb(used_for_size);
  std::fill(L_acb.begin(), L_acb.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexB>(
      make_not_null(&L_acb), rhs_expression);

  // L_{bac} = R_{abc}
  using L_bac_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_c>;
  using L_bac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_c>;
  using L_bac_type =
      Tensor<DataType, L_bac_symmetry, L_bac_tensorindextype_list>;
  L_bac_type L_bac(used_for_size);
  std::fill(L_bac.begin(), L_bac.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexC>(
      make_not_null(&L_bac), rhs_expression);

  // L_{bca} = R_{abc}
  using L_bca_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_a>;
  using L_bca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_a>;
  using L_bca_type =
      Tensor<DataType, L_bca_symmetry, L_bca_tensorindextype_list>;
  L_bca_type L_bca(used_for_size);
  std::fill(L_bca.begin(), L_bca.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexA>(
      make_not_null(&L_bca), rhs_expression);

  // L_{cab} = R_{abc}
  using L_cab_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_b>;
  using L_cab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_b>;
  using L_cab_type =
      Tensor<DataType, L_cab_symmetry, L_cab_tensorindextype_list>;
  L_cab_type L_cab(used_for_size);
  std::fill(L_cab.begin(), L_cab.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexB>(
      make_not_null(&L_cab), rhs_expression);

  // L_{cba} = R_{abc}
  using L_cba_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_a>;
  using L_cba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_a>;
  using L_cba_type =
      Tensor<DataType, L_cba_symmetry, L_cba_tensorindextype_list>;
  L_cba_type L_cba(used_for_size);
  std::fill(L_cba.begin(), L_cba.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexA>(
      make_not_null(&L_cba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<LhsTensorIndexTypeList, 2>::dim;

  for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
    for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
      for (size_t lhs_c = 0; lhs_c < dim_c; ++lhs_c) {
        const auto& expected_result = expected_L_abc.get(lhs_a, lhs_b, lhs_c);

        CHECK(L_abc.get(lhs_a, lhs_b, lhs_c) == expected_result);
        CHECK(L_acb.get(lhs_a, lhs_c, lhs_b) == expected_result);
        CHECK(L_bac.get(lhs_b, lhs_a, lhs_c) == expected_result);
        CHECK(L_bca.get(lhs_b, lhs_c, lhs_a) == expected_result);
        CHECK(L_cab.get(lhs_c, lhs_a, lhs_b) == expected_result);
        CHECK(L_cba.get(lhs_c, lhs_b, lhs_a) == expected_result);
      }
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    // TODO combine all of these into one Variables

    // L_{abc} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_abc_type>>> L_abc_var{
        used_for_size};
    L_abc_type& L_abc_temp = get<::Tags::TempTensor<1, L_abc_type>>(L_abc_var);
    std::fill(L_abc_temp.begin(), L_abc_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC>(
        make_not_null(&L_abc_temp), rhs_expression);

    // L_{acb} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_acb_type>>> L_acb_var{
        used_for_size};
    L_acb_type& L_acb_temp = get<::Tags::TempTensor<1, L_acb_type>>(L_acb_var);
    std::fill(L_acb_temp.begin(), L_acb_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexB>(
        make_not_null(&L_acb_temp), rhs_expression);

    // L_{bac} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bac_type>>> L_bac_var{
        used_for_size};
    L_bac_type& L_bac_temp = get<::Tags::TempTensor<1, L_bac_type>>(L_bac_var);
    std::fill(L_bac_temp.begin(), L_bac_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexC>(
        make_not_null(&L_bac_temp), rhs_expression);

    // L_{bca} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bca_type>>> L_bca_var{
        used_for_size};
    L_bca_type& L_bca_temp = get<::Tags::TempTensor<1, L_bca_type>>(L_bca_var);
    std::fill(L_bca_temp.begin(), L_bca_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexA>(
        make_not_null(&L_bca_temp), rhs_expression);

    // L_{cab} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cab_type>>> L_cab_var{
        used_for_size};
    L_cab_type& L_cab_temp = get<::Tags::TempTensor<1, L_cab_type>>(L_cab_var);
    std::fill(L_cab_temp.begin(), L_cab_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexB>(
        make_not_null(&L_cab_temp), rhs_expression);

    // L_{cba} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cba_type>>> L_cba_var{
        used_for_size};
    L_cba_type& L_cba_temp = get<::Tags::TempTensor<1, L_cba_type>>(L_cba_var);
    std::fill(L_cba_temp.begin(), L_cba_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexA>(
        make_not_null(&L_cba_temp), rhs_expression);

    for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
      for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
        for (size_t lhs_c = 0; lhs_c < dim_c; ++lhs_c) {
          const auto& expected_result = expected_L_abc.get(lhs_a, lhs_b, lhs_c);

          CHECK(L_abc_temp.get(lhs_a, lhs_b, lhs_c) == expected_result);
          CHECK(L_acb_temp.get(lhs_a, lhs_c, lhs_b) == expected_result);
          CHECK(L_bac_temp.get(lhs_b, lhs_a, lhs_c) == expected_result);
          CHECK(L_bca_temp.get(lhs_b, lhs_c, lhs_a) == expected_result);
          CHECK(L_cab_temp.get(lhs_c, lhs_a, lhs_b) == expected_result);
          CHECK(L_cba_temp.get(lhs_c, lhs_b, lhs_a) == expected_result);
        }
      }
    }
  }
}
}  // namespace TestHelpers::tenex
