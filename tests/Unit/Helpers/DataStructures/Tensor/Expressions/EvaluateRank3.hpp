// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
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
template <typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& LhsTensorIndexA,
          auto& LhsTensorIndexB, auto& LhsTensorIndexC,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_3_impl() {
  const size_t used_for_size = 3;
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> R_abc(used_for_size);
  std::iota(R_abc.begin(), R_abc.end(),
            component_placeholder_value<DataType>::value);
  const DataType component_placeholder = R_abc[0];
  const auto rhs_expression =
      R_abc(LhsTensorIndexA, LhsTensorIndexB, LhsTensorIndexC);

  // Used for enforcing the ordering of the symmetry and TensorIndexTypes of the
  // LHS Tensor returned by `evaluate`
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
      get_index_value_range<lhs_tensorindextype_a, LhsTensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, LhsTensorIndexB>();
  lhs_index_value_ranges[2] =
      get_index_value_range<lhs_tensorindextype_c, LhsTensorIndexC>();
  std::array<std::pair<size_t, size_t>, 3> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, LhsTensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, LhsTensorIndexB>();
  rhs_index_value_ranges[2] =
      get_index_value_range<rhs_tensorindextype_c, LhsTensorIndexC>();
  std::array<bool, 3> shift_lhs_to_rhs_index_down{};
  shift_lhs_to_rhs_index_down[0] =
      lhs_index_value_ranges[0].first > lhs_index_value_ranges[0].first;
  shift_lhs_to_rhs_index_down[1] =
      lhs_index_value_ranges[1].first > lhs_index_value_ranges[1].first;
  shift_lhs_to_rhs_index_down[2] =
      lhs_index_value_ranges[2].first > lhs_index_value_ranges[2].first;

  // If we have the same index structure on the LHS and RHS, we can call the
  // `evaluate` overload that returns the LHS tensor. Otherwise, we need to call
  // the `evaluate` overload that accepts the LHS tensor as an argument to
  // override the LHS index structure that would have been deduced from the RHS
  // index structure.
  constexpr bool use_evaluate_that_returns_lhs =
      std::is_same_v<LhsSymmetry, RhsSymmetry> and
      std::is_same_v<LhsTensorIndexTypeList, RhsTensorIndexTypeList>;

  // L_{abc} = R_{abc}
  using L_abc_type = Tensor<DataType, RhsSymmetry, LhsTensorIndexTypeList>;
  L_abc_type L_abc;
  call_evaluate<use_evaluate_that_returns_lhs, LhsTensorIndexA, LhsTensorIndexB,
                LhsTensorIndexC>(make_not_null(&L_abc), rhs_expression);

  // L_{acb} = R_{abc}
  using L_acb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_b>;
  using L_acb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_b>;
  using L_acb_type =
      Tensor<DataType, L_acb_symmetry, L_acb_tensorindextype_list>;
  L_acb_type L_acb;
  call_evaluate<use_evaluate_that_returns_lhs, LhsTensorIndexA, LhsTensorIndexC,
                LhsTensorIndexB>(make_not_null(&L_acb), rhs_expression);

  // L_{bac} = R_{abc}
  using L_bac_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_c>;
  using L_bac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_c>;
  using L_bac_type =
      Tensor<DataType, L_bac_symmetry, L_bac_tensorindextype_list>;
  L_bac_type L_bac;
  call_evaluate<use_evaluate_that_returns_lhs, LhsTensorIndexB, LhsTensorIndexA,
                LhsTensorIndexC>(make_not_null(&L_bac), rhs_expression);

  // L_{bca} = R_{abc}
  using L_bca_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_a>;
  using L_bca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_a>;
  using L_bca_type =
      Tensor<DataType, L_bca_symmetry, L_bca_tensorindextype_list>;
  L_bca_type L_bca;
  call_evaluate<use_evaluate_that_returns_lhs, LhsTensorIndexB, LhsTensorIndexC,
                LhsTensorIndexA>(make_not_null(&L_bca), rhs_expression);

  // L_{cab} = R_{abc}
  using L_cab_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_b>;
  using L_cab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_b>;
  using L_cab_type =
      Tensor<DataType, L_cab_symmetry, L_cab_tensorindextype_list>;
  L_cab_type L_cab;
  call_evaluate<use_evaluate_that_returns_lhs, LhsTensorIndexC, LhsTensorIndexA,
                LhsTensorIndexB>(make_not_null(&L_cab), rhs_expression);

  // L_{cba} = R_{abc}
  using L_cba_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_a>;
  using L_cba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_a>;
  using L_cba_type =
      Tensor<DataType, L_cba_symmetry, L_cba_tensorindextype_list>;
  L_cba_type L_cba;
  call_evaluate<use_evaluate_that_returns_lhs, LhsTensorIndexC, LhsTensorIndexB,
                LhsTensorIndexA>(make_not_null(&L_cba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<LhsTensorIndexTypeList, 2>::dim;

  for (size_t lhs_i = 0; lhs_i < dim_a; ++lhs_i) {
    for (size_t lhs_j = 0; lhs_j < dim_b; ++lhs_j) {
      for (size_t lhs_k = 0; lhs_k < dim_c; ++lhs_k) {
        DataType expected_result;
        if (lhs_i < lhs_index_value_ranges[0].first or
            lhs_i > lhs_index_value_ranges[0].second or
            lhs_j < lhs_index_value_ranges[1].first or
            lhs_j > lhs_index_value_ranges[1].second or
            lhs_k < lhs_index_value_ranges[2].first or
            lhs_k > lhs_index_value_ranges[2].second) {
          expected_result = component_placeholder;
        } else {
          const size_t rhs_i = shift_lhs_to_rhs_index_down[0]
                                   ? lhs_i - rhs_index_value_ranges[0].first
                                   : lhs_i + rhs_index_value_ranges[0].first;
          const size_t rhs_j = shift_lhs_to_rhs_index_down[1]
                                   ? lhs_j - rhs_index_value_ranges[1].first
                                   : lhs_j + rhs_index_value_ranges[1].first;
          const size_t rhs_k = shift_lhs_to_rhs_index_down[2]
                                   ? lhs_k - rhs_index_value_ranges[2].first
                                   : lhs_k + rhs_index_value_ranges[2].first;
          expected_result = R_abc.get(rhs_i, rhs_j, rhs_k);
        }

        // L_{abc} = R_{abc}
        CHECK(L_abc.get(lhs_i, lhs_j, lhs_k) == expected_result);
        // L_{acb} = R_{abc}
        CHECK(L_acb.get(lhs_i, lhs_k, lhs_j) == expected_result);
        // L_{bac} = R_{abc}
        CHECK(L_bac.get(lhs_j, lhs_i, lhs_k) == expected_result);
        // L_{bca} = R_{abc}
        CHECK(L_bca.get(lhs_j, lhs_k, lhs_i) == expected_result);
        // L_{cab} = R_{abc}
        CHECK(L_cab.get(lhs_k, lhs_i, lhs_j) == expected_result);
        // L_{cba} = R_{abc}
        CHECK(L_cba.get(lhs_k, lhs_j, lhs_i) == expected_result);
      }
    }
  }

  // Test with TempTensor for LHS tensor
  if constexpr (not std::is_same_v<DataType, double>) {
    // L_{abc} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_abc_type>>> L_abc_var{
        used_for_size};
    L_abc_type& L_abc_temp = get<::Tags::TempTensor<1, L_abc_type>>(L_abc_var);
    ::tenex::evaluate<LhsTensorIndexA, LhsTensorIndexB, LhsTensorIndexC>(
        make_not_null(&L_abc_temp), rhs_expression);

    // L_{acb} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_acb_type>>> L_acb_var{
        used_for_size};
    L_acb_type& L_acb_temp = get<::Tags::TempTensor<1, L_acb_type>>(L_acb_var);
    ::tenex::evaluate<LhsTensorIndexA, LhsTensorIndexC, LhsTensorIndexB>(
        make_not_null(&L_acb_temp), rhs_expression);

    // L_{bac} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bac_type>>> L_bac_var{
        used_for_size};
    L_bac_type& L_bac_temp = get<::Tags::TempTensor<1, L_bac_type>>(L_bac_var);
    ::tenex::evaluate<LhsTensorIndexB, LhsTensorIndexA, LhsTensorIndexC>(
        make_not_null(&L_bac_temp), rhs_expression);

    // L_{bca} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bca_type>>> L_bca_var{
        used_for_size};
    L_bca_type& L_bca_temp = get<::Tags::TempTensor<1, L_bca_type>>(L_bca_var);
    ::tenex::evaluate<LhsTensorIndexB, LhsTensorIndexC, LhsTensorIndexA>(
        make_not_null(&L_bca_temp), rhs_expression);

    // L_{cab} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cab_type>>> L_cab_var{
        used_for_size};
    L_cab_type& L_cab_temp = get<::Tags::TempTensor<1, L_cab_type>>(L_cab_var);
    ::tenex::evaluate<LhsTensorIndexC, LhsTensorIndexA, LhsTensorIndexB>(
        make_not_null(&L_cab_temp), rhs_expression);

    // L_{cba} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cba_type>>> L_cba_var{
        used_for_size};
    L_cba_type& L_cba_temp = get<::Tags::TempTensor<1, L_cba_type>>(L_cba_var);
    ::tenex::evaluate<LhsTensorIndexC, LhsTensorIndexB, LhsTensorIndexA>(
        make_not_null(&L_cba_temp), rhs_expression);

    for (size_t lhs_i = 0; lhs_i < dim_a; ++lhs_i) {
      for (size_t lhs_j = 0; lhs_j < dim_b; ++lhs_j) {
        for (size_t lhs_k = 0; lhs_k < dim_c; ++lhs_k) {
          DataType expected_result;
          if (lhs_i < lhs_index_value_ranges[0].first or
              lhs_i > lhs_index_value_ranges[0].second or
              lhs_j < lhs_index_value_ranges[1].first or
              lhs_j > lhs_index_value_ranges[1].second or
              lhs_k < lhs_index_value_ranges[2].first or
              lhs_k > lhs_index_value_ranges[2].second) {
            expected_result = component_placeholder;
          } else {
            const size_t rhs_i = shift_lhs_to_rhs_index_down[0]
                                     ? lhs_i - rhs_index_value_ranges[0].first
                                     : lhs_i + rhs_index_value_ranges[0].first;
            const size_t rhs_j = shift_lhs_to_rhs_index_down[1]
                                     ? lhs_j - rhs_index_value_ranges[1].first
                                     : lhs_j + rhs_index_value_ranges[1].first;
            const size_t rhs_k = shift_lhs_to_rhs_index_down[2]
                                     ? lhs_k - rhs_index_value_ranges[2].first
                                     : lhs_k + rhs_index_value_ranges[2].first;
            expected_result = R_abc.get(rhs_i, rhs_j, rhs_k);
          }

          // L_{abc} = R_{abc}
          CHECK(L_abc_temp.get(lhs_i, lhs_j, lhs_k) == expected_result);
          // L_{acb} = R_{abc}
          CHECK(L_acb_temp.get(lhs_i, lhs_k, lhs_j) == expected_result);
          // L_{bac} = R_{abc}
          CHECK(L_bac_temp.get(lhs_j, lhs_i, lhs_k) == expected_result);
          // L_{bca} = R_{abc}
          CHECK(L_bca_temp.get(lhs_j, lhs_k, lhs_i) == expected_result);
          // L_{cab} = R_{abc}
          CHECK(L_cab_temp.get(lhs_k, lhs_i, lhs_j) == expected_result);
          // L_{cba} = R_{abc}
          CHECK(L_cba_temp.get(lhs_k, lhs_j, lhs_i) == expected_result);
        }
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Iterate testing of evaluating single rank 3 Tensors on multiple Frame
/// types and dimension combinations
///
/// We test various different symmetries across several functions to ensure that
/// the code works correctly with symmetries. This function tests one of the
/// following symmetries:
/// - <3, 2, 1> (`test_evaluate_rank_3_no_symmetry`)
/// - <2, 2, 1> (`test_evaluate_rank_3_ab_symmetry`)
/// - <2, 1, 2> (`test_evaluate_rank_3_ac_symmetry`)
/// - <2, 1, 1> (`test_evaluate_rank_3_bc_symmetry`)
/// - <1, 1, 1> (`test_evaluate_rank_3_abc_symmetry`)
///
/// \details `TensorIndexA`, `TensorIndexB`, and `TensorIndexC` can be any type
/// of TensorIndex and are not necessarily `ti::a`, `ti::b`, and `ti::c`. The
/// "A", "B", and "C" suffixes just denote the ordering of the generic indices
/// of the RHS tensor expression. In the RHS tensor expression, it means
/// `TensorIndexA` is the first index used, `TensorIndexB` is the second index
/// used, and `TensorIndexC` is the third index used.
///
///
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsTensorIndexTypeA the \ref SpacetimeIndex "RhsTensorIndexType" of
/// the first index of the RHS Tensor \tparam RhsTensorIndexTypeB the \ref
/// SpacetimeIndex "RhsTensorIndexType" of the second index of the RHS Tensor
/// \tparam RhsTensorIndexTypeC the \ref SpacetimeIndex "RhsTensorIndexType" of
/// the third index of the RHS Tensor \tparam TensorIndexA the first TensorIndex
/// used on the RHS of the TensorExpression, e.g. `ti::a` \tparam TensorIndexB
/// the second TensorIndex used on the RHS of the TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
template <typename DataType, typename Frame,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeA,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeB,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeC,
          auto& LhsTensorIndexA, auto& LhsTensorIndexB, auto& LhsTensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeA,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeC>
void test_evaluate_rank_3_no_symmetry() {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(2, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                              \
  test_evaluate_rank_3_impl<                                                 \
      DataType, Symmetry<3, 2, 1>,                                           \
      index_list<                                                            \
          RhsTensorIndexTypeA<DIM_A(data), LhsTensorIndexA.valence, Frame>,  \
          RhsTensorIndexTypeB<DIM_B(data), LhsTensorIndexB.valence, Frame>,  \
          RhsTensorIndexTypeC<DIM_C(data), LhsTensorIndexC.valence, Frame>>, \
      LhsTensorIndexA, LhsTensorIndexB, LhsTensorIndexC, Symmetry<3, 2, 1>,  \
      index_list<                                                            \
          LhsTensorIndexTypeA<DIM_A(data), LhsTensorIndexA.valence, Frame>,  \
          LhsTensorIndexTypeB<DIM_B(data), LhsTensorIndexB.valence, Frame>,  \
          LhsTensorIndexTypeC<DIM_C(data), LhsTensorIndexC.valence,          \
                              Frame>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3),
                          (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM_C
#undef DIM_B
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType, typename Frame,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeAB,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeC,
          auto& LhsTensorIndexA, auto& LhsTensorIndexB, auto& LhsTensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeAB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeAB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeC>
void test_evaluate_rank_3_ab_symmetry() {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                             \
  {                                                                         \
    using rhs_symmetry = Symmetry<2, 2, 1>;                                 \
    using rhs_index_list = index_list<                                      \
        RhsTensorIndexTypeAB<DIM_AB(data), LhsTensorIndexA.valence, Frame>, \
        RhsTensorIndexTypeAB<DIM_AB(data), LhsTensorIndexB.valence, Frame>, \
        RhsTensorIndexTypeC<DIM_C(data), LhsTensorIndexC.valence, Frame>>;  \
    using lhs_index_list = index_list<                                      \
        LhsTensorIndexTypeA<DIM_AB(data), LhsTensorIndexA.valence, Frame>,  \
        LhsTensorIndexTypeB<DIM_AB(data), LhsTensorIndexB.valence, Frame>,  \
        LhsTensorIndexTypeC<DIM_C(data), LhsTensorIndexC.valence, Frame>>;  \
                                                                            \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,       \
                              LhsTensorIndexA, LhsTensorIndexB,             \
                              LhsTensorIndexC, Symmetry<2, 2, 1>,           \
                              lhs_index_list>();                            \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,       \
                              LhsTensorIndexA, LhsTensorIndexB,             \
                              LhsTensorIndexC, Symmetry<3, 2, 1>,           \
                              lhs_index_list>();                            \
  }

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM_C
#undef DIM_AB
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType, typename Frame,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeAC,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeB,
          auto& LhsTensorIndexA, auto& LhsTensorIndexB, auto& LhsTensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeAC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeAC>
void test_evaluate_rank_3_ac_symmetry() {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                              \
  {                                                                          \
    using rhs_symmetry = Symmetry<1, 2, 1>;                                  \
    using rhs_index_list = index_list<                                       \
        RhsTensorIndexTypeAC<DIM_AC(data), LhsTensorIndexA.valence, Frame>,  \
        RhsTensorIndexTypeB<DIM_B(data), LhsTensorIndexB.valence, Frame>,    \
        RhsTensorIndexTypeAC<DIM_AC(data), LhsTensorIndexC.valence, Frame>>; \
    using lhs_index_list = index_list<                                       \
        LhsTensorIndexTypeA<DIM_AC(data), LhsTensorIndexA.valence, Frame>,   \
        LhsTensorIndexTypeB<DIM_B(data), LhsTensorIndexB.valence, Frame>,    \
        LhsTensorIndexTypeC<DIM_AC(data), LhsTensorIndexC.valence, Frame>>;  \
                                                                             \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,        \
                              LhsTensorIndexA, LhsTensorIndexB,              \
                              LhsTensorIndexC, Symmetry<1, 2, 1>,            \
                              lhs_index_list>();                             \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,        \
                              LhsTensorIndexA, LhsTensorIndexB,              \
                              LhsTensorIndexC, Symmetry<3, 2, 1>,            \
                              lhs_index_list>();                             \
  }

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM_B
#undef DIM_AC
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType, typename Frame,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeA,
          template <size_t, UpLo, typename> class RhsTensorIndexTypeBC,
          auto& LhsTensorIndexA, auto& LhsTensorIndexB, auto& LhsTensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeA,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeBC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeBC>
void test_evaluate_rank_3_bc_symmetry() {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                              \
  {                                                                          \
    using rhs_symmetry = Symmetry<2, 1, 1>;                                  \
    using rhs_index_list = index_list<                                       \
        RhsTensorIndexTypeA<DIM_A(data), LhsTensorIndexA.valence, Frame>,    \
        RhsTensorIndexTypeBC<DIM_BC(data), LhsTensorIndexB.valence, Frame>,  \
        RhsTensorIndexTypeBC<DIM_BC(data), LhsTensorIndexC.valence, Frame>>; \
    using lhs_index_list = index_list<                                       \
        LhsTensorIndexTypeA<DIM_A(data), LhsTensorIndexA.valence, Frame>,    \
        LhsTensorIndexTypeB<DIM_BC(data), LhsTensorIndexB.valence, Frame>,   \
        LhsTensorIndexTypeC<DIM_BC(data), LhsTensorIndexC.valence, Frame>>;  \
                                                                             \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,        \
                              LhsTensorIndexA, LhsTensorIndexB,              \
                              LhsTensorIndexC, Symmetry<2, 1, 1>,            \
                              lhs_index_list>();                             \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,        \
                              LhsTensorIndexA, LhsTensorIndexB,              \
                              LhsTensorIndexC, Symmetry<3, 2, 1>,            \
                              lhs_index_list>();                             \
  }

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM_BC
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType, typename Frame,
          template <size_t, UpLo, typename> class RhsTensorIndexType,
          auto& LhsTensorIndexA, auto& LhsTensorIndexB, auto& LhsTensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexType,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexType,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexType>
void test_evaluate_rank_3_abc_symmetry() {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                          \
  {                                                                      \
    using rhs_symmetry = Symmetry<1, 1, 1>;                              \
    using rhs_index_list = index_list<                                   \
        RhsTensorIndexType<DIM(data), LhsTensorIndexA.valence, Frame>,   \
        RhsTensorIndexType<DIM(data), LhsTensorIndexB.valence, Frame>,   \
        RhsTensorIndexType<DIM(data), LhsTensorIndexC.valence, Frame>>;  \
    using lhs_index_list = index_list<                                   \
        LhsTensorIndexTypeA<DIM(data), LhsTensorIndexA.valence, Frame>,  \
        LhsTensorIndexTypeB<DIM(data), LhsTensorIndexB.valence, Frame>,  \
        LhsTensorIndexTypeC<DIM(data), LhsTensorIndexC.valence, Frame>>; \
                                                                         \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,    \
                              LhsTensorIndexA, LhsTensorIndexB,          \
                              LhsTensorIndexC, Symmetry<1, 1, 1>,        \
                              lhs_index_list>();                         \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,    \
                              LhsTensorIndexA, LhsTensorIndexB,          \
                              LhsTensorIndexC, Symmetry<2, 1, 1>,        \
                              lhs_index_list>();                         \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,    \
                              LhsTensorIndexA, LhsTensorIndexB,          \
                              LhsTensorIndexC, Symmetry<1, 2, 1>,        \
                              lhs_index_list>();                         \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,    \
                              LhsTensorIndexA, LhsTensorIndexB,          \
                              LhsTensorIndexC, Symmetry<1, 1, 2>,        \
                              lhs_index_list>();                         \
    test_evaluate_rank_3_impl<DataType, rhs_symmetry, rhs_index_list,    \
                              LhsTensorIndexA, LhsTensorIndexB,          \
                              LhsTensorIndexC, Symmetry<3, 2, 1>,        \
                              lhs_index_list>();                         \
  }

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM
}

}  // namespace TestHelpers::tenex
