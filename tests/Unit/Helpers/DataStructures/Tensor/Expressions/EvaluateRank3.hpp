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

// TODO : instead, return beginning and end indices
template <typename LhsIndexType, typename RhsIndexType>
constexpr int get_spatial_spacetime_index_shift() {
  if constexpr (LhsIndexType::index_type == RhsIndexType::index_type) {
    return 0;
  } else if constexpr (LhsIndexType::index_type == IndexType::Spatial) {
    return -1;
  } else {
    return 1;
  }
  //   return static_cast<int>(RhsIndexType::index_type == IndexType::Spacetime)
  //   -
  //              static_cast<int>(LhsIndexType::index_type ==
  //              IndexType::Spacetime);
}

// TODO : see if we can do two ... variadic of Indexs and TensorIndexs since TensorIndex
// is auto& and may be recognized as different?
template <typename Index, auto& TensorIndex>
constexpr size_t get_start_index_value() {
    return Index::index_type == IndexType::Spacetime and
          not TensorIndex.is_spacetime ? 1 : 0;
}

// template <typename LhsIndexType, typename RhsIndexType>
// constexpr int get_start_index_values() {
//   if constexpr (LhsIndexType::index_type == RhsIndexType::index_type) {
//     return 0;
//   } else if constexpr (LhsIndexType::index_type == IndexType::Spatial) {
//     return -1;
//   } else {
//     return 1;
//   }
//   //   return static_cast<int>(RhsIndexType::index_type == IndexType::Spacetime)
//   //   -
//   //              static_cast<int>(LhsIndexType::index_type ==
//   //              IndexType::Spacetime);
// }

// template <size_t NumIndices>
// constexpr int get_spatial_spacetime_multi_index_shift() {
//   std::array<int, NumIndices> multi_index_shift{};
//   for (size_t int = 0; i < NumIndices; i++) {
//     multi_index_shift[i] = get_spatial_spacetime_index_shift
//   }
//   return static_cast<int>(RhsIndexType::index_type == IndexType::Spacetime) -
//              static_cast<int>(LhsIndexType::index_type ==
//              IndexType::Spacetime);
// }

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
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_3_impl() {
  const size_t used_for_size = 3;
  Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList> R_abc(used_for_size);
  std::iota(R_abc.begin(), R_abc.end(), 0.0);
  const auto rhs_expression = R_abc(TensorIndexA, TensorIndexB, TensorIndexC);

  // Used for enforcing the ordering of the symmetry and TensorIndexTypes of the
  // LHS Tensor returned by `evaluate`
  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<RhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<RhsSymmetry, 1>::value;
  const std::int32_t lhs_symmetry_element_c = tmpl::at_c<RhsSymmetry, 2>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using lhs_tensorindextype_c = tmpl::at_c<LhsTensorIndexTypeList, 2>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;
//   std::array<std::int32_t, 3> multi_index_shift{};
//   multi_index_shift[0] = get_spatial_spacetime_index_shift<
//       lhs_tensorindextype_a, tmpl::at_c<RhsTensorIndexTypeList, 0>>();
//   multi_index_shift[1] = get_spatial_spacetime_index_shift<
//       lhs_tensorindextype_a, tmpl::at_c<RhsTensorIndexTypeList, 1>>();
//   multi_index_shift[2] = get_spatial_spacetime_index_shift<
//       lhs_tensorindextype_a, tmpl::at_c<RhsTensorIndexTypeList, 2>>();


  std::array<size_t, 3> lhs_start_index_values{};
  lhs_start_index_values[0] =
      get_start_index_value<lhs_tensorindextype_a, TensorIndexA>();
  lhs_start_index_values[1] =
      get_start_index_value<lhs_tensorindextype_b, TensorIndexB>();
  lhs_start_index_values[2] =
      get_start_index_value<lhs_tensorindextype_c, TensorIndexC>();
  std::array<size_t, 3> rhs_start_index_values{};
  rhs_start_index_values[0] =
      get_start_index_value<rhs_tensorindextype_a, TensorIndexA>();
  rhs_start_index_values[1] =
      get_start_index_value<rhs_tensorindextype_b, TensorIndexB>();
  rhs_start_index_values[2] =
      get_start_index_value<rhs_tensorindextype_c, TensorIndexC>();
//   std::array<std::int32_t, 3> index_value_shifts{};
//   index_value_shifts[0] =
//       get_start_index_value<lhs_tensorindextype_a, TensorIndexA>();
//   index_value_shifts[1] =
//       get_start_index_value<lhs_tensorindextype_b, TensorIndexB>();
//   index_value_shifts[2] =
//       get_start_index_value<lhs_tensorindextype_c, TensorIndexC>();
  std::array<bool, 3> shift_lhs_to_rhs_index_down{};
  shift_lhs_to_rhs_index_down[0] =
      lhs_start_index_values[0] > rhs_start_index_values[0];
  shift_lhs_to_rhs_index_down[1] =
      lhs_start_index_values[1] > rhs_start_index_values[1];
  shift_lhs_to_rhs_index_down[2] =
      lhs_start_index_values[2] > rhs_start_index_values[2];

//   std::array<bool, 3> lhs_index_is_spacetime{};
//   lhs_index_is_spacetime[0] =
//       lhs_tensorindextype_a::index_type == IndexType::Spacetime;
//   lhs_index_is_spacetime[1] =
//       lhs_tensorindextype_b::index_type == IndexType::Spacetime;
//   lhs_index_is_spacetime[2] =
//       lhs_tensorindextype_c::index_type == IndexType::Spacetime;
//   std::array<std::int32_t, 3> rhs_start_index_values{};
//   rhs_start_index_values[0] =
//       rhs_tensorindextype_a::index_type == IndexType::Spacetime;
//   rhs_start_index_values[1] =
//       rhs_tensorindextype_b::index_type == IndexType::Spacetime;
//   rhs_start_index_values[2] =
//       rhs_tensorindextype_c::index_type == IndexType::Spacetime;

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
  call_evaluate<use_evaluate_that_returns_lhs, TensorIndexA, TensorIndexB,
                TensorIndexC>(make_not_null(&L_abc), rhs_expression);

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
  call_evaluate<use_evaluate_that_returns_lhs, TensorIndexA, TensorIndexC,
                TensorIndexB>(make_not_null(&L_acb), rhs_expression);

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
  call_evaluate<use_evaluate_that_returns_lhs, TensorIndexB, TensorIndexA,
                TensorIndexC>(make_not_null(&L_bac), rhs_expression);

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
  call_evaluate<use_evaluate_that_returns_lhs, TensorIndexB, TensorIndexC,
                TensorIndexA>(make_not_null(&L_bca), rhs_expression);

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
  call_evaluate<use_evaluate_that_returns_lhs, TensorIndexC, TensorIndexA,
                TensorIndexB>(make_not_null(&L_cab), rhs_expression);

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
  call_evaluate<use_evaluate_that_returns_lhs, TensorIndexC, TensorIndexB,
                TensorIndexA>(make_not_null(&L_cba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<LhsTensorIndexTypeList, 2>::dim;

//   for (size_t rhs_i = rhs_start_index_values[0], lhs_i = lhs_start_index_values[0];
//        rhs_i < dim_a; ++rhs_i, ++lhs_i) {
//     for (size_t rhs_j = rhs_start_index_values[1], lhs_j = lhs_start_index_values[1];
//          rhs_j < dim_b; ++rhs_j, ++lhs_j) {
//       for (size_t rhs_k = rhs_start_index_values[2], lhs_k = lhs_start_index_values[2];
//            rhs_k < dim_c; ++rhs_k, ++lhs_k) {
    for (size_t i = 0; 0 < dim_a; ++i) {
    for (size_t j = 0; 0 < dim_b; ++j) {
      for (size_t k = 0; k < dim_c; ++k) {
        DataType expected_result;
        if ((i == 0 and lhs_start_index_values[0] == 1) or
            (j == 0 and lhs_start_index_values[1] == 1) or
            (k == 0 and lhs_start_index_values[2] == 1)) {
           expected_result = component_placeholder_value<DataType>::value;
        } else {
          const size_t rhs_i =
          shift_lhs_to_rhs_index_down[0] ? i - rhs_start_index_values[0] : i + rhs_start_index_values[0];
        const size_t rhs_j =
            shift_lhs_to_rhs_index_down[1] ? j - rhs_start_index_values[1] : j + rhs_start_index_values[1];
        const size_t rhs_k =
          shift_lhs_to_rhs_index_down[2] ? k - rhs_start_index_values[2] : k + rhs_start_index_values[2];
           expected_result = R_abc.get(
            rhs_i,
            rhs_j,
            rhs_k);
        }

        const size_t lhs_i = i;// + lhs_start_index_values[0];
        const size_t lhs_j = j;// + lhs_start_index_values[1];
        const size_t lhs_k = k;// + lhs_start_index_values[2];

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
    ::tenex::evaluate<TensorIndexA, TensorIndexB, TensorIndexC>(
        make_not_null(&L_abc_temp), rhs_expression);

    // L_{acb} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_acb_type>>> L_acb_var{
        used_for_size};
    L_acb_type& L_acb_temp = get<::Tags::TempTensor<1, L_acb_type>>(L_acb_var);
    ::tenex::evaluate<TensorIndexA, TensorIndexC, TensorIndexB>(
        make_not_null(&L_acb_temp), rhs_expression);

    // L_{bac} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bac_type>>> L_bac_var{
        used_for_size};
    L_bac_type& L_bac_temp = get<::Tags::TempTensor<1, L_bac_type>>(L_bac_var);
    ::tenex::evaluate<TensorIndexB, TensorIndexA, TensorIndexC>(
        make_not_null(&L_bac_temp), rhs_expression);

    // L_{bca} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_bca_type>>> L_bca_var{
        used_for_size};
    L_bca_type& L_bca_temp = get<::Tags::TempTensor<1, L_bca_type>>(L_bca_var);
    ::tenex::evaluate<TensorIndexB, TensorIndexC, TensorIndexA>(
        make_not_null(&L_bca_temp), rhs_expression);

    // L_{cab} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cab_type>>> L_cab_var{
        used_for_size};
    L_cab_type& L_cab_temp = get<::Tags::TempTensor<1, L_cab_type>>(L_cab_var);
    ::tenex::evaluate<TensorIndexC, TensorIndexA, TensorIndexB>(
        make_not_null(&L_cab_temp), rhs_expression);

    // L_{cba} = R_{abc}
    Variables<tmpl::list<::Tags::TempTensor<1, L_cba_type>>> L_cba_var{
        used_for_size};
    L_cba_type& L_cba_temp = get<::Tags::TempTensor<1, L_cba_type>>(L_cba_var);
    ::tenex::evaluate<TensorIndexC, TensorIndexB, TensorIndexA>(
        make_not_null(&L_cba_temp), rhs_expression);

    // for (size_t i = 0, lhs_i = static_cast<size_t>(multi_index_shift[0] + i);
    //      i < dim_a; ++i, ++lhs_i) {
    //   for (size_t j = 0, lhs_j = static_cast<size_t>(multi_index_shift[1] + j);
    //        rhs+j < dim_b; ++j, ++lhs_j) {
    //     for (size_t rhs_k = 0,
    //                 lhs_k = static_cast<size_t>(multi_index_shift[2] + rhs_k);
    //          rhs_k < dim_c; ++k, ++lhs_k) {
    //       // L_{abc} = R_{abc}, check that L_{ijk} == R_{ijk}
    //       CHECK(L_abc_temp.get(lhs_i, lhs_j, lhs_k) == expected_result);
    //       // L_{acb} = R_{abc}, check that L_{ikj} == R_{ijk}
    //       CHECK(L_acb_temp.get(lhs_i, lhs_k, lhs_j) == expected_result);
    //       // L_{bac} = R_{abc}, check that L_{jik} == R_{ijk}
    //       CHECK(L_bac_temp.get(lhs_j, lhs_i, lhs_k) == expected_result);
    //       // L_{bca} = R_{abc}, check that L_{jki} == R_{ijk}
    //       CHECK(L_bca_temp.get(lhs_j, lhs_k, lhs_i) == expected_result);
    //       // L_{cab} = R_{abc}, check that L_{kij} == R_{ijk}
    //       CHECK(L_cab_temp.get(lhs_k, lhs_i, lhs_j) == expected_result);
    //       // L_{cba} = R_{abc}, check that L_{kji} == R_{ijk}
    //       CHECK(L_cba_temp.get(lhs_k, lhs_j, lhs_i) == expected_result);
    //     }
    //   }
    // }
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
          auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC,
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

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                           \
  test_evaluate_rank_3_impl<                                              \
      DataType, Symmetry<3, 2, 1>,                                        \
      index_list<                                                         \
          RhsTensorIndexTypeA<DIM_A(data), TensorIndexA.valence, Frame>,  \
          RhsTensorIndexTypeB<DIM_B(data), TensorIndexB.valence, Frame>,  \
          RhsTensorIndexTypeC<DIM_C(data), TensorIndexC.valence, Frame>>, \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<3, 2, 1>,        \
      index_list<                                                         \
          LhsTensorIndexTypeA<DIM_A(data), TensorIndexA.valence, Frame>,  \
          LhsTensorIndexTypeB<DIM_B(data), TensorIndexB.valence, Frame>,  \
          LhsTensorIndexTypeC<DIM_C(data), TensorIndexC.valence, Frame>>>();

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
          auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeAB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeAB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeC>
void test_evaluate_rank_3_ab_symmetry() {
#define DIM_AB(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_C(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                              \
  test_evaluate_rank_3_impl<                                                 \
      DataType, Symmetry<2, 2, 1>,                                           \
      index_list<                                                            \
          RhsTensorIndexTypeAB<DIM_AB(data), TensorIndexA.valence, Frame>,   \
          RhsTensorIndexTypeAB<DIM_AB(data), TensorIndexB.valence, Frame>,   \
          RhsTensorIndexTypeC<DIM_C(data), TensorIndexC.valence, Frame>>,    \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<2, 2, 1>,           \
      index_list<                                                            \
          LhsTensorIndexTypeA<DIM_AB(data), TensorIndexA.valence, Frame>,    \
          LhsTensorIndexTypeB<DIM_AB(data), TensorIndexB.valence, Frame>,    \
          LhsTensorIndexTypeC<DIM_C(data), TensorIndexC.valence, Frame>>>(); \
  test_evaluate_rank_3_impl<                                                 \
      DataType, Symmetry<2, 2, 1>,                                           \
      index_list<                                                            \
          RhsTensorIndexTypeAB<DIM_AB(data), TensorIndexA.valence, Frame>,   \
          RhsTensorIndexTypeAB<DIM_AB(data), TensorIndexB.valence, Frame>,   \
          RhsTensorIndexTypeC<DIM_C(data), TensorIndexC.valence, Frame>>,    \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<3, 2, 1>,           \
      index_list<                                                            \
          LhsTensorIndexTypeA<DIM_AB(data), TensorIndexA.valence, Frame>,    \
          LhsTensorIndexTypeB<DIM_AB(data), TensorIndexB.valence, Frame>,    \
          LhsTensorIndexTypeC<DIM_C(data), TensorIndexC.valence, Frame>>>();

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
          auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeAC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeB,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeAC>
void test_evaluate_rank_3_ac_symmetry() {
#define DIM_AC(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_B(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                               \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 2, 1>,                                            \
      index_list<                                                             \
          RhsTensorIndexTypeAC<DIM_AC(data), TensorIndexA.valence, Frame>,    \
          RhsTensorIndexTypeB<DIM_B(data), TensorIndexB.valence, Frame>,      \
          RhsTensorIndexTypeAC<DIM_AC(data), TensorIndexC.valence, Frame>>,   \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<1, 2, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM_AC(data), TensorIndexA.valence, Frame>,     \
          LhsTensorIndexTypeB<DIM_B(data), TensorIndexB.valence, Frame>,      \
          LhsTensorIndexTypeC<DIM_AC(data), TensorIndexC.valence, Frame>>>(); \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 2, 1>,                                            \
      index_list<                                                             \
          RhsTensorIndexTypeAC<DIM_AC(data), TensorIndexA.valence, Frame>,    \
          RhsTensorIndexTypeB<DIM_B(data), TensorIndexB.valence, Frame>,      \
          RhsTensorIndexTypeAC<DIM_AC(data), TensorIndexC.valence, Frame>>,   \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<3, 2, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM_AC(data), TensorIndexA.valence, Frame>,     \
          LhsTensorIndexTypeB<DIM_B(data), TensorIndexB.valence, Frame>,      \
          LhsTensorIndexTypeC<DIM_AC(data), TensorIndexC.valence, Frame>>>();

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
          auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexTypeA,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexTypeBC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexTypeBC>
void test_evaluate_rank_3_bc_symmetry() {
#define DIM_A(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM_BC(data) BOOST_PP_TUPLE_ELEM(1, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                               \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<2, 1, 1>,                                            \
      index_list<                                                             \
          RhsTensorIndexTypeA<DIM_A(data), TensorIndexA.valence, Frame>,      \
          RhsTensorIndexTypeBC<DIM_BC(data), TensorIndexB.valence, Frame>,    \
          RhsTensorIndexTypeBC<DIM_BC(data), TensorIndexC.valence, Frame>>,   \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<2, 1, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM_A(data), TensorIndexA.valence, Frame>,      \
          LhsTensorIndexTypeB<DIM_BC(data), TensorIndexB.valence, Frame>,     \
          LhsTensorIndexTypeC<DIM_BC(data), TensorIndexC.valence, Frame>>>(); \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<2, 1, 1>,                                            \
      index_list<                                                             \
          RhsTensorIndexTypeA<DIM_A(data), TensorIndexA.valence, Frame>,      \
          RhsTensorIndexTypeBC<DIM_BC(data), TensorIndexB.valence, Frame>,    \
          RhsTensorIndexTypeBC<DIM_BC(data), TensorIndexC.valence, Frame>>,   \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<3, 2, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM_A(data), TensorIndexA.valence, Frame>,      \
          LhsTensorIndexTypeB<DIM_BC(data), TensorIndexB.valence, Frame>,     \
          LhsTensorIndexTypeC<DIM_BC(data), TensorIndexC.valence, Frame>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3), (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM_BC
#undef DIM_A
}

/// \ingroup TestingFrameworkGroup
/// \copydoc test_evaluate_rank_3_no_symmetry()
template <typename DataType, typename Frame,
          template <size_t, UpLo, typename> class RhsTensorIndexType,
          auto& TensorIndexA, auto& TensorIndexB, auto& TensorIndexC,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeA = RhsTensorIndexType,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeB = RhsTensorIndexType,
          template <size_t, UpLo, typename>
          class LhsTensorIndexTypeC = RhsTensorIndexType>
void test_evaluate_rank_3_abc_symmetry() {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define CALL_TEST_EVALUATE_RANK_3_IMPL(_, data)                               \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 1, 1>,                                            \
      index_list<RhsTensorIndexType<DIM(data), TensorIndexA.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexB.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexC.valence, Frame>>, \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<1, 1, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM(data), TensorIndexA.valence, Frame>,        \
          LhsTensorIndexTypeB<DIM(data), TensorIndexB.valence, Frame>,        \
          LhsTensorIndexTypeC<DIM(data), TensorIndexC.valence, Frame>>>();    \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 1, 1>,                                            \
      index_list<RhsTensorIndexType<DIM(data), TensorIndexA.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexB.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexC.valence, Frame>>, \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<2, 1, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM(data), TensorIndexA.valence, Frame>,        \
          LhsTensorIndexTypeB<DIM(data), TensorIndexB.valence, Frame>,        \
          LhsTensorIndexTypeC<DIM(data), TensorIndexC.valence, Frame>>>();    \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 1, 1>,                                            \
      index_list<RhsTensorIndexType<DIM(data), TensorIndexA.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexB.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexC.valence, Frame>>, \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<1, 2, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM(data), TensorIndexA.valence, Frame>,        \
          LhsTensorIndexTypeB<DIM(data), TensorIndexB.valence, Frame>,        \
          LhsTensorIndexTypeC<DIM(data), TensorIndexC.valence, Frame>>>();    \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 1, 1>,                                            \
      index_list<RhsTensorIndexType<DIM(data), TensorIndexA.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexB.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexC.valence, Frame>>, \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<1, 1, 2>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM(data), TensorIndexA.valence, Frame>,        \
          LhsTensorIndexTypeB<DIM(data), TensorIndexB.valence, Frame>,        \
          LhsTensorIndexTypeC<DIM(data), TensorIndexC.valence, Frame>>>();    \
  test_evaluate_rank_3_impl<                                                  \
      DataType, Symmetry<1, 1, 1>,                                            \
      index_list<RhsTensorIndexType<DIM(data), TensorIndexA.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexB.valence, Frame>,  \
                 RhsTensorIndexType<DIM(data), TensorIndexC.valence, Frame>>, \
      TensorIndexA, TensorIndexB, TensorIndexC, Symmetry<3, 2, 1>,            \
      index_list<                                                             \
          LhsTensorIndexTypeA<DIM(data), TensorIndexA.valence, Frame>,        \
          LhsTensorIndexTypeB<DIM(data), TensorIndexB.valence, Frame>,        \
          LhsTensorIndexTypeC<DIM(data), TensorIndexC.valence, Frame>>>();

  GENERATE_INSTANTIATIONS(CALL_TEST_EVALUATE_RANK_3_IMPL, (1, 2, 3))

#undef CALL_TEST_EVALUATE_RANK_3_IMPL
#undef DIM
}

}  // namespace TestHelpers::tenex
