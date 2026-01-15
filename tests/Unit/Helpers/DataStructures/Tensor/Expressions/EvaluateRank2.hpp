// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
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
// TODO : add static_asserts that things are the right size (the rank)
// TODO : perhaps make one function that dispatches to the individual
// rank tests so we don't have to repeat the docs so much

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 2 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details `TensorIndexA` and `TensorIndexB` can be any type of TensorIndex
/// and are not necessarily `ti::a` and `ti::b`. The "A" and "B" suffixes just
/// denote the ordering of the generic indices of the RHS tensor expression. In
/// the RHS tensor expression, it means `TensorIndexA` is the first index used
/// and `TensorIndexB` is the second index used.
///
/// If we consider the RHS tensor's generic indices to be (a, b), then this test
/// checks that the data in the evaluated LHS tensor is correct according to the
/// index orders of the LHS and RHS. The two possible cases that are checked are
/// when the LHS tensor is evaluated with index order (a, b) and when it is
/// evaluated with the index order (b, a).
///
/// If `ReturnLhsTensor == true`, the `tenex::evaluate` overload that returns
/// the LHS tensor will be tested. This, in turn, includes testing whether
/// `tenex::evaluate` is deducing the correct LHS tensor return type, where
/// `LhsSymmetry` is its expected symmetry and `LhsTensorIndexType` is its
/// expected list of indices.
///
/// If `ReturnLhsTensor == false`, the `tenex::evaluate` overload that takes a
/// preallocated LHS tensor will be tested. In this case, `LhsSymmetry` and
/// `LhsTensorIndexList` can be different from and will override what would be
/// automatically deduced from the RHS tensor expression. This is useful for
/// testing evaluations where the desired LHS tensor type would not
/// automatically be deduced from the RHS expression. For example, given some
/// tensor \f$R_{ab}\f$ with two spacetime indices, one can test whether
/// \f$R_{ij} = ...\f$ correctly only assigns to the spatial-spatial
/// components of the tensor. Likewise, `ReturnLhsTensor == false` is
/// necessary to test cases where the LHS symmetry is different from what
/// would be deduced.
///
/// \param ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// preallocating the result tensor and filling it)
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_2_core() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  using R_ab_type = Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>;
  const auto R_ab = make_with_random_values<R_ab_type>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_ab =
      ReturnLhsTensor
          ? Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>{}
          : make_with_value<
                Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>>(
                used_for_size, component_placeholder_value<DataType>::value);

  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<LhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<LhsSymmetry, 1>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;

  std::array<std::pair<size_t, size_t>, 2> lhs_index_value_ranges{};
  lhs_index_value_ranges[0] =
      get_index_value_range<lhs_tensorindextype_a, TensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, TensorIndexB>();
  std::array<std::pair<size_t, size_t>, 2> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, TensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, TensorIndexB>();

  for (size_t lhs_a = lhs_index_value_ranges[0].first,
              rhs_a = rhs_index_value_ranges[0].first;
       lhs_a <= lhs_index_value_ranges[0].second; lhs_a++, rhs_a++) {
    for (size_t lhs_b = lhs_index_value_ranges[1].first,
                rhs_b = rhs_index_value_ranges[1].first;
         lhs_b <= lhs_index_value_ranges[1].second; lhs_b++, rhs_b++) {
      expected_L_ab.get(lhs_a, lhs_b) = R_ab.get(rhs_a, rhs_b);
    }
  }

  const auto rhs_expression = R_ab(TensorIndexA, TensorIndexB);
  // L_{ab} = R_{ab}
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  using L_ab_type = Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>;
  L_ab_type L_ab(used_for_size);
  std::fill(L_ab.begin(), L_ab.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB>(
      make_not_null(&L_ab), rhs_expression);

  // L_{ba} = R_{ab}
  using L_ba_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a>;
  using L_ba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a>;
  using L_ba_type = Tensor<DataType, L_ba_symmetry, L_ba_tensorindextype_list>;
  L_ba_type L_ba(used_for_size);
  std::fill(L_ba.begin(), L_ba.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA>(
      make_not_null(&L_ba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;

  // check LHS evaluated correctly
  for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
    for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
      const auto& expected_result = expected_L_ab.get(lhs_a, lhs_b);

      CHECK(L_ab.get(lhs_a, lhs_b) == expected_result);
      CHECK(L_ba.get(lhs_b, lhs_a) == expected_result);
    }
  }

  // TODO : change filling with NaN to filling with placeholder because
  // sometimes we purposely won't fill out a component

  // Test with Variables
  if constexpr (is_derived_of_vector_impl_v<DataType>) {
    Variables<tmpl::list<::Tags::TempTensor<0, R_ab_type>,
                         ::Tags::TempTensor<1, L_ab_type>,
                         ::Tags::TempTensor<2, L_ba_type>>>
        vars(used_for_size, std::numeric_limits<double>::signaling_NaN());

    R_ab_type& R_ab_temp = get<::Tags::TempTensor<0, R_ab_type>>(vars);
    R_ab_temp = R_ab;

    // L_{ab} = R_{ab}
    L_ab_type& L_ab_temp = get<::Tags::TempTensor<1, L_ab_type>>(vars);
    std::fill(L_ab_temp.begin(), L_ab_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB>(
        make_not_null(&L_ab_temp), rhs_expression);

    // L_{ba} = R_{ab}
    L_ba_type& L_ba_temp = get<::Tags::TempTensor<2, L_ba_type>>(vars);
    std::fill(L_ba_temp.begin(), L_ba_temp.end(),
              component_placeholder_value<DataType>::value);
    call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA>(
        make_not_null(&L_ba_temp), rhs_expression);

    // check RHS wasn't modified
    CHECK(R_ab_temp == R_ab);

    // check LHS evaluated correctly
    for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
      for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
        const auto& expected_result = expected_L_ab.get(lhs_a, lhs_b);

        CHECK(L_ab_temp.get(lhs_a, lhs_b) == expected_result);
        CHECK(L_ba_temp.get(lhs_b, lhs_a) == expected_result);
      }
    }
  }
}

// TODOTODOTODO: just scrap the idea of instantiating for every LHS symmetry

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 2 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details This simply runs `test_evaluate_rank_2_core` but if the RHS
/// tensor is symmetric, it is run twice: once for when the LHS tensor is
/// also symmetric and again for when it is not
///
/// \param ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// preallocating the result tensor and filling it)
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_2_impl() {
  using symmetry_21 = Symmetry<2, 1>;
  using symmetry_11 = Symmetry<1, 1>;

  // TODO : check for case where lhs is spatial and rhs is spacetime because
  // then we can't do RHS and LHS symmetry as <1, 1>. Logic here and in other
  // rank files should just handle what "all symmetries" is for the user

  // TODO : maybe we need to not let the user specify ReturnLhsTensor? well no
  // it's good to have but it can conflict with the symmetry and tensor index
  // type lists. Need to find a solution.

  test_evaluate_rank_2_core<ReturnLhsTensor, TensorIndexA, TensorIndexB,
                            DataType, RhsSymmetry, RhsTensorIndexTypeList,
                            RhsSymmetry, LhsTensorIndexTypeList>();
  if constexpr (std::is_same_v<RhsSymmetry, symmetry_11>) {
    test_evaluate_rank_2_core<false, TensorIndexA, TensorIndexB, DataType,
                              RhsSymmetry, RhsTensorIndexTypeList, symmetry_21,
                              LhsTensorIndexTypeList>();
  }

  // note: below is temp work

  // TODO: consider defining alias to RHS and LHS tensor type to make
  // getting porperties below easier. This will also catch if you create
  // a tensor that doesn't make sense, so this sounds like a good idea
  using rhs_structure =
      typename Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>::structure;
  // note : can't do this because LHS might have symmetry broken due to
  // usign spatial spacetime indices, for example
  // TODO : ok maybe figure this own then lol
  // using lhs_tensor = Tensor<DataType, RhsSymmetry, LhsTensorIndexList>
  // here's an idea:
  // using lhs_symmetry =
  //     std::conditional_t<TensorMetafunctions::check_index_symmetry_v<
  //                            RhsSymmetry, tmpl::at_c<RhsTensorIndexTypeList,
  //                            0>, tmpl::at_c<RhsTensorIndexTypeList, 1>>,
  //                        RhsSymmetry, Symmetry<2, 1>>;
  // using lhs_structure = typename Tensor<DataType, lhs_symmetry,
  //                                       LhsTensorIndexTypeList>::structure;

  if constexpr (std::is_same_v<RhsSymmetry, symmetry_11>) {
    // constexpr auto rhs_spatial_spacetime_index_positions =
    //     ::tenex::detail::get_spatial_spacetime_index_positions<
    //         RhsTensorIndexTypeList,
    //         make_tensorindex_list<TensorIndexA, TensorIndexB>>();

    // constexpr auto lhs_spatial_spacetime_index_positions =
    //     ::tenex::detail::get_spatial_spacetime_index_positions<
    //         LhsTensorIndexTypeList,
    //         make_tensorindex_list<TensorIndexA, TensorIndexB>>();

    // constexpr auto time_index_positions =
    //     ::tenex::detail::get_time_index_positions<
    //         make_tensorindex_list<TensorIndexA, TensorIndexB>>();

    constexpr std::array<bool, 2> rhs_is_spatial_spacetime_index = {
        ::tenex::detail::is_spatial_spacetime_index<
            tmpl::at_c<RhsTensorIndexTypeList, 0>,
            std::decay_t<decltype(TensorIndexA)>>(),
        ::tenex::detail::is_spatial_spacetime_index<
            tmpl::at_c<RhsTensorIndexTypeList, 1>,
            std::decay_t<decltype(TensorIndexB)>>()};

    constexpr auto lhs_spatial_spacetime_index_positions =
        ::tenex::detail::get_spatial_spacetime_index_positions<
            LhsTensorIndexTypeList,
            make_tensorindex_list<TensorIndexA, TensorIndexB>>();

    constexpr auto time_index_positions =
        ::tenex::detail::get_time_index_positions<
            make_tensorindex_list<TensorIndexA, TensorIndexB>>();

    constexpr auto rhs_index_types = rhs_structure::index_types();
    // constexpr auto lhs_index_types = lhs_structure::index_types();

    constexpr std::array<size_t, 2> tensorindex_values = {TensorIndexA.value,
                                                          TensorIndexB.value};

    // TensorMetafunctions::check_index_symmetry_v<Symm, Indices...>

    // L_ai = R_ai
    if constexpr (::tenex::detail::is_generic_spacetime_index_value(
                      tensorindex_values[0]) and
                  ::tenex::detail::is_generic_spatial_index_value(
                      tensorindex_values[1])) {
      if constexpr (ReturnLhsTensor) {
        // static_assert();
      } else {
      }
    }

    // L_ia = R_ia
    if constexpr (::tenex::detail::is_generic_spatial_index_value(
                      tensorindex_values[0]) and
                  ::tenex::detail::is_generic_spacetime_index_value(
                      tensorindex_values[1])) {
      if constexpr (ReturnLhsTensor) {
      } else {
      }
    }

    // L_ij = R_ij
    if constexpr (::tenex::detail::is_generic_spatial_index_value(
                      tensorindex_values[0]) and
                  ::tenex::detail::is_generic_spatial_index_value(
                      tensorindex_values[1])) {
      if constexpr (ReturnLhsTensor) {
      } else {
      }
    }

    // L_it = R_it
    if constexpr (::tenex::detail::is_generic_spatial_index_value(
                      tensorindex_values[0]) and
                  ::tenex::detail::is_time_index_value(tensorindex_values[1])) {
      if constexpr (ReturnLhsTensor) {
      } else {
      }
    }

    // L_ti = R_ti
    if constexpr (::tenex::detail::is_time_index_value(
                      tensorindex_values[0]) and
                  ::tenex::detail::is_generic_spatial_index_value(
                      tensorindex_values[1])) {
      if constexpr (ReturnLhsTensor) {
      } else {
      }
    }

    // L_tt  = R_tt
    if constexpr (::tenex::detail::is_time_index_value(
                      tensorindex_values[0]) and
                  ::tenex::detail::is_time_index_value(tensorindex_values[1])) {
      if constexpr (ReturnLhsTensor) {
      } else {
      }
    }
  } else {
    // TODO ?
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 2 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details This simply runs `test_evaluate_rank_2_impl` but for different
/// data types for the tensor components
///
/// \param ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not (which instead tests evaluation by
/// preallocating the result tensor and filling it)
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, auto& TensorIndexA, auto& TensorIndexB,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_2() {
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      ReturnLhsTensor, TensorIndexA, TensorIndexB, double, RhsSymmetry,
      RhsTensorIndexTypeList, RhsTensorIndexTypeList>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      ReturnLhsTensor, TensorIndexA, TensorIndexB, DataVector, RhsSymmetry,
      RhsTensorIndexTypeList, RhsTensorIndexTypeList>();
}
}  // namespace TestHelpers::tenex
