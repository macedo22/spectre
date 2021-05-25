// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include <array>
#include <iostream>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

namespace TensorExpressions {

namespace detail {
template <size_t NumIndices>
constexpr bool contains_indices_to_contract(
    const std::array<size_t, NumIndices>& tensorindices) noexcept {
  if constexpr (NumIndices < 2) {
    return false;
  } else {
    for (size_t i = 0; i < NumIndices - 1; i++) {
      for (size_t j = i + 1; j < NumIndices; j++) {
        if (gsl::at(tensorindices, i) ==
            get_tensorindex_value_with_opposite_valence(
                gsl::at(tensorindices, j))) {
          return true;
        }
      }
    }
    return false;
  }
}

template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices>
compute_index_transformation(
    const std::array<size_t, NumIndices>& lhs_tensorindices,
    const std::array<size_t, NumIndices>& rhs_tensorindices) noexcept {
  // constexpr std::array<size_t, NumIndices> rhs_tensorindices = {
  //     {Args::value...}};
  std::array<size_t, NumIndices> index_transformation{};
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(index_transformation, i) = static_cast<size_t>(std::distance(
        lhs_tensorindices.begin(),
        alg::find(lhs_tensorindices, gsl::at(rhs_tensorindices, i))));
  }
  return index_transformation;
}

template <size_t NumIndices>
SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, NumIndices>
compute_rhs_multi_index(
    const std::array<size_t, NumIndices>& lhs_multi_index,
    const std::array<size_t, NumIndices>& index_transformation) noexcept {
  std::array<size_t, NumIndices> rhs_multi_index{};
  for (size_t i = 0; i < NumIndices; i++) {
    gsl::at(rhs_multi_index, i) =
        gsl::at(lhs_multi_index, gsl::at(index_transformation, i));
  }
  return rhs_multi_index;
}

// template <typename LhsTensorIndexTypeList>
// struct PositionsOfLhsSpatialSpacetimeIndices;

// template <typename... LhsTensorIndexTypes>
// struct
// PositionsOfLhsSpatialSpacetimeIndices<index_list<LhsTensorIndexTypes...>>{
// constexpr auto apply(
//     const std::array<size_t, sizeof...(LhsTensorIndexTypes)>& tensorindices)
//     noexcept {
//   constexpr auto indices =
//       {{LhsTensorIndexTypes::index_type == IndexType::Spacetime}...};
//   // constexpr std::array<bool, sizeof...(LhsTensorIndexTypes)>
//   lhs_tensorindextypes_are_spacetime =
//   //     {{LhsTensorIndexTypes::index_type == IndexType::Spacetime}...};
//   // size_t num_spatial_spacetime_indices = 0;
//   // for (int i = 0; i < sizeof...(LhsTensorIndexTypes); i++) {
//   //   num_spatial_spacetime_indices +=
//   // }
// }
// };

// // TODO: try to make this an alias instead of a struct
// template <typename State, typename Element, typename Iteration,
//           typename TensorIndexList>
// struct spatial_spacetime_index_positions_impl {
//   using type = typename std::conditional<
//       Element::index_type == IndexType::Spacetime and not
//       tmpl::at<TensorIndexList, Iteration>::is_spacetime,
//       tmpl::push_back<State, Iteration>, State>::type;
// };

// template <typename TensorIndexTypeList, typename TensorIndexList>
// using spatial_spacetime_index_positions = tmpl::enumerated_fold<
//     TensorIndexTypeList, tmpl::list<>,
//     spatial_spacetime_index_positions_impl<tmpl::_state, tmpl::_element,
//     tmpl::_3, tmpl::pin<TensorIndexList>>>;

}  // namespace detail

// template <typename TensorIndexTypeList, typename TensorIndexList>
// constexpr auto get_spatial_spacetime_index_positions() noexcept{
//   return
//   make_array_from_list<
//       detail::spatial_spacetime_index_positions<TensorIndexTypeList,
//   TensorIndexList>>();
// }

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`RhsTE::args_list`) and the desired left hand side (LHS) tensor's index
 * ordering (`LhsTensorIndices`) to fill the provided LHS Tensor with that LHS
 * index ordering. This can carry out the evaluation of a RHS tensor expression
 * to a LHS tensor with the same index ordering, such as \f$L_{ab} = R_{ab}\f$,
 * or different ordering, such as \f$L_{ba} = R_{ab}\f$.
 *
 * The symmetry of the provided LHS Tensor need not match the symmetry
 * determined from evaluating the RHS TensorExpression according to its order of
 * operations. This allows one to specify LHS symmetries (via `lhs_tensor`) that
 * may not be preserved by the RHS expression's order of operations, which
 * depends on how the expression is written and implemented.
 *
 * ### Example usage
 * Given two rank 2 Tensors `R` and `S` with index order (a, b), add them
 * together and fill the provided resultant LHS Tensor `L` with index order
 * (b, a):
 * \code{.cpp}
 * TensorExpressions::evaluate<ti_b, ti_a>(
 *     make_not_null(&L), R(ti_a, ti_b) + S(ti_a, ti_b));
 * \endcode
 *
 * This represents evaluating: \f$L_{ba} = R_{ab} + S_{ab}\f$
 *
 * Note: `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the TensorIndexs of the Tensor on the LHS of the
 * tensor expression, e.g. `ti_a`, `ti_b`, `ti_c`
 * @param lhs_tensor pointer to the resultant LHS Tensor to fill
 * @param rhs_tensorexpression the RHS TensorExpression to be evaluated
 */
// template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
//           typename LhsIndexList, typename RhsTE,
//           Requires<std::is_base_of_v<Expression, RhsTE>> = nullptr>
template <auto&... LhsTensorIndices, typename X, typename LhsSymmetry,
          typename LhsIndexList, typename Derived, typename RhsSymmetry,
          typename RhsIndexList, typename... RhsTensorIndices>
void evaluate(
    const gsl::not_null<Tensor<X, LhsSymmetry, LhsIndexList>*> lhs_tensor,
    const TensorExpression<Derived, X, RhsSymmetry, RhsIndexList,
                           tmpl::list<RhsTensorIndices...>>&
        rhs_tensorexpression) {
  // std::cout << "\n=== EVALUATE ===" << std::endl;
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list =
      tmpl::list<RhsTensorIndices...>;  // typename RhsTE::args_list;
  static_assert(
      tmpl::equal_members<lhs_tensorindex_list, rhs_tensorindex_list>::value,
      "The generic indices on the LHS of a tensor equation (that is, the "
      "template parameters specified in evaluate<...>) must match the generic "
      "indices of the RHS TensorExpression. This error occurs as a result of a "
      "call like evaluate<ti_a, ti_b>(R(ti_A, ti_b) * S(ti_a, ti_c)), where "
      "the generic indices of the evaluated RHS expression are ti_b and ti_c, "
      "but the generic indices provided for the LHS are ti_a and ti_b.");
  static_assert(
      tmpl::is_set<std::decay_t<decltype(LhsTensorIndices)>...>::value,
      "Cannot evaluate a tensor expression to a LHS tensor with a repeated "
      "generic index, e.g. evaluate<ti_a, ti_a>.");
  static_assert(
      not detail::contains_indices_to_contract<sizeof...(LhsTensorIndices)>(
          {{std::decay_t<decltype(LhsTensorIndices)>::value...}}),
      "Cannot evaluate a tensor expression to a LHS tensor with generic "
      "indices that would be contracted, e.g. evaluate<ti_A, ti_a>.");
  // using rhs_symmetry = RhsSymmetry;//typename RhsTE::symmetry;
  // using rhs_tensorindextype_list = RhsIndexList;//typename RhsTE::index_list;

  // Stores (potentially reordered) symmetry and indices expected for the LHS
  // tensor, with index order specified by LhsTensorIndices
  // using lhs_tensor_symm_and_indices =
  //     LhsTensorSymmAndIndices<rhs_tensorindex_list, lhs_tensorindex_list,
  //                             rhs_symmetry, rhs_tensorindextype_list>;
  // Instead of simply checking that the LHS Tensor type is correct, individual
  // checks for the data type and index list are carried out because it provides
  // more fine-grained feedback and because the provided LHS symmetry may differ
  // from the symmetry determined by the order of operations in the RHS
  // expression.
  // static_assert(
  //     std::is_same_v<X, typename RhsTE::type>,
  //     "The data type stored by the LHS tensor does not match the data type "
  //     "stored by the RHS expression.");
  // static_assert(
  //     std::is_same_v<
  //         LhsIndexList,
  //         typename lhs_tensor_symm_and_indices::tensorindextype_list>,
  //     "The index list of the LHS tensor does not match the index list of the
  //     " "evaluated RHS expression.");

  using lhs_tensor_type = typename std::decay_t<decltype(*lhs_tensor)>;

  constexpr auto lhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<LhsIndexList,
                                            lhs_tensorindex_list>();

  // std::cout << "lhs_spatial_spacetime_index_positions : " <<
  // lhs_spatial_spacetime_index_positions << std::endl;

  constexpr auto rhs_spatial_spacetime_index_positions =
      get_spatial_spacetime_index_positions<RhsIndexList,
                                            rhs_tensorindex_list>();

  // std::cout << "rhs_spatial_spacetime_index_positions : " <<
  // rhs_spatial_spacetime_index_positions << std::endl;

  for (size_t i = 0; i < lhs_tensor_type::size(); i++) {
    if constexpr (lhs_spatial_spacetime_index_positions.size() == 0 and
                  rhs_spatial_spacetime_index_positions.size() == 0) {
      // std::cout << "--NEITHER has spatial spacetime indices --" << std::endl;
      (*lhs_tensor)[i] =
          (~rhs_tensorexpression)
              .template get<std::decay_t<decltype(LhsTensorIndices)>...>(
                  lhs_tensor_type::structure::get_canonical_tensor_index(i));
    } else if constexpr (lhs_spatial_spacetime_index_positions.size() != 0 and
                         rhs_spatial_spacetime_index_positions.size() != 0) {
      // std::cout << "--BOTH have spatial spacetime indices --" << std::endl;
      auto lhs_multi_index =
          lhs_tensor_type::structure::get_canonical_tensor_index(i);
      if (alg::none_of(lhs_spatial_spacetime_index_positions,
                       [lhs_multi_index](size_t j) {
                         return lhs_multi_index[j] == 0;
                       })) {
        // std::cout << "lhs_multi_index before replacing: " << lhs_multi_index
        // << std::endl;
        for (size_t j = 0; j < lhs_spatial_spacetime_index_positions.size();
             j++) {
          lhs_multi_index[lhs_spatial_spacetime_index_positions[j]] -= 1;
        }
        // std::cout << "lhs_multi_index after replacing: " << lhs_multi_index
        // << std::endl;
        auto rhs_multi_index = detail::compute_rhs_multi_index(
            lhs_multi_index,
            detail::compute_index_transformation<sizeof...(RhsTensorIndices)>(
                {{std::decay_t<decltype(LhsTensorIndices)>::value...}},
                {{RhsTensorIndices::value...}}));
        for (size_t j = 0; j < rhs_spatial_spacetime_index_positions.size();
             j++) {
          rhs_multi_index[rhs_spatial_spacetime_index_positions[j]] += 1;
        }

        (*lhs_tensor)[i] =
            (~rhs_tensorexpression)
                .template get<RhsTensorIndices...>(rhs_multi_index);
      }
    } else if constexpr (rhs_spatial_spacetime_index_positions.size() != 0) {
      // std::cout << "--RHS has spatial spacetime indices --" << std::endl;
      auto rhs_multi_index = detail::compute_rhs_multi_index(
          lhs_tensor_type::structure::get_canonical_tensor_index(i),
          detail::compute_index_transformation<sizeof...(RhsTensorIndices)>(
              {{std::decay_t<decltype(LhsTensorIndices)>::value...}},
              {{RhsTensorIndices::value...}}));
      for (size_t j = 0; j < rhs_spatial_spacetime_index_positions.size();
           j++) {
        rhs_multi_index[rhs_spatial_spacetime_index_positions[j]] += 1;
      }

      (*lhs_tensor)[i] =
          (~rhs_tensorexpression)
              .template get<RhsTensorIndices...>(rhs_multi_index);

    } else {  // if constexpr (lhs_spatial_spacetime_index_positions.size() !=
              // 0) {
      // std::cout << "--LHS has spatial spacetime indices --" << std::endl;
      auto lhs_multi_index =
          lhs_tensor_type::structure::get_canonical_tensor_index(i);
      if (alg::none_of(lhs_spatial_spacetime_index_positions,
                       [lhs_multi_index](size_t j) {
                         return lhs_multi_index[j] == 0;
                       })) {
        // std::cout << "lhs_multi_index before replacing: " << lhs_multi_index
        // << std::endl;
        for (size_t j = 0; j < lhs_spatial_spacetime_index_positions.size();
             j++) {
          lhs_multi_index[lhs_spatial_spacetime_index_positions[j]] -= 1;
        }
        // std::cout << "lhs_multi_index after replacing: " << lhs_multi_index
        // << std::endl;
        auto rhs_multi_index = detail::compute_rhs_multi_index(
            lhs_multi_index,
            detail::compute_index_transformation<sizeof...(RhsTensorIndices)>(
                {{std::decay_t<decltype(LhsTensorIndices)>::value...}},
                {{RhsTensorIndices::value...}}));
        for (size_t j = 0; j < rhs_spatial_spacetime_index_positions.size();
             j++) {
          rhs_multi_index[rhs_spatial_spacetime_index_positions[j]] += 1;
        }

        (*lhs_tensor)[i] =
            (~rhs_tensorexpression)
                .template get<RhsTensorIndices...>(rhs_multi_index);
      }
    }
  }
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`RhsTE::args_list`) and the desired left hand side (LHS) tensor's index
 * ordering (`LhsTensorIndices`) to construct a LHS Tensor with that LHS index
 * ordering. This can carry out the evaluation of a RHS tensor expression to a
 * LHS tensor with the same index ordering, such as \f$L_{ab} = R_{ab}\f$, or
 * different ordering, such as \f$L_{ba} = R_{ab}\f$.
 *
 * The symmetry of the returned LHS Tensor depends on the order of operations in
 * the RHS TensorExpression, i.e. how the expression is written. If you would
 * like to specify the symmetry of the LHS Tensor instead of it being determined
 * by the order of operations in the RHS expression, please use the other
 * `evaluate` overload that takes an empty LHS Tensor as its first argument.
 *
 * ### Example usage
 * Given two rank 2 Tensors `R` and `S` with index order (a, b), add them
 * together and generate the resultant LHS Tensor `L` with index order (b, a):
 * \code{.cpp}
 * auto L = TensorExpressions::evaluate<ti_b, ti_a>(
 *     R(ti_a, ti_b) + S(ti_a, ti_b));
 * \endcode
 * \metareturns Tensor
 *
 * This represents evaluating: \f$L_{ba} = R_{ab} + S_{ab}\f$
 *
 * Note: `LhsTensorIndices` must be passed by reference because non-type
 * template parameters cannot be class types until C++20.
 *
 * @tparam LhsTensorIndices the TensorIndexs of the Tensor on the LHS of the
 * tensor expression, e.g. `ti_a`, `ti_b`, `ti_c`
 * @param rhs_tensorexpression the RHS TensorExpression to be evaluated
 * @return the resultant LHS Tensor with index order specified by
 * LhsTensorIndices
 */
template <auto&... LhsTensorIndices, typename RhsTE,
          Requires<std::is_base_of_v<Expression, RhsTE>> = nullptr>
auto evaluate(const RhsTE& rhs_tensorexpression) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list = typename RhsTE::args_list;
  using rhs_symmetry = typename RhsTE::symmetry;
  using rhs_tensorindextype_list = typename RhsTE::index_list;

  // Stores (potentially reordered) symmetry and indices needed for constructing
  // the LHS tensor, with index order specified by LhsTensorIndices
  using lhs_tensor_symm_and_indices =
      LhsTensorSymmAndIndices<rhs_tensorindex_list, lhs_tensorindex_list,
                              rhs_symmetry, rhs_tensorindextype_list>;

  Tensor<typename RhsTE::type, typename lhs_tensor_symm_and_indices::symmetry,
         typename lhs_tensor_symm_and_indices::tensorindextype_list>
      lhs_tensor{};
  evaluate<LhsTensorIndices...>(make_not_null(&lhs_tensor),
                                rhs_tensorexpression);
  return lhs_tensor;
}
}  // namespace TensorExpressions
