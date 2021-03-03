// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/LhsTensorSymmAndIndices.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
}  // namespace detail

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`T::args_list`) and the desired left hand side (LHS) tensor's index ordering
 * (`LhsTensorIndices`) to construct a LHS Tensor with that LHS index ordering.
 * This can carry out the evaluation of a RHS tensor expression to a LHS tensor
 * with the same index ordering, such as \f$L_{ab} = R_{ab}\f$, or different
 * ordering, such as \f$L_{ba} = R_{ab}\f$.
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
 * @param rhs_te the RHS TensorExpression to be evaluated
 * @return the resultant LHS Tensor with index order specified by
 * LhsTensorIndices
 */
template <auto&... LhsTensorIndices, typename TE,
          Requires<std::is_base_of<Expression, TE>::value> = nullptr>
auto evaluate(const TE& rhs_te) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list = typename TE::args_list;
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
  using rhs_symmetry = typename TE::symmetry;
  using rhs_tensorindextype_list = typename TE::index_list;

  // Stores (potentially reordered) symmetry and indices needed for constructing
  // the LHS tensor, with index order specified by LhsTensorIndices
  using lhs_tensor_symm_and_indices =
      LhsTensorSymmAndIndices<rhs_tensorindex_list, lhs_tensorindex_list,
                              rhs_symmetry, rhs_tensorindextype_list>;

  // Construct and return LHS tensor
  return Tensor<typename TE::type,
                typename lhs_tensor_symm_and_indices::symmetry,
                typename lhs_tensor_symm_and_indices::tensorindextype_list>(
      rhs_te, lhs_tensorindex_list{});
}

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a RHS tensor expression to a tensor with the LHS index order
 * set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`T::args_list`) and the desired left hand side (LHS) tensor's index ordering
 * (`LhsTensorIndices`) to fill the provided LHS Tensor with that LHS index
 * ordering. This can carry out the evaluation of a RHS tensor expression to a
 * LHS tensor with the same index ordering, such as \f$L_{ab} = R_{ab}\f$, or
 * different ordering, such as \f$L_{ba} = R_{ab}\f$.
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
 * @param rhs_te the RHS TensorExpression to be evaluated
 */
template <auto&... LhsTensorIndices, typename T, typename TE,
          Requires<Tensor_detail::is_tensor<T>::value> = nullptr,
          Requires<std::is_base_of<Expression, TE>::value> = nullptr>
void evaluate(const gsl::not_null<T*> lhs_tensor, const TE& rhs_te) {
  using lhs_tensorindex_list =
      tmpl::list<std::decay_t<decltype(LhsTensorIndices)>...>;
  using rhs_tensorindex_list = typename TE::args_list;
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
  using rhs_symmetry = typename TE::symmetry;
  using rhs_tensorindextype_list = typename TE::index_list;

  // Stores (potentially reordered) symmetry and indices expected for the LHS
  // tensor, with index order specified by LhsTensorIndices
  using lhs_tensor_symm_and_indices =
      LhsTensorSymmAndIndices<rhs_tensorindex_list, lhs_tensorindex_list,
                              rhs_symmetry, rhs_tensorindextype_list>;
  // Instead of simply checking that `T` is the correct LHS Tensor type,
  // individual checks for the data type, symmetry, and index_list are carried
  // out to provide more fine-grained feedback
  static_assert(
      std::is_same_v<typename T::type, typename TE::type>,
      "The data type stored by the LHS tensor does not match the data type "
      "stored by the RHS expression.");
  static_assert(
      std::is_same_v<typename T::symmetry,
                     typename lhs_tensor_symm_and_indices::symmetry>,
      "The symmetry of the LHS tensor does not match the symmetry of the "
      "evaluated RHS expression.");
  static_assert(
      std::is_same_v<
          typename T::index_list,
          typename lhs_tensor_symm_and_indices::tensorindextype_list>,
      "The index list of the LHS tensor does not match the index list of the "
      "evaluated RHS expression.");

  // Fill LHS tensor
  for (size_t i = 0; i < T::size(); i++) {
    (*lhs_tensor)[i] =
        rhs_te.template get<typename T::structure,
                            std::decay_t<decltype(LhsTensorIndices)>...>(i);
  }
}
}  // namespace TensorExpressions
