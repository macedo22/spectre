// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Requires.hpp"

#include <iostream>

namespace TensorExpressions {
/*!
 * \ingroup TensorExpressionsGroup
 * \brief Determines and stores a LHS tensor's symmetry and index list from a
 * RHS tensor expression and desired LHS index order
 *
 * \details Given the generic index order of a RHS TensorExpression and the
 * generic index order of the desired LHS Tensor, this creates a mapping between
 * the two that is used to determine the (potentially reordered) ordering of the
 * elements of the desired LHS Tensor`s ::Symmetry and typelist of
 * \ref SpacetimeIndex "TensorIndexType"s.
 *
 * @tparam RhsTensorIndexList the typelist of TensorIndex of the RHS
 * TensorExpression, e.g. `ti_a_t`, `ti_b_t`, `ti_c_t`
 * @tparam LhsTensorIndexList the typelist of TensorIndexs of the desired LHS
 * tensor, e.g. `ti_b_t`, `ti_c_t`, `ti_a_t`
 * @tparam RhsSymmetry the ::Symmetry of the RHS indices
 * @tparam RhsTensorIndexTypeList the RHS TensorExpression's typelist of
 * \ref SpacetimeIndex "TensorIndexType"s
 */
template <typename RhsTensorIndexList, typename LhsTensorIndexList,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t RhsNumIndices = tmpl::size<RhsTensorIndexList>::value,
          size_t LhsNumIndices = tmpl::size<LhsTensorIndexList>::value,
          typename RhsIndexSequence = std::make_index_sequence<RhsNumIndices>,
          typename LhsIndexSequence = std::make_index_sequence<LhsNumIndices>>
struct LhsTensor;

template <typename RhsTensorIndexList, typename... LhsTensorIndices,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t RhsNumIndices, size_t LhsNumIndices, size_t... RhsInts,
          size_t... LhsInts>
struct LhsTensor<RhsTensorIndexList, tmpl::list<LhsTensorIndices...>,
                 RhsSymmetry, RhsTensorIndexTypeList, RhsNumIndices,
                 LhsNumIndices, std::index_sequence<RhsInts...>,
                 std::index_sequence<LhsInts...>> {
  static constexpr std::array<size_t, LhsNumIndices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  static constexpr std::array<size_t, RhsNumIndices> rhs_tensorindex_values = {
      {tmpl::at_c<RhsTensorIndexList, RhsInts>::value...}};
  static constexpr std::array<size_t, LhsNumIndices> lhs_to_rhs_map = {
      {std::distance(rhs_tensorindex_values.begin(),
                     alg::find(rhs_tensorindex_values,
                               lhs_tensorindex_values[LhsInts]))...}};

  // Desired LHS Tensor's Symmetry and typelist of TensorIndexTypes
  using symmetry =
      Symmetry<tmpl::at_c<RhsSymmetry, lhs_to_rhs_map[LhsInts]>::value...>;
  using tensorindextype_list = tmpl::list<
      tmpl::at_c<RhsTensorIndexTypeList, lhs_to_rhs_map[LhsInts]>...>;
};

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a right hand side tensor expression to a tensor with the LHS
 * index order set in the template parameters
 *
 * \details Uses the right hand side (RHS) TensorExpression's index ordering
 * (`T::args_list`) and the desired left hand side (LHS) tensor's index ordering
 * (`LhsTensorIndices`) to construct a LHS Tensor with that LHS index ordering.
 * This can carry out the evaluation of a RHS tensor expression to a LHS tensor
 * with the same index ordering, such as \f$L_{ab} = R_{ab}\f$, or different
 * ordering, such as \f$L_{ba} = R_{ab}\f$.
 *
 * ### Usage
 * Given two rank 2 Tensors `R` and `S` with index order (a, b), add them
 * together and generate the resultant LHS Tensor `L` with index order (b, a):
 * \code{.cpp}
 * auto L = TensorExpressions::evaluate<ti_b_t, ti_a_t>(
 *     R(ti_a, ti_b) + S(ti_a, ti_b));
 * \endcode
 * \metareturns Tensor
 *
 * This represents evaluating: \f$L_{ba} = \R_{ab} + S_{ab}\f$
 *
 * @tparam LhsTensorIndices the TensorIndex of the Tensor on the LHS of the
 * tensor expression, e.g. `ti_a_t`, `ti_b_t`, `ti_c_t`
 * @tparam T the type of the RHS TensorExpression
 * @param rhs_te the RHS TensorExpression to be evaluated
 * @return the LHS Tensor with index order specified by LhsTensorIndices
 */
template <typename... LhsTensorIndices, typename T,
          Requires<std::is_base_of<Expression, T>::value> = nullptr>
auto evaluate(const T& rhs_te) {
  static_assert(
      sizeof...(LhsTensorIndices) == tmpl::size<typename T::args_list>::value,
      "Must have the same number of indices on the LHS and RHS of a tensor "
      "equation.");
  using rhs = tmpl::transform<tmpl::remove_duplicates<typename T::args_list>,
                              std::decay<tmpl::_1>>;
  static_assert(
      tmpl::equal_members<tmpl::list<std::decay_t<LhsTensorIndices>...>,
                          rhs>::value,
      "All indices on the LHS of a Tensor Expression (that is, those specified "
      "in evaluate<Indices::...>) must be present on the RHS of the expression "
      "as well.");

  // e.g. (ti_a_t, ti_B_t)
  using rhs_tensorindex_list = typename T::args_list;
  // e.g. (ti_B_t, ti_a_t)
  using lhs_tensorindex_list = tmpl::list<LhsTensorIndices...>;
  using rhs_symmetry = typename T::symmetry;
  // e.g. (SpatialIndex<3, UpLo::Lo, Frame::Grid>,
  //       SpatialIndex<2, UpLo::Up, Frame::Grid>)
  using rhs_tensorindextype_list = typename T::index_list;

  // Stores (potentially reordered) symmetry and indices needed for constructing
  // the LHS tensor with index order specified by LhsTensorIndices
  using lhs_tensor = LhsTensor<rhs_tensorindex_list, lhs_tensorindex_list,
                               rhs_symmetry, rhs_tensorindextype_list>;

  /*std::cout << "rhs valence of first index: " <<
  tmpl::at_c<rhs_tensorindextype_list, 0>::ul << std::endl; std::cout << "rhs
  valence of second index: " << tmpl::at_c<rhs_tensorindextype_list, 1>::ul <<
  std::endl; std::cout << "rhs TensorIndex of first index: " <<
  tmpl::at_c<rhs_tensorindex_list, 0>::value << std::endl; // <--  is j but
  should be k std::cout << "rhs TensorIndex of second index: " <<
  tmpl::at_c<rhs_tensorindex_list, 1>::value << std::endl;// <-- is k but should
  be j std::cout << "lhs_tensor valence of first index: " << tmpl::at_c<typename
  lhs_tensor::tensorindextype_list, 0>::ul << std::endl; std::cout <<
  "lhs_tensor valence of second index: " << tmpl::at_c<typename
  lhs_tensor::tensorindextype_list, 1>::ul << std::endl;*/

  // Construct and return LHS tensor
  return Tensor<typename T::type, typename lhs_tensor::symmetry,
                typename lhs_tensor::tensorindextype_list>(
      rhs_te, tmpl::list<LhsTensorIndices...>{});
}
}  // namespace TensorExpressions
