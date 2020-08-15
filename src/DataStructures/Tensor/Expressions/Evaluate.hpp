// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"

namespace TensorExpressions {
// LhsTensor computes and stores the reordered LHS symmetry and indices
//
// RhsTensorIndexList & LhsTensorIndexList are lists of TensorIndex<#>s
//   - e.g. (ti_a_t, ti_B_t) and (ti_B_t, ti_a_t)
// RhseTensorIndexTypeList is the list of RHS TensorIndexTypes
//   - e.g. (SpatialIndex<3, UpLo::Lo, Frame::Grid>,
//           SpatialIndex<2, UpLo::Up, Frame::Grid>)
// IntSequence is a sequence of ints from [0 ... number of indices)
template <typename RhsTensorIndexList, typename LhsTensorIndexList,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          typename IntSequence>
struct LhsTensor;

template <typename RhsTensorIndexList, typename... LhsTensorIndices,
          typename RhsSymmetry, typename RhsTensorIndexTypeList, size_t... Ints>
struct LhsTensor<RhsTensorIndexList, tmpl::list<LhsTensorIndices...>,
                 RhsSymmetry, RhsTensorIndexTypeList,
                 std::integer_sequence<size_t, Ints...>> {
  static constexpr size_t num_indices = sizeof...(LhsTensorIndices);
  static constexpr std::make_integer_sequence<size_t, num_indices>
      running_int_seq{};
  static constexpr std::array<size_t, num_indices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  static constexpr std::array<size_t, num_indices> rhs_tensorindex_values = {
      {tmpl::at_c<RhsTensorIndexList, Ints>::value...}};
  static constexpr std::array<size_t, num_indices> rhs_to_lhs_map = {
      {array_index_of<size_t, num_indices>(lhs_tensorindex_values,
                                           rhs_tensorindex_values[Ints])...}};

  using symmetry = tmpl::list<tmpl::at_c<RhsSymmetry, rhs_to_lhs_map[Ints]>...>;
  using tensorindextype_list =
      tmpl::list<tmpl::at_c<RhsTensorIndexTypeList, rhs_to_lhs_map[Ints]>...>;
};

/*!
 * \ingroup TensorExpressionsGroup
 * \brief Evaluate a Tensor Expression with LHS indices set in the template
 * parameters
 *
 * @tparam LhsIndices the indices on the left hand side of the tensor expression
 * @return Tensor<typename T::type, typename T::symmetry, typename
 * T::index_list>
 */
template <typename... LhsIndices, typename T,
          Requires<std::is_base_of<Expression, T>::value> = nullptr>
auto evaluate(const T& te) {
  static_assert(
      sizeof...(LhsIndices) == tmpl::size<typename T::args_list>::value,
      "Must have the same number of indices on the LHS and RHS of a tensor "
      "equation.");
  using rhs = tmpl::transform<tmpl::remove_duplicates<typename T::args_list>,
                              std::decay<tmpl::_1>>;
  static_assert(
      tmpl::equal_members<tmpl::list<std::decay_t<LhsIndices>...>, rhs>::value,
      "All indices on the LHS of a Tensor Expression (that is, those specified "
      "in evaluate<Indices::...>) must be present on the RHS of the expression "
      "as well.");

  // (e.g. ti_a_t, ti_B_t)
  using rhs_tensorindex_list = typename T::args_list;
  // (e.g. ti_B_t, ti_a_t)
  using lhs_tensorindex_list = tmpl::list<LhsIndices...>;
  using rhs_symmetry = typename T::symmetry;
  // e.g. (SpatialIndex<3, UpLo::Lo, Frame::Grid>,
  //       SpatialIndex<2, UpLo::Up, Frame::Grid>)
  using rhs_tensorindextype_list = typename T::index_list;

  constexpr size_t num_indices = sizeof...(LhsIndices);

  // Running integer sequence from [0 ... num_indices)
  // used for indexing into template template parameters with tmpl::at_c
  // in LhsTensor
  using running_int_seq = std::make_integer_sequence<size_t, num_indices>;

  using lhs_tensor =
      LhsTensor<rhs_tensorindex_list, lhs_tensorindex_list, rhs_symmetry,
                rhs_tensorindextype_list, running_int_seq>;

  return Tensor<typename T::type, typename lhs_tensor::symmetry,
                typename lhs_tensor::tensorindextype_list>(
      te, tmpl::list<LhsIndices...>{});
}

}  // namespace TensorExpressions
