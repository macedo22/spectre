// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"

namespace TensorExpressions {

// here, LhsTensorIndexList and RhsTensorIndexList refer to lists of ti_a_t,
// ti_b_t, etc.
template <typename RhsTensorIndexList, typename LhsTensorIndexList,
          typename IntSequence>
struct RhsToLhsIndexMap;

template <typename RhsTensorIndexList, typename... LhsIndices, size_t... Ints>
struct RhsToLhsIndexMap<RhsTensorIndexList, tmpl::list<LhsIndices...>,
                        std::integer_sequence<size_t, Ints...>> {
  static constexpr size_t num_indices = sizeof...(LhsIndices);
  static constexpr std::make_integer_sequence<size_t, num_indices>
      running_int_seq{};
  static constexpr std::array<size_t, num_indices> lhs_tensorindex_values = {
      {LhsIndices::value...}};
  static constexpr std::array<size_t, num_indices> rhs_tensorindex_values = {
      {tmpl::at_c<RhsTensorIndexList, Ints>::value...}};
  static constexpr std::array<size_t, num_indices> rhs_to_lhs_map = {
      {array_index_of<size_t, num_indices>(lhs_tensorindex_values,
                                           rhs_tensorindex_values[Ints])...}};

  using type = std::integer_sequence<size_t, rhs_to_lhs_map[Ints]...>;
};

// here, LhsTensorIndexTypeList and RhsTensorIndexTypeList refer to lists of
// index types like SpatialIndex<3, UpLo::Lo, Frame::Grid>
template <typename RhsSymm, typename RhsToLhsMap>
struct LhsSymm;

template <typename RhsSymm, size_t... MapIndices>
struct LhsSymm<RhsSymm, std::integer_sequence<size_t, MapIndices...>> {
  using type = tmpl::list<tmpl::at_c<RhsSymm, MapIndices>...>;
};

// here, RhsTensorIndexTypeList refers to a list of
// index types like SpatialIndex<3, UpLo::Lo, Frame::Grid>
template <typename RhsTensorIndexTypeList, typename RhsToLhsMap>
struct LhsTensorIndexTypeList;

template <typename RhsTensorIndexTypeList, size_t... MapIndices>
struct LhsTensorIndexTypeList<RhsTensorIndexTypeList,
                              std::integer_sequence<size_t, MapIndices...>> {
  using type = tmpl::list<tmpl::at_c<RhsTensorIndexTypeList, MapIndices>...>;
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
auto /*Tensor<typename T::type, typename T::symmetry, typename T::index_list>*/
evaluate(const T& te) {
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

  // TODO: should there be an assert that you don't have an index repeated?
  //       whether you have repeat with the same or opposite valence, e.g.:
  //           evaluate<ti_a_t, ti_A_t> or evaluate<ti_a_t, ti_a_t>
  //       and other cases, perhaps with more indices even if not repeats, e.g.:
  //           evaluate<ti_a_t, ti_b_t, ti_A_t, ti_c_t>

  // Lists of TensorIndex<#>s for RHS and LHS
  // e.g. (ti_a_t, ti_b_t)
  using rhs_tensorindex_list = typename T::args_list;
  using lhs_tensorindex_list = tmpl::list<LhsIndices...>;

  using rhs_symmetry = typename T::symmetry;

  // List of RHS TensorIndexTypes
  // e.g. (SpatialIndex<3, UpLo::Lo, Frame::Grid>,
  //       SpatialIndex<2, UpLo::Up, Frame::Grid>)
  using rhs_tensorindextype_list = typename T::index_list;

  constexpr size_t num_indices = sizeof...(LhsIndices);

  // Running integer sequence from [0 ... num_indices)
  // used for indexing into template template parameters
  using running_int_seq = std::make_integer_sequence<size_t, num_indices>;

  // Mapping from RHS to LHS indices
  using rhs_to_lhs_index_map =
      typename RhsToLhsIndexMap<rhs_tensorindex_list, lhs_tensorindex_list,
                                running_int_seq>::type;

  // List of LHS TensorIndexTypes according to index reordering from
  // rhs_to_lhs_index_map e.g. if the LHS indices are the above RHS indices
  // reversed, then LHS ordering is:
  //      (SpatialIndex<2, UpLo::Up, Frame::Grid>,
  //       SpatialIndex<3, UpLo::Lo, Frame::Grid>)
  using lhs_tensorindextype_list =
      typename LhsTensorIndexTypeList<rhs_tensorindextype_list,
                                      rhs_to_lhs_index_map>::type;

  // LHS symmetry according to reordering from rhs_to_lhs_index_map
  using lhs_symmetry =
      typename LhsSymm<rhs_symmetry, rhs_to_lhs_index_map>::type;

  return Tensor<typename T::type, lhs_symmetry, lhs_tensorindextype_list>(
      te, tmpl::list<LhsIndices...>{});
}

}  // namespace TensorExpressions
