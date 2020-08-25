// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function TensorExpressions::evaluate(TensorExpression)

#pragma once

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Algorithm.hpp"

namespace TensorExpressions {
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

  using mapping = tmpl::transform<
      lhs_tensorindex_list,
      tmpl::bind<tmpl::index_of, tmpl::pin<rhs_tensorindex_list>,
                 tmpl::_1>>;
  using lhs_symmetry =
      tmpl::transform<mapping,
                      tmpl::bind<tmpl::at, tmpl::pin<rhs_symmetry>, tmpl::_1>>;
  using lhs_tensorindextype_list =
      tmpl::transform<mapping,
                      tmpl::bind<tmpl::at, tmpl::pin<rhs_tensorindextype_list>,
                                 tmpl::_1>>;

  return Tensor<typename T::type, lhs_symmetry, lhs_tensorindextype_list>(
      te, tmpl::list<LhsIndices...>{});
}
}  // namespace TensorExpressions
