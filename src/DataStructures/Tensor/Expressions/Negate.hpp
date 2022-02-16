// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the tensor expression representing negation

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the negation of a tensor
/// expression
///
/// \tparam T the type of tensor expression being negated
template <typename T>
struct Negate
    : public TensorExpression<Negate<T>, typename T::type, typename T::symmetry,
                              typename T::index_list, typename T::args_list> {
  using type = typename T::type;
  using symmetry = typename T::symmetry;
  using index_list = typename T::index_list;
  using args_list = typename T::args_list;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;

  static constexpr size_t num_ops_left_child = T::num_ops_subtree;
  static constexpr size_t num_ops_right_child = 0;
  static constexpr size_t num_ops_subtree = num_ops_left_child + 1;

  static constexpr bool is_primary_end = T::is_primary_start;
  static constexpr size_t num_ops_to_evaluate_primary_left_child =
      is_primary_end ? 0 : T::num_ops_to_evaluate_primary_subtree;
  static constexpr size_t num_ops_to_evaluate_primary_right_child =
      num_ops_right_child;
  static constexpr size_t num_ops_to_evaluate_primary_subtree =
      num_ops_to_evaluate_primary_left_child +
      num_ops_to_evaluate_primary_right_child + 1;
  static constexpr bool is_primary_start =
      num_ops_to_evaluate_primary_subtree >=
      detail::max_num_ops_in_sub_expression<type>;

  static constexpr bool primary_child_subtree_contains_primary_start =
      T::primary_subtree_contains_primary_start;
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start or primary_child_subtree_contains_primary_start;

  Negate(T t) : t_(std::move(t)) {}
  ~Negate() override = default;

  /// \brief Assert that the LHS tensor of the equation does not also appear in
  /// this expression's subtree
  template <typename LhsTensor>
  SPECTRE_ALWAYS_INLINE void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*> lhs_tensor) const {
    if constexpr (not std::is_base_of_v<NumberAsExpression, T>) {
      t_.assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
    }
  }

  SPECTRE_ALWAYS_INLINE auto get_used_for_size() const {
    return t_.get_used_for_size();
  }

  /// \brief Return the value of the component of the negated tensor expression
  /// at a given multi-index
  ///
  /// \param multi_index the multi-index of the component to retrieve from the
  /// negated tensor expression
  /// \return the value of the component at `multi_index` in the negated tensor
  /// expression
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return -t_.get(multi_index);
  }

  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const type& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (is_primary_end) {
      (void)multi_index;
      return -result_component;
    } else {
      return -t_.get_primary(result_component, multi_index);
    }
  }

  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      type& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (primary_child_subtree_contains_primary_start) {
      t_.evaluate_primary_subtree(result_component, multi_index);
    }
    if constexpr (is_primary_start) {
      if constexpr (primary_child_subtree_contains_primary_start) {
        result_component = get_primary(result_component, multi_index);
      } else {
        result_component = get(multi_index);
      }
    }
  }

 private:
  T t_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the negation of a tensor
/// expression
///
/// \param t the tensor expression
/// \return the tensor expression representing the negation of `t`
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return TensorExpressions::Negate<T>(~t);
}
