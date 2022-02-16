// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double`
struct NumberAsExpression
    : public TensorExpression<NumberAsExpression, double, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>> {
  using type = double;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  static constexpr auto num_tensor_indices = 0;

  static constexpr size_t num_ops_left_child = 0;
  static constexpr size_t num_ops_right_child = 0;
  static constexpr size_t num_ops_subtree = 0;

  static constexpr bool is_primary_end = true;
  static constexpr size_t num_ops_to_evaluate_primary_left_child = 0;
  static constexpr size_t num_ops_to_evaluate_primary_right_child = 0;
  static constexpr size_t num_ops_to_evaluate_primary_subtree = 0;
  static constexpr bool is_primary_start = false;

  static constexpr bool primary_child_subtree_contains_primary_start = false;
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start;

  NumberAsExpression(const double number) : number_(number) {}
  ~NumberAsExpression() override = default;

  // This expression does not represent a tensor, nor does it have any children,
  // so we should never be asking this expression to return a component of a
  // `Tensor` that appears in the overall RHS expression
  type get_used_for_size() const = delete;
  // This expression does not represent a tensor, nor does it have any children,
  // so we should never need to assert that the LHS `Tensor` is not equal to the
  // `double` stored by this expression
  template <typename LhsTensor>
  void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*>) const = delete;

  /// \brief Returns the number represented by the expression
  ///
  /// \details
  /// While a NumberAsExpression does not store a rank 0 Tensor, it does
  /// represent one. This is why the multi-index argument is always an array
  /// of size 0.
  ///
  /// \return the number represented by this expression
  SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE double get_primary(
      const ResultType& /*result_component*/,
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (is_primary_start) {
      result_component = get(multi_index);
    }
  }

 private:
  double number_;
};
}  // namespace TensorExpressions
