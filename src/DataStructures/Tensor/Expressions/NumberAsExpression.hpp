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
/// \brief Marks a class as being a `TensorExpressions::NumberAsExpression`
///
/// \details
/// The empty base class provides a simple means for checking if a type is a
/// `TensorExpressions::NumberAsExpression`.
struct MarkAsNumberAsExpression {};

/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double`
struct NumberAsExpression
    : public TensorExpression<NumberAsExpression, double, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>>,
      MarkAsNumberAsExpression {
  using type = double;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  static constexpr auto num_tensor_indices = 0;
  static constexpr size_t num_ops_left = 0;
  static constexpr size_t num_ops_right = 0;
  static constexpr size_t num_ops_subtree = 0;
  static constexpr size_t num_addsub_ops_subtree = 0;

  NumberAsExpression(const double number) : number_(number) {}
  ~NumberAsExpression() override = default;

  /// \brief Returns the number represented by the expression
  ///
  /// \details
  /// While a NumberAsExpression does not store a rank 0 Tensor, it does
  /// represent one. This is why the multi-index argument is always an array of
  /// size 0.
  ///
  /// \return the number represented by this expression
  SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

  // TODO : remove? don't need at leaves?
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit(
      const ResultType& /*result_component*/,
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {}

 private:
  double number_;
};
}  // namespace TensorExpressions
