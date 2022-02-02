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

  static constexpr bool is_main_end = true;
  static constexpr bool is_main_beg = true;

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

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE double get_main(
      const ResultType& /*result_component*/,
      const std::array<size_t, num_tensor_indices>& /*multi_index*/) const {
    return number_;
  }

  // TODO : remove? don't need at leaves?
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit_main(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (is_main_beg) {
      result_component = get_main(result_component, multi_index);
    }
  }

 private:
  double number_;
};
}  // namespace TensorExpressions
