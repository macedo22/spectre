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

  NumberAsExpression(const double number) : number_(number) {}
  ~NumberAsExpression() override = default;

  /// \brief Returns the number represented by the expression
  ///
  /// \return the number represented by this expression
  SPECTRE_ALWAYS_INLINE double get(
      const std::array<size_t, 0>& /*multi_index*/) const noexcept {
    return number_;
  }

 private:
  double number_;
};
}  // namespace TensorExpressions
