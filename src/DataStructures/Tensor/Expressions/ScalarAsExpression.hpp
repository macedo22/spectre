// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a scalar
struct ScalarAsExpression
    : public TensorExpression<ScalarAsExpression, double, tmpl::list<>,
                              tmpl::list<>, tmpl::list<>> {
  using type = double;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;

  /// \brief Create an expression from a scalar
  ScalarAsExpression(double s) : s_(s) {}
  ~ScalarAsExpression() override = default;

  /// \brief Returns the value represented by the expression
  ///
  /// \details
  /// While a ScalarAsExpression does not store a rank 0 Tensor, it does
  /// represent one. This is why, unlike other derived TensorExpression types,
  /// there is no second variadic template parameter for the generic indices.
  /// In addition, this is why this template is only instantiated for the case
  /// where `Structure` is equal to the Structure of such a Tensor.
  ///
  /// \tparam Structure the Structure of the rank 0 Tensor represented by this
  /// expression
  /// \return the value of the scalar represented by this expression
  template <typename Structure>
  SPECTRE_ALWAYS_INLINE double get(
      const size_t storage_index) const noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE double get<structure>(
      const size_t /*storage_index*/) const noexcept {
    return s_;
  }

 private:
  /// The scalar value being represented as an expression
  double s_;
};
}  // namespace TensorExpressions
