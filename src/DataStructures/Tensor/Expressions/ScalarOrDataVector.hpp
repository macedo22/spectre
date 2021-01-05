// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent scalars and DataVectors

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representating a rank 0 tensor
///
/// \tparam DataType the type being represented, a double or DataVector
template <typename DataType>
struct ScalarOrDataVector
    : public TensorExpression<ScalarOrDataVector<DataType>, DataType,
                              tmpl::list<>, tmpl::list<>, tmpl::list<>> {
  using type = DataType;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;

  ScalarOrDataVector(const DataType t) : t_(std::move(t)) {}

  /// \brief Returns the value represented by the expression
  ///
  /// \details
  /// While a ScalarOrDataVector expression does not store a rank 0 Tensor, it
  /// does represent one. This is why, unlike other derived TensorExpression
  /// types, there is no second variadic template parameter for the generic
  /// indices. In addition, this is why this template is only instantiated for
  /// the case where `Structure` is equal to the structure of such a Tensor.
  ///
  /// \tparam Structure the Structure of the rank 0 Tensor represented by this
  /// expression
  /// \return the value of the rank 0 Tensor represented by this expression
  template <typename Structure>
  SPECTRE_ALWAYS_INLINE const DataType& get(
      const size_t storage_index) const noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE const DataType& get<structure>(
      const size_t /*storage_index*/) const noexcept {
    return t_;
  }

 private:
  const DataType t_;
};
}  // namespace TensorExpressions
