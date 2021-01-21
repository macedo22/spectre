// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent scalars and DataVectors

#pragma once

#include <array>
#include <cstddef>
#include <iostream>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representating a scalar or a DataVector of
/// scalars
///
/// \tparam DataType the type being represented, a `double` or DataVector
template <typename DataType>
struct ScalarDataTypeRValue
    : public TensorExpression<ScalarDataTypeRValue<DataType>, DataType,
                              tmpl::list<>, tmpl::list<>, tmpl::list<>> {
  using type = DataType;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;

  /// \brief Create an expression from a scalar type r-value
  ScalarDataTypeRValue(DataType t) : t_(std::move(t)) {
    // std::cout << "rvalue constructor, t_ is : " << t_ << ", t is " << t
    //           << std::endl;
  }

  // copy constructor to see if and when it's called
  ScalarDataTypeRValue(const ScalarDataTypeRValue& other) : t_(other.t_) {
    std::cout << "rvalue copy constructor, t_ is : " << t_ << std::endl;
  }

  // Note: if copy constructor is defined, this must be too, or move
  // will default to the copy? seems to be this way from experimenting,
  // but would be good to confirm otherwise
  ScalarDataTypeRValue(ScalarDataTypeRValue&&) noexcept = default;

  /// \brief Returns the value represented by the expression
  ///
  /// \details
  /// While a ScalarDataTypeRValue expression does not store a rank 0 Tensor, it
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

  // template <class...T>
  // struct td;

  template <>
  SPECTRE_ALWAYS_INLINE const DataType& get<structure>(
      const size_t /*storage_index*/) const noexcept {
    // std::cout << "in rvalue get" << std::endl;
    // std::cout << "t_ : " << t_ << std::endl;
    // std::cout << "t_ptr_ : " << t_ptr_ << std::endl;
    // std::cout << "*t_ptr_ : " << *t_ptr_ << std::endl;
    // td<DataType>idk;
    return t_;
  }

 private:
  /// The scalar type value being represented as an expression. If the
  /// expression was constructed with an r-value, this will store the moved
  /// value. Otherwise, the represented value is instead referred to by
  /// `t_ptr_`.
  const DataType t_;
  //   /// Refers to the scalar type value being represented as an expression.
  //   If the
  //   /// expression was constructed with an r-value, this will point to `t_`.
  //   /// Otherwise, it points to an outside value being represented.
  //   const DataType* const t_ptr_ = nullptr;
};
}  // namespace TensorExpressions
