// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines expressions that represent doubles and DataVectors

#pragma once

#include <array>
#include <cstddef>
#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double` or DataVector lvalue
///
/// \details
/// Assumes the scalar used to construct the expression outlives the
/// expression, itself
///
/// \tparam DataType the type being represented, a `double` or DataVector
template <typename DataType,
          Requires<std::is_same_v<DataType, double> or
                   std::is_same_v<DataType, DataVector>> = nullptr>
struct ScalarDataTypeLValue
    : public TensorExpression<ScalarDataTypeLValue<DataType>, DataType,
                              tmpl::list<>, tmpl::list<>, tmpl::list<>> {
  using type = DataType;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;

  /// \brief Create an expression from a scalar type lvalue
  ScalarDataTypeLValue(const DataType& t) : t_(&t) {}
  ScalarDataTypeLValue(const ScalarDataTypeLValue& other) = default;
  ScalarDataTypeLValue(ScalarDataTypeLValue&& other) = default;
  ScalarDataTypeLValue& operator=(const ScalarDataTypeLValue& other) = default;
  ScalarDataTypeLValue& operator=(ScalarDataTypeLValue&& other) = default;
  ~ScalarDataTypeLValue() override = default;

  /// \brief Returns the value represented by the expression
  ///
  /// \details
  /// While a ScalarDataTypeLValue expression does not store a rank 0 Tensor, it
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
    return *t_;
  }

 private:
  /// Refers to the externally stored scalar type lvalue being represented as
  /// an expression
  const DataType* const t_ = nullptr;
};

/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representing a `double` or DataVector rvalue
///
/// \details
/// Unless copy constructed, owns the scalar used to construct the expression
///
/// \tparam DataType the type being represented, a `double` or DataVector
template <typename DataType,
          Requires<std::is_same_v<DataType, double> or
                   std::is_same_v<DataType, DataVector>> = nullptr>
struct ScalarDataTypeRValue
    : public TensorExpression<ScalarDataTypeRValue<DataType>, DataType,
                              tmpl::list<>, tmpl::list<>, tmpl::list<>> {
  using type = DataType;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;

  /// \brief Create an expression from a scalar type rvalue
  ScalarDataTypeRValue(DataType t) : t_(std::move(t)), t_ptr_(&t_) {
    std::cout << "rvalue constructor, t_ is : " << t_ << ", t is " << t
              << std::endl;
  }

  /// \brief Copy constructor
  ///
  /// \details
  /// Instead of deep copying the other ScalarDataTypeRValue, this will simply
  /// copy the other's pointer. This assumes the other ScalarDataTypeRValue
  /// being copied from will outlive this new ScalarDataTypeRValue.
  ScalarDataTypeRValue(const ScalarDataTypeRValue& other) {
    if constexpr (std::is_same_v<DataType, double>) {
      t_ = std::numeric_limits<double>::signaling_NaN();
    } else {
      t_ = DataVector(1, std::numeric_limits<double>::signaling_NaN());
    }
    t_ptr_ = other.t_ptr_;
  }

  /// \brief Move constructor
  ///
  /// \details
  /// This takes ownership of the other ScalarDataTypeRValue's value, which is
  /// relevant for when `DataType==DataVector`.
  ScalarDataTypeRValue(ScalarDataTypeRValue&& other)
      : t_(std::move(other.t_)), t_ptr_(&t_) {}

  /// \brief Copy assignment operator
  ///
  /// \details
  /// Instead of deep copying the other ScalarDataTypeRValue, this will simply
  /// copy the other's pointer. This assumes the other ScalarDataTypeRValue
  /// being copied from will outlive this new ScalarDataTypeRValue.
  ScalarDataTypeRValue& operator=(const ScalarDataTypeRValue& other) {
    std::cout << "rvalue copy assignment " << std::endl;
    if (this != &other) {
      if constexpr (std::is_same_v<DataType, double>) {
        t_ = std::numeric_limits<double>::signaling_NaN();
      } else {
        t_ = DataVector(1, std::numeric_limits<double>::signaling_NaN());
      }
      t_ptr_ = other.t_ptr_;
    }
    return *this;
  }

  /// \brief Move assignment operator
  ///
  /// \details
  /// This takes ownership of the other ScalarDataTypeRValue's value, which is
  /// relevant for when `DataType==DataVector`.
  ScalarDataTypeRValue& operator=(ScalarDataTypeRValue&& other) {
    if (this != &other) {
      t_ = std::move(other.t_);
      t_ptr_ = &t_;
    }
    return *this;
  }

  ~ScalarDataTypeRValue() override = default;

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

  template <>
  SPECTRE_ALWAYS_INLINE const DataType& get<structure>(
      const size_t /*storage_index*/) const noexcept {
    return *t_ptr_;
  }

 private:
  /// If the expression was constructed with a scalar rvalue, this will store
  /// the moved value. Otherwise, if the expression was copy or move
  /// constructed from another ScalarDataTypeRValue, this stores NaN in the
  /// case `DataType==double` and `DataVector{NaN}` in the case `DataType==NaN`.
  DataType t_;
  /// Refers to the scalar type value being represented as an expression.
  /// If the expression was constructed with a scalar rvalue, this refer to the
  /// internally stored value, `t_`. Otherwise, if the expression was copy or
  /// move constructed from another ScalarDataTypeRValue, this refers to the
  /// scalar that the other expression's `t_ptr_` refers to, which may be its
  /// own `t_`, or another ScalarDataTypeRValue's `t_`, if it was also copy or
  /// move constructed.
  const DataType* t_ptr_ = nullptr;
};
}  // namespace TensorExpressions
