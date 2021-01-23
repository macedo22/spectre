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
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representating a scalar or a DataVector of
/// scalars
///
/// \tparam DataType the type being represented, a `double` or DataVector
template <typename DataType, bool IsRValue,
          Requires<std::is_same_v<DataType, double> or
                   std::is_same_v<DataType, DataVector>> = nullptr>
struct ScalarDataType
    : public TensorExpression<ScalarDataType<DataType, IsRValue>, DataType,
                              tmpl::list<>, tmpl::list<>, tmpl::list<>> {
  using type = DataType;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;

  /// \brief Create an expression from a scalar type r-value
  ScalarDataType(std::conditional_t<IsRValue, DataType, const DataType&> t) {
    if constexpr (IsRValue) {
      // double or DataVector rvalue
      t_ = std::move(t);
      t_ptr_ = &t_;
    } else if constexpr (std::is_same_v<DataType, double>) {
      // double lvalue
      t_ = std::numeric_limits<double>::signaling_NaN();
      t_ptr_ = &t;
    } else {
      // DataVector lvalue
      t_ = DataVector(1, std::numeric_limits<double>::signaling_NaN());
      t_ptr_ = &t;
    }
    std::cout << "constructor, *t_ptr_ is : " << *t_ptr_ << std::endl;
  }

  // copy constructor
  ScalarDataType(const ScalarDataType& other) {
    if constexpr (std::is_same_v<DataType, double>) {
      t_ = std::numeric_limits<double>::signaling_NaN();
    } else {
      t_ = DataVector(1, std::numeric_limits<double>::signaling_NaN());
    }
    t_ptr_ = other.t_ptr_;
    std::cout << "copy constructor, *t_ptr_ is : " << *t_ptr_ << std::endl;
  }

  // ScalarDataType(const ScalarDataType<DataVector>& other) =
  // delete;

  // Note: if copy constructor is defined, this must be too, or move
  // will default to the copy? seems to be this way from experimenting,
  // but would be good to confirm otherwise
  ScalarDataType(ScalarDataType&& other) {
    if constexpr (std::is_same_v<DataType, double>) {
      t_ = std::numeric_limits<double>::signaling_NaN();
    } else {
      t_ = DataVector(1, std::numeric_limits<double>::signaling_NaN());
    }
    t_ptr_ = other.t_ptr_;
    std::cout << "move constructor, *t_ptr_ is : " << *t_ptr_ << std::endl;
  }

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
    return *t_ptr_;
  }

 private:
  /// The scalar type value being represented as an expression. If the
  /// expression was constructed with an r-value, this will store the moved
  /// value. Otherwise, the represented value is instead referred to by
  /// `t_ptr_`.
  DataType t_;
  /// Refers to the scalar type value being represented as an expression.
  /// If the
  /// expression was constructed with an r-value, this will point to `t_`.
  /// Otherwise, it points to an outside value being represented.
  const DataType* t_ptr_ = nullptr;
};

/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representating a scalar or a DataVector of
/// scalars
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

  /// \brief Create an expression from a scalar type l-value
  ScalarDataTypeLValue(const DataType& t) : t_(&t) {
    // std::cout << "lvalue constructor, t_ is : " << t_ << ", t is " << t
    //           << std::endl;
  }

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

  // template <class...T>
  // struct td;

  template <>
  SPECTRE_ALWAYS_INLINE const DataType& get<structure>(
      const size_t /*storage_index*/) const noexcept {
    // std::cout << "in lvalue get" << std::endl;
    // std::cout << "t_ : " << t_ << std::endl;
    // std::cout << "t_ptr_ : " << t_ptr_ << std::endl;
    // std::cout << "*t_ptr_ : " << *t_ptr_ << std::endl;
    // td<DataType>idk;
    return *t_;
  }

 private:
  //   /// The scalar type value being represented as an expression. If the
  //   /// expression was constructed with an r-value, this will store the moved
  //   /// value. Otherwise, the represented value is instead referred to by
  //   /// `t_ptr_`.
  //   const DataType t_;
  /// Refers to the scalar type value being represented as an expression. If the
  /// expression was constructed with an r-value, this will point to `t_`.
  /// Otherwise, it points to an outside value being represented.
  const DataType* const t_ = nullptr;
};

/// \ingroup TensorExpressionsGroup
/// \brief Defines an expression representating a scalar or a DataVector of
/// scalars
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

  /// \brief Create an expression from a scalar type r-value
  ScalarDataTypeRValue(DataType t) : t_(std::move(t)), t_ptr_(&t_) {
    // std::cout << "rvalue constructor, t_ is : " << t_ << ", t is " << t
    //           << std::endl;
  }

  // copy constructor
  ScalarDataTypeRValue(const ScalarDataTypeRValue& other) {
    if constexpr (std::is_same_v<DataType, double>) {
      t_ = std::numeric_limits<double>::signaling_NaN();
      t_ptr_ = &other.t_ptr_;
    } else {
      t_ = DataVector(1, std::numeric_limits<double>::signaling_NaN());
      t_ptr_ = &other.t_ptr_;
    }
    std::cout << "rvalue copy constructor, *t_ptr_ is : " << *t_ptr_
              << std::endl;
  }

  // ScalarDataTypeRValue(const ScalarDataTypeRValue<DataVector>& other) =
  // delete;

  // Note: if copy constructor is defined, this must be too, or move
  // will default to the copy? seems to be this way from experimenting,
  // but would be good to confirm otherwise
  ScalarDataTypeRValue(ScalarDataTypeRValue&& other)
      : t_(std::move(other.t_)), t_ptr_(&t_) {
    std::cout << "rvalue move constructor, t_ is : " << t_ << std::endl;
  }

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
    return *t_ptr_;
  }

 private:
  /// The scalar type value being represented as an expression. If the
  /// expression was constructed with an r-value, this will store the moved
  /// value. Otherwise, the represented value is instead referred to by
  /// `t_ptr_`.
  DataType t_;
  /// Refers to the scalar type value being represented as an expression.
  /// If the
  /// expression was constructed with an r-value, this will point to `t_`.
  /// Otherwise, it points to an outside value being represented.
  DataType* t_ptr_ = nullptr;
};
}  // namespace TensorExpressions
