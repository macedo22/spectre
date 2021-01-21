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
template <typename DataType, bool IsRValue>
struct ScalarDataType
    : public TensorExpression<ScalarDataType<DataType, IsRValue>, DataType,
                              tmpl::list<>, tmpl::list<>, tmpl::list<>> {
  using type = DataType;
  // using ctor_type = std::conditional_t<IsRValue, DataType, const DataType&>;
  using symmetry = tmpl::list<>;
  using index_list = tmpl::list<>;
  using args_list = tmpl::list<>;
  using structure = Tensor_detail::Structure<symmetry>;
  static constexpr auto num_tensor_indices = 0;
  static constexpr bool is_rvalue = IsRValue;

  /// \brief Create an expression from a scalar type l-value
  /// TODO: make note about DataVector default value for t_ for this
  ScalarDataType(std::conditional_t<IsRValue, DataType, const DataType&> t)
      : t_(IsRValue ? std::move(t)
                    : std::numeric_limits<double>::signaling_NaN()),
        // : t_(IsRValue ? std::move(t) : (std::is_same_v<DataType, double> ?
        // std::numeric_limits<double>::signaling_NaN() : DataVector{})),
        t_ptr_(IsRValue ? &t_ : &t) {
    std::cout << "lvalue constructor, t_ is : " << t_ << ", t is " << t
              << std::endl;
  }

  // Copy constructor - TODO: should this be deleted instead?
  ScalarDataType(const ScalarDataType& other)
      : t_(other.t_),
        t_ptr_(IsRValue ? &t_
                        : other.t_ptr_) { /*std::cout << " copy constructor, t_
                   is : " << t_ << ", t is " << t << std::endl;*/
    // std::cout << "copy constructor body, t_ is : " << t_ << std::endl;
    // std::cout << "copy constructor body, *t_ptr_ is : " << *t_ptr_ <<
    // std::endl; std::cout << "copy constructor body, t_ptr_ is : " << t_ptr_
    // << std::endl;
  }

  // Move constructor
  ScalarDataType(ScalarDataType&& other)
      : t_(std::move(other.t_)),
        t_ptr_(IsRValue ? &t_ : other.t_ptr_) { /*std::cout << "move
              constructor, t_ is : " << t_ << ", t is " << t << std::endl;*/
    // std::cout << "move constructor body, t_ is : " << t_ << std::endl;
    // std::cout << "move constructor body, *t_ptr_ is : " << *t_ptr_ <<
    // std::endl; std::cout << "move constructor body, t_ptr_ is : " << t_ptr_
    // << std::endl;
  }

  //   /// \brief Create an expression from a scalar type l-value
  //   /// TODO: make note about DataVector default value for t_ for this
  //   ScalarDataType(const DataType& t)
  //       : t_(std::numeric_limits<double>::signaling_NaN()),
  //         t_ptr_(&t) { /*std::cout << "lvalue constructor, t_ is : " << t_ <<
  //         ", t
  //                         is " << t << std::endl;*/
  //   }

  //   /// \brief Create an expression from a scalar type r-value
  //   ///
  //   /// \details
  //   /// This overload is necessary so that DataVector r-values are moved to
  //   this
  //   /// expression instead of pointing to an object that will go out of
  //   scope. ScalarDataType(DataType&& t)
  //       : t_(std::move(t)),
  //         t_ptr_(&t_) { /*std::cout << "rvalue constructor, t_ is : " << t_
  //         << ",
  //                          t is " << t << std::endl;*/
  //     // std::cout << "rvalue constructor body, t_ is : " << t_ << std::endl;
  //     // std::cout << "rvalue constructor body, *t_ptr_ is : " << *t_ptr_ <<
  //     // std::endl; std::cout << "rvalue constructor body, t_ptr_ is : " <<
  //     t_ptr_
  //     // << std::endl;
  //   }

  // ScalarDataType(DataType&& t)
  //     : t_((std::cout << "&t in rvalue constructor : " << &t << std::endl,
  //     std::move(t))),
  //       t_ptr_((std::cout << "&t_ in rvalue constructor : " << &t_ <<
  //       std::endl, &t_)) { /*std::cout << "rvalue constructor, t_ is : " <<
  //       t_ << ", t is " << t << std::endl;*/ std::cout << "rvalue constructor
  //       body, t_ is : " << t_ << std::endl;
  //     std::cout << "rvalue constructor body, *t_ptr_ is : " << *t_ptr_ <<
  //     std::endl; std::cout << "rvalue constructor body, t_ptr_ is : " <<
  //     t_ptr_ << std::endl;
  // }

  //   ScalarDataType(const ScalarDataType& other)
  //       : t_(std::move(other.t_)), t_ptr_( &t_) { /*std::cout << "rvalue
  //       constructor, t_ is : " << t_ << ", t is " << t << std::endl;*/
  //       // std::cout << "rvalue constructor body, t_ is : " << t_ <<
  //       std::endl;
  //       // std::cout << "rvalue constructor body, *t_ptr_ is : " << *t_ptr_
  //       << std::endl;
  //       // std::cout << "rvalue constructor body, t_ptr_ is : " << t_ptr_ <<
  //       std::endl;
  //   }

  // ScalarDataType(const ScalarDataType<DataType, true>& other)
  //     : t_(other.t_), t_ptr_(&t_) { /*std::cout << "rvalue constructor, t_ is
  //     :
  //                                      " << t_ << ", t is " << t <<
  //                                      std::endl;*/
  //   // std::cout << "rvalue constructor body, t_ is : " << t_ << std::endl;
  //   // std::cout << "rvalue constructor body, *t_ptr_ is : " << *t_ptr_ <<
  //   // std::endl; std::cout << "rvalue constructor body, t_ptr_ is : " <<
  //   t_ptr_
  //   // << std::endl;
  // }

  // ScalarDataType(const ScalarDataType<DataType, false>& other)
  //     : t_(other.t_),
  //       t_ptr_(other.t_ptr_) { /*std::cout << "rvalue constructor, t_ is : "
  //       <<
  //                                 t_ << ", t is " << t << std::endl;*/
  //   // std::cout << "rvalue constructor body, t_ is : " << t_ << std::endl;
  //   // std::cout << "rvalue constructor body, *t_ptr_ is : " << *t_ptr_ <<
  //   // std::endl; std::cout << "rvalue constructor body, t_ptr_ is : " <<
  //   t_ptr_
  //   // << std::endl;
  // }

  /// \brief Returns the value represented by the expression
  ///
  /// \details
  /// While a ScalarDataType expression does not store a rank 0 Tensor, it
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
    // std::cout << "in get" << std::endl;
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
  const DataType t_;
  /// Refers to the scalar type value being represented as an expression. If the
  /// expression was constructed with an r-value, this will point to `t_`.
  /// Otherwise, it points to an outside value being represented.
  const DataType* const t_ptr_ = nullptr;
};
}  // namespace TensorExpressions