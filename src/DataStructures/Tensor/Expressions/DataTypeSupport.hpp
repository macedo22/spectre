// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// TODO

#pragma once

#include <complex>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/VectorImpl.hpp"

namespace tenex {
namespace detail {
template <typename X>
struct is_supported_tensorexpression_datatype {
  using type = std::bool_constant<
      std::is_same_v<X, double> or std::is_same_v<X, std::complex<double>> or
      std::is_same_v<X, DataVector> or std::is_same_v<X, ComplexDataVector>>;
};

/// \brief The maximum number of arithmetic tensor operations allowed in a
/// `TensorExpression` subtree before having it be a splitting point in the
/// overall RHS expression, according to the data type held by the `Tensor`s in
/// the expression
///
/// \details
/// To enable splitting for `TensorExpression`s with data type, define a
/// template specialization below for your data type and set the `value`.
///
/// Before defining a max operations cap for some data type, the change should
/// first be justified by benchmarking many different tensor expressions before
/// and after introducing the new cap. The optimal cap will likely be
/// hardware-dependent, so fine-tuning this would ideally involve benchmarking
/// on each hardware architecture and then controling the value based on the
/// hardware.
///
/// The current value set for when the data type is `DataVector` was benchmarked
/// by compiling with clang-10 Release and running on Intel(R) Xeon(R)
/// CPU E5-2630 v4 @ 2.20GHz.
///
/// The current value for when the data type is `ComplexDataVector` is set to
/// the same value as for `DataVector`, but its value should also be
/// investigated and fined-tuned.
template <typename DataType>
struct max_num_ops_in_sub_expression_impl {
  // effectively, no splitting for any unspecialized template type
  static constexpr size_t value = std::numeric_limits<size_t>::max();
};

/// \brief When the data type of the result of a `TensorExpression` is
/// `DataVector`, the maximum number of arithmetic tensor operations allowed in
/// a subtree before having it be a splitting point in the overall RHS
/// expression
///
/// \details
/// The current value set for when the data type is `DataVector` was benchmarked
/// by compiling with clang-10 Release and running on Intel(R) Xeon(R)
/// CPU E5-2630 v4 @ 2.20GHz.
template <>
struct max_num_ops_in_sub_expression_impl<DataVector> {
  static constexpr size_t value = 8;
};

/// \brief When the data type of the result of a `TensorExpression` is
/// `ComplexDataVector`, the maximum number of arithmetic tensor operations
/// allowed in a subtree before having it be a splitting point in the overall
/// RHS expression
///
/// \details
/// The current value set for when the data type is `ComplexDataVector` is set
/// to the value for `DataVector`, but the best `value` for `ComplexDataVector`
/// should also be investigated and fine-tuned.
template <>
struct max_num_ops_in_sub_expression_impl<ComplexDataVector> {
  static constexpr size_t value =
      max_num_ops_in_sub_expression_impl<DataVector>::value;
};

/// \brief Get maximum number of arithmetic tensor operations allowed in a
/// `TensorExpression` subtree before having it be a splitting point in the
/// overall RHS expression, according to the `DataType` held by the `Tensor`s in
/// the expression
template <typename DataType>
inline constexpr size_t max_num_ops_in_sub_expression =
    max_num_ops_in_sub_expression_impl<DataType>::value;

// TODO : update this documentation
// For any `T1`, `T2` that does not match a specialization below, binary
// operations between the two types is said to be undefined. To add support for
// binary operations between two data types, define a new specialization with
// a `type` alias to the resulting data type.
template <typename X1, typename X2, typename = void>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl : std::false_type {};

// A binary operation between two terms of the same type will yield a result
// with that type
template <typename X>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl<X, X> : std::true_type {
};

// A binary operation between a `double` and `std::complex<double>` will yield a
// `std::complex<double>` result
template <typename T>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl<std::complex<T>, T>
    : std::true_type {};

template <>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl<DataVector, double>
    : std::true_type {};

// A binary operation between a `VectorImpl` type and its `value_type` will
// yield a result with the `VectorImpl` type, e.g. adding a `DataVector` and a
// `double` is defined and the result is a `DataVector`
template <>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl<ComplexDataVector,
                                                       double>
    : std::true_type {};

template <>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl<ComplexDataVector,
                                                       std::complex<double>>
    : std::true_type {};

// A binary operation between a `ComplexDataVector` and `DataVector` will yield
// a `ComplexDataVector` result
template <>
struct rhs_datatype_is_assignable_to_lhs_datatype_impl<ComplexDataVector,
                                                       DataVector>
    : std::true_type {};

template <typename LhsDataType, typename RhsDataType>
struct rhs_datatype_is_assignable_to_lhs_datatype {
  using type =
      rhs_datatype_is_assignable_to_lhs_datatype_impl<LhsDataType, RhsDataType>;
};

template <typename X1, typename X2, typename = void>
struct binop_datatypes_are_valid_impl : std::false_type {};

template <typename X>
struct binop_datatypes_are_valid_impl<X, X> : std::true_type {};

template <typename ValueType>
struct binop_datatypes_are_valid_impl<ValueType, std::complex<ValueType>>
    : std::true_type {};
template <typename ValueType>
struct binop_datatypes_are_valid_impl<std::complex<ValueType>, ValueType>
    : std::true_type {};

template <>
struct binop_datatypes_are_valid_impl<DataVector, double> : std::true_type {};
template <>
struct binop_datatypes_are_valid_impl<double, DataVector> : std::true_type {};

template <>
struct binop_datatypes_are_valid_impl<ComplexDataVector, double>
    : std::true_type {};
template <>
struct binop_datatypes_are_valid_impl<double, ComplexDataVector>
    : std::true_type {};

template <typename ValueType>
struct binop_datatypes_are_valid_impl<ComplexDataVector,
                                      std::complex<ValueType>>
    : std::true_type {};
template <typename ValueType>
struct binop_datatypes_are_valid_impl<std::complex<ValueType>,
                                      ComplexDataVector> : std::true_type {};

template <>
struct binop_datatypes_are_valid_impl<ComplexDataVector, DataVector>
    : std::true_type {};
template <>
struct binop_datatypes_are_valid_impl<DataVector, ComplexDataVector>
    : std::true_type {};

template <typename X1, typename X2>
struct binop_datatypes_are_valid {
  using type =
      std::bool_constant<(binop_datatypes_are_valid_impl<X1, X2>::value)>;
};

// For any `X1`, `X2` that does not match a specialization below, binary
// operations between the two types is said to be undefined. To add support for
// binary operations between two data types, define a new specialization with
// a `type` alias to the resulting data type.
template <typename X1, typename X2>
struct get_binop_datatype_impl {
  using type = std::bool_constant<false>;
};

// A binary operation between two terms of the same type will yield a result
// with that type
template <typename X>
struct get_binop_datatype_impl<X, X> {
  using type = X;
};

template <typename ValueType>
struct get_binop_datatype_impl<ValueType, std::complex<ValueType>> {
  using type = std::complex<ValueType>;
};
template <typename ValueType>
struct get_binop_datatype_impl<std::complex<ValueType>, ValueType> {
  using type = std::complex<ValueType>;
};

// A binary operation between a `VectorImpl` type and its `value_type` will
// yield a result with the `VectorImpl` type, e.g. adding a `DataVector` and a
// `double` is defined and the result is a `DataVector`
template <>
struct get_binop_datatype_impl<DataVector, double> {
  using type = DataVector;
};
template <>
struct get_binop_datatype_impl<double, DataVector> {
  using type = DataVector;
};

// A binary operation between a `ComplexDataVector` and `double` will yield a
// `ComplexDataVector` result
template <>
struct get_binop_datatype_impl<ComplexDataVector, double> {
  using type = ComplexDataVector;
};
template <>
struct get_binop_datatype_impl<double, ComplexDataVector> {
  using type = ComplexDataVector;
};

template <>
struct get_binop_datatype_impl<ComplexDataVector, std::complex<double>> {
  using type = ComplexDataVector;
};
template <>
struct get_binop_datatype_impl<std::complex<double>, ComplexDataVector> {
  using type = ComplexDataVector;
};

// A binary operation between a `ComplexDataVector` and `DataVector` will yield
// a `ComplexDataVector` result
template <>
struct get_binop_datatype_impl<ComplexDataVector, DataVector> {
  using type = ComplexDataVector;
};
template <>
struct get_binop_datatype_impl<DataVector, ComplexDataVector> {
  using type = ComplexDataVector;
};

/// \brief Get the data type of a binary operation between two data types
/// that may occur in a `TensorExpression`
///
/// \tparam X1 the data type of one operand
/// \tparam X2 the data type of the other operand
template <typename X1, typename X2>
struct get_binop_datatype {
  static_assert(
      binop_datatypes_are_valid<X1, X2>::type::value,
      "You are attempting to perform a binary arithmetic operation between "
      "two data types, but binary arithmetic operations between these two "
      "types is not valid within TensorExpressions.");

  using type = typename get_binop_datatype_impl<X1, X2>::type;

  static_assert(
      not std::is_same_v<type, std::bool_constant<false>>,
      "You are attempting to perform a binary arithmetic operation between "
      "two data types, but binary arithmetic operations between these two "
      "types is not defined within TensorExpressions.");
};

// For any `T1`, `T2` that does not match a specialization below, binary
// operations between the two types is said to be undefined. To add support for
// binary operations between two data types, define a new specialization with
// a `type` alias to the resulting data type.
template <typename X1, typename X2, typename = void>
struct tensor_binop_datatypes_are_valid_impl : std::false_type {};

// A binary operation between two terms of the same type will yield a result
// with that type
template <typename X>
struct tensor_binop_datatypes_are_valid_impl<X, X> : std::true_type {};

// A binary operation between a `double` and `std::complex<double>` will yield a
// `std::complex<double>` result
template <typename T>
struct tensor_binop_datatypes_are_valid_impl<T, std::complex<T>>
    : std::true_type {};
template <typename T>
struct tensor_binop_datatypes_are_valid_impl<std::complex<T>, T>
    : std::true_type {};

// A binary operation between a `VectorImpl` type and its `value_type` will
// yield a result with the `VectorImpl` type, e.g. adding a `DataVector` and a
// `double` is defined and the result is a `DataVector`
template <typename ValueType, typename VectorType>
struct tensor_binop_datatypes_are_valid_impl<VectorImpl<ValueType, VectorType>,
                                             ValueType> : std::true_type {};
template <typename ValueType, typename VectorType>
struct tensor_binop_datatypes_are_valid_impl<ValueType,
                                             VectorImpl<ValueType, VectorType>>
    : std::true_type {};

// A binary operation between a `ComplexDataVector` and `DataVector` will yield
// a `ComplexDataVector` result
template <>
struct tensor_binop_datatypes_are_valid_impl<ComplexDataVector, DataVector>
    : std::true_type {};
template <>
struct tensor_binop_datatypes_are_valid_impl<DataVector, ComplexDataVector>
    : std::true_type {};

// TODO : define the valid bin_ops in each bin_op file but reduce this
// helper to just checking which TENSOR expression bin op data types are ok
/// \brief Check whether or not a binary operation between two
/// `TensorExpression`s is valid
///
/// \details This is different from `get_binop_datatype` in that while it may
/// be possible to perform a binary operation between two data types such as
/// `double` and `DataVector`, it may not be valid to perform a binary operation
/// between two `TensorExpression`s with those data types. For example,
/// `TensorAsExpression<double, ...> + TensorAsExpression<DataVector, ...>` is
/// not valid even though it is a valid C++ operation to do
/// `double + DataVector`. Alternatively, it is valid to do
/// `NumberAsExpression + TensorAsExpression<DataVector, ...>`.
///
/// \tparam X1 TODO
/// \tparam X2 TODO
template <typename X1, typename X2>
struct tensor_binop_datatypes_are_valid {
  using type = tensor_binop_datatypes_are_valid_impl<X1, X2>;
};

template <typename T1, typename T2>
struct tensorexpression_binop_datatypes_are_valid_impl {
  using type =
      typename tensor_binop_datatypes_are_valid<typename T1::type,
                                                typename T2::type>::type;
};

template <typename TensorExpressionType, typename NumberType>
struct tensorexpression_binop_datatypes_are_valid_impl<
    TensorExpressionType, NumberAsExpression<NumberType>> {
  using result_datatype =
      typename get_binop_datatype<typename TensorExpressionType::type,
                                  NumberType>::type;
  using type = std::bool_constant<(
      not std::is_same_v<result_datatype, std::false_type>)>;
};
template <typename NumberType, typename TensorExpressionType>
struct tensorexpression_binop_datatypes_are_valid_impl<
    NumberAsExpression<NumberType>, TensorExpressionType> {
  using type = typename tensorexpression_binop_datatypes_are_valid_impl<
      TensorExpressionType, NumberAsExpression<NumberType>>::type;
};

template <typename T1, typename T2>
struct tensorexpression_binop_datatypes_are_valid {
  using type =
      typename tensorexpression_binop_datatypes_are_valid_impl<T1, T2>::type;
};
}  // namespace detail
}  // namespace tenex
