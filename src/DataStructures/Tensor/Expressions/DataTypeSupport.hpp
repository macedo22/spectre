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

template <typename T, typename = void>
struct is_complex_number : std::false_type {};
template <typename T>
struct is_complex_number<std::complex<T>> : std::true_type {};

template <typename T>
using is_real_number = std::bool_constant<(std::is_arithmetic_v<T>)>;

template <typename T>
using is_number = std::bool_constant<(is_real_number<T>::value or
                                      is_complex_number<T>::value)>;

template <typename T>
using is_vector = std::bool_constant<std::is_base_of_v<MarkAsVectorImpl, T>>;

template <typename T>
using is_number_or_vector =
    std::bool_constant<is_number<T>::value or is_vector<T>::value>;

// template <typename T, Requires<is_vector<T>::value> = nullptr>
// struct is_real_vector_impl {
//   using type =
//       std::bool_constant<is_vector<T>::value and
//                          is_real_number<typename T::value_type>::value>;
// };

// template <typename T, Requires<not is_vector<T>::value> = nullptr>
// struct is_real_vector_impl {
//   using type = std::bool_constant<false>;
// };

// template <typename T>
// using is_real_vector = typename is_real_vector_impl<T>::type;

template <typename T, Requires<is_vector<T>::value> = nullptr>
constexpr bool is_real_vector_impl() {
  return is_real_number<typename T::value_type>::value;
}

template <typename T, Requires<not is_vector<T>::value> = nullptr>
constexpr bool is_real_vector_impl() {
  return false;
}

template <typename T>
using is_real_vector = std::bool_constant<is_real_vector_impl<T>()>;

// template <typename T, Requires<is_vector<T>::value> = nullptr>
// struct is_complex_vector_impl {
//   using type = std::bool_constant<is_vector<T>::value and
//                                   is_complex_number<typename
//                                   T::value_type>::value>;
// };

// template <typename T, Requires<not is_vector<T>::value> = nullptr>
// struct is_complex_vector_impl {
//   using type = std::bool_constant<false>;
// };

// template <typename T>
// using is_complex_vector = typename is_complex_vector_impl<T>::type;

template <typename T, Requires<is_vector<T>::value> = nullptr>
constexpr bool is_complex_vector_impl() {
  return is_complex_number<typename T::value_type>::value;
}

template <typename T, Requires<not is_vector<T>::value> = nullptr>
constexpr bool is_complex_vector_impl() {
  return false;
}

template <typename T>
using is_complex_vector = std::bool_constant<is_complex_vector_impl<T>()>;

template <typename T>
using is_real = std::bool_constant<is_real_number<T>::value or is_real_vector_impl<T>()>;

template <typename T>
using is_complex = std::bool_constant<is_complex_number<T>::value or is_complex_vector_impl<T>()>;

template <typename MaybeComplexDataType, typename DataType, typename = void>
struct is_complex_datatype_of : std::false_type {};
template <typename T>
struct is_complex_datatype_of<std::complex<T>, T> : std::true_type {};
template <>
struct is_complex_datatype_of<ComplexDataVector, DataVector> : std::true_type {
};
template <>
struct is_complex_datatype_of<ComplexModalVector, ModalVector>
    : std::true_type {};
// template <>
// struct is_complex_datatype_of<typename ComplexDataVector::BaseType,
//                               typename DataVector::BaseType> : std::true_type
//                               {
// };
// template <>
// struct is_complex_datatype_of<typename ComplexModalVector::BaseType,
//                               typename ModalVector::BaseType> :
//                               std::true_type {
// };

template <typename T, typename = void>
struct upcast_if_derived_vector_type {
  using type = T;
};

template <typename T>
struct upcast_if_derived_vector_type<
    T, typename std::enable_if<is_vector<T>::value>::type> {
  using upcasted_type = typename T::BaseType;
  // if we have a derived VectorImpl, get base type, else T is a base VectorImpl
  // and we use that
  using type =
      tmpl::conditional_t<std::is_base_of_v<MarkAsVectorImpl, upcasted_type>,
                          upcasted_type, T>;
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
struct lhs_datatype_is_assignable_to_rhs_datatype_impl : std::false_type {};

// A binary operation between two terms of the same type will yield a result
// with that type
template <typename X>
struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X, X> : std::true_type {
};

// A binary operation between a `double` and `std::complex<double>` will yield a
// `std::complex<double>` result
template <typename T>
struct lhs_datatype_is_assignable_to_rhs_datatype_impl<std::complex<T>, T>
    : std::true_type {};

template <typename ValueType, typename VectorType>
struct lhs_datatype_is_assignable_to_rhs_datatype_impl<
    VectorImpl<ValueType, VectorType>, ValueType> : std::true_type {};

template <typename ValueType, typename VectorType>
struct lhs_datatype_is_assignable_to_rhs_datatype_impl<
    VectorImpl<std::complex<ValueType>, VectorType>, ValueType>
    : std::true_type {};

template <>
struct lhs_datatype_is_assignable_to_rhs_datatype_impl<
    typename ComplexDataVector::BaseType, typename DataVector::BaseType>
    : std::true_type {};

// // template <>
// // struct lhs_datatype_is_assignable_to_rhs_datatype_impl<DataVector, double>
// //     : std::true_type {};

// // A binary operation between a `VectorImpl` type and its `value_type` will
// // yield a result with the `VectorImpl` type, e.g. adding a `DataVector` and
// a
// // `double` is defined and the result is a `DataVector`
// template <>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<ComplexDataVector,
//                                                        double>
//     : std::true_type {};

// template <>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<ComplexDataVector,
//                                                        std::complex<double>>
//     : std::true_type {};

// // A binary operation between a `ComplexDataVector` and `DataVector` will
// yield
// // a `ComplexDataVector` result
// template <>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<ComplexDataVector,
//                                                        DataVector>
//     : std::true_type {};

// template <typename X1, typename X2, typename = void>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl : std::false_type {};

// // template <typename X1, typename X2>
// // struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X1, X2,
// Requires<(is_vector<X1>::value and not is_vector<X2>::value)>> :
// std::false_type {};

// // template <typename X1, typename X2>
// // struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X1, X2,
// Requires<(not is_vector<X1>::value and is_vector<X2>::value)>> :
// std::false_type {};

// // template <typename X1, typename X2>
// // struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X1, X2,
// Requires<(not is_vector<X1>::value and not is_vector<X2>::value)>> :
// std::false_type {};

// // template <typename X1, typename X2>
// // struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X1, X2,
// Requires<(is_vector<X1>::value is_vector<X2>::value)>> : std::false_type {};

// template <typename X>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X, X> : std::true_type
// {};

// template <typename T>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<std::complex<T>, T> :
// std::true_type {};

// template <typename X1, typename X2>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X1, X2, typename
// std::enable_if<is_vector<X1>::value>::type> :
//     std::bool_constant<(std::is_same_v<typename X1::value_type, X2> or
//     std::is_same_v<typename X1::value_type, std::complex<X2>> or
//                         is_complex_datatype_of<X1, X2>::value)> {};

// template <typename X1, typename X2>
// struct lhs_datatype_is_assignable_to_rhs_datatype_impl<X1, X2, typename
// std::enable_if<not std::is_same_v<X1, X2> and not
// is_vector<X1>::value>::type> :
//     std::false_type {};

template <typename LhsDataType, typename RhsDataType>
struct lhs_datatype_is_assignable_to_rhs_datatype {
  static_assert(
      is_supported_tensorexpression_datatype<LhsDataType>::type::value and
          is_supported_tensorexpression_datatype<RhsDataType>::type::value,
      "Cannot assign the LHS Tensor's data type to the RHS TensorExpression's "
      "data type because at least one of the data types is not supported by "
      "TensorExpressions.");
  using type = lhs_datatype_is_assignable_to_rhs_datatype_impl<
      typename upcast_if_derived_vector_type<LhsDataType>::type,
      typename upcast_if_derived_vector_type<RhsDataType>::type>;
};

// TODO : remove binop_datatypes_are_supported and then anything
// that needs to check if binop datatypes are valid can just call
// get_binop_data_types_impl to avoid the static assert check in
// get_binop_data_types

// TODO simplify things in general by just separating into categories:
// 1. number, make a helper struct that checks if std::complex or std::is_arithmetic
// 2. vector, add a VectorType base class and check if base of
// 3. else, false type
//
// Then, for get_bin_op, do each combo:
// 1. number op number
// 2. number op vector
// 3. vector op number
// 4. vector op vector
//
// Then for tensor/TE ops, need to fix data type checking in failed tests

// TODO : only have one ordering of nested helper type calls to
// reduce # of instantiations? e.g. <double, complex> and <complex, double>

// template <typename X1, typename X2, typename = void>
// struct binop_datatypes_are_supported_impl : std::false_type {};

// template <typename X>
// struct binop_datatypes_are_supported_impl<X, X> : std::true_type {};

// template <typename ValueType>
// struct binop_datatypes_are_supported_impl<ValueType, std::complex<ValueType>>
//     : std::true_type {};
// template <typename ValueType>
// struct binop_datatypes_are_supported_impl<std::complex<ValueType>, ValueType>
//     : std::true_type {};

// template <>
// struct binop_datatypes_are_supported_impl<DataVector, double> :
// std::true_type {
// };
// template <>
// struct binop_datatypes_are_supported_impl<double, DataVector> :
// std::true_type {
// };

// template <>
// struct binop_datatypes_are_supported_impl<ComplexDataVector, double>
//     : std::true_type {};
// template <>
// struct binop_datatypes_are_supported_impl<double, ComplexDataVector>
//     : std::true_type {};

// template <typename ValueType>
// struct binop_datatypes_are_supported_impl<ComplexDataVector,
//                                           std::complex<ValueType>>
//     : std::true_type {};
// template <typename ValueType>
// struct binop_datatypes_are_supported_impl<std::complex<ValueType>,
//                                           ComplexDataVector> : std::true_type
//                                           {
// };

// template <>
// struct binop_datatypes_are_supported_impl<ComplexDataVector, DataVector>
//     : std::true_type {};
// template <>
// struct binop_datatypes_are_supported_impl<DataVector, ComplexDataVector>
//     : std::true_type {};

// template <typename X1, typename X2>
// struct binop_datatypes_are_supported {
//   static_assert(
//       is_supported_tensorexpression_datatype<X1>::type::value and
//           is_supported_tensorexpression_datatype<X2>::type::value,
//       "Cannot perform binary operations between the two given data types "
//       "because at least one of the data types is not supported by "
//       "TensorExpressions.");
//   using type =
//       std::bool_constant<(binop_datatypes_are_supported_impl<X1,
//       X2>::value)>;
// };

// For any `X1`, `X2` that does not match a specialization below, binary
// operations between the two types is said to be undefined. To add support for
// binary operations between two data types, define a new specialization with
// a `type` alias to the resulting data type.
// template <typename X1, typename X2,
//           Requires<not is_number_or_vector<X1>::value or
//                    not is_number_or_vector<X1>::value> = nullptr>
// struct get_binop_datatype_impl {
//   using type = std::bool_constant<false>;
// };

// template <typename X1, typename X2,
//           Requires<is_number_or_vector<X1>::value and
//                    is_number_or_vector<X1>::value> = nullptr>
// struct get_binop_datatype_impl {
//   using type = std::bool_constant<false>;
// };

template <typename X1, typename X2, typename = void>
struct get_binop_datatype_impl {
  using type = std::bool_constant<false>;
};

// A binary operation between two terms of the same type will yield a result
// with that type
template <typename X>
struct get_binop_datatype_impl<X, X> {
  using type = X;
};

template <typename ValueType, typename VectorType>
struct get_binop_datatype_impl<VectorImpl<ValueType, VectorType>,
                               VectorImpl<ValueType, VectorType>> {
  using type = VectorType;
};

template <typename ValueType>
struct get_binop_datatype_impl<ValueType, std::complex<ValueType>> {
  using type = std::complex<ValueType>;
};
template <typename ValueType>
struct get_binop_datatype_impl<std::complex<ValueType>, ValueType> {
  using type = std::complex<ValueType>;
};

template <typename ValueType, typename VectorType>
struct get_binop_datatype_impl<VectorImpl<ValueType, VectorType>, ValueType> {
  using type = VectorType;
};
template <typename ValueType, typename VectorType>
struct get_binop_datatype_impl<ValueType, VectorImpl<ValueType, VectorType>> {
  using type = VectorType;
};

template <typename ValueType, typename VectorType>
struct get_binop_datatype_impl<VectorImpl<std::complex<ValueType>, VectorType>,
                               ValueType> {
  using type = VectorType;
};
template <typename ValueType, typename VectorType>
struct get_binop_datatype_impl<
    ValueType, VectorImpl<std::complex<ValueType>, VectorType>> {
  using type = VectorType;
};

template <>
struct get_binop_datatype_impl<typename ComplexDataVector::BaseType,
                               typename DataVector::BaseType> {
  using type = ComplexDataVector;
};
template <>
struct get_binop_datatype_impl<typename DataVector::BaseType,
                               typename ComplexDataVector::BaseType> {
  using type = ComplexDataVector;
};

// // A binary operation between a `VectorImpl` type and its `value_type` will
// // yield a result with the `VectorImpl` type, e.g. adding a `DataVector` and
// a
// // `double` is defined and the result is a `DataVector`
// template <>
// struct get_binop_datatype_impl<DataVector, double> {
//   using type = DataVector;
// };
// template <>
// struct get_binop_datatype_impl<double, DataVector> {
//   using type = DataVector;
// };

// // A binary operation between a `ComplexDataVector` and `double` will yield a
// // `ComplexDataVector` result
// template <>
// struct get_binop_datatype_impl<ComplexDataVector, double> {
//   using type = ComplexDataVector;
// };
// template <>
// struct get_binop_datatype_impl<double, ComplexDataVector> {
//   using type = ComplexDataVector;
// };

// template <>
// struct get_binop_datatype_impl<ComplexDataVector, std::complex<double>> {
//   using type = ComplexDataVector;
// };
// template <>
// struct get_binop_datatype_impl<std::complex<double>, ComplexDataVector> {
//   using type = ComplexDataVector;
// };

// // A binary operation between a `ComplexDataVector` and `DataVector` will
// yield
// // a `ComplexDataVector` result
// template <>
// struct get_binop_datatype_impl<ComplexDataVector, DataVector> {
//   using type = ComplexDataVector;
// };
// template <>
// struct get_binop_datatype_impl<DataVector, ComplexDataVector> {
//   using type = ComplexDataVector;
// };

template <typename X1, typename X2>
struct binop_datatypes_are_supported {
  using type = std::bool_constant<not std::is_same_v<
      typename get_binop_datatype_impl<
          typename upcast_if_derived_vector_type<X1>::type,
          typename upcast_if_derived_vector_type<X2>::type>::type,
      std::bool_constant<false>>>;
};

/// \brief Get the data type of a binary operation between two data types
/// that may occur in a `TensorExpression`
///
/// \tparam X1 the data type of one operand
/// \tparam X2 the data type of the other operand
template <typename X1, typename X2>
struct get_binop_datatype {
  using type = typename get_binop_datatype_impl<
      typename upcast_if_derived_vector_type<X1>::type,
      typename upcast_if_derived_vector_type<X2>::type>::type;

  static_assert(
      not std::is_same_v<type, std::bool_constant<false>>,
      "You are attempting to perform a binary arithmetic operation between "
      "two data types, but the data type of the result is not known within "
      "TensorExpressions.");
};

// // For any `T1`, `T2` that does not match a specialization below, binary
// // operations between the two types is said to be undefined. To add support
// for
// // binary operations between two data types, define a new specialization with
// // a `type` alias to the resulting data type.
// template <typename X1, typename X2, typename = void>
// struct tensor_binop_datatypes_are_supported_impl :
// std::bool_constant<(is_complex_datatype_of<X1, X2>::value or
// is_complex_datatype_of<X1, X2>::value)> {};

// // A binary operation between two terms of the same type will yield a result
// // with that type
// template <typename X>
// struct tensor_binop_datatypes_are_supported_impl<X, X> : std::true_type {};

// // A binary operation between a `double` and `std::complex<double>` will
// yield a
// // `std::complex<double>` result
// template <typename T>
// struct tensor_binop_datatypes_are_supported_impl<T, std::complex<T>>
//     : std::true_type {};
// template <typename T>
// struct tensor_binop_datatypes_are_supported_impl<std::complex<T>, T>
//     : std::true_type {};

// // A binary operation between a `VectorImpl` type and its `value_type` will
// // yield a result with the `VectorImpl` type, e.g. adding a `DataVector` and
// a
// // `double` is defined and the result is a `DataVector`
// template <typename ValueType, typename VectorType>
// struct tensor_binop_datatypes_are_supported_impl<
//     VectorImpl<ValueType, VectorType>, ValueType> : std::true_type {};
// template <typename ValueType, typename VectorType>
// struct tensor_binop_datatypes_are_supported_impl<
//     ValueType, VectorImpl<ValueType, VectorType>> : std::true_type {};

// // A binary operation between a `ComplexDataVector` and `DataVector` will
// yield
// // a `ComplexDataVector` result
// template <>
// struct tensor_binop_datatypes_are_supported_impl<ComplexDataVector,
// DataVector>
//     : std::true_type {};
// template <>
// struct tensor_binop_datatypes_are_supported_impl<DataVector,
// ComplexDataVector>
//     : std::true_type {};
template <typename X1, typename X2>
struct tensor_binop_datatypes_are_supported_impl
    : std::bool_constant<(std::is_same_v<X1, X2> or
                          is_complex_datatype_of<X1, X2>::value or
                          is_complex_datatype_of<X2, X1>::value)> {};

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
struct tensor_binop_datatypes_are_supported {
  static_assert(
      is_supported_tensorexpression_datatype<X1>::type::value and
          is_supported_tensorexpression_datatype<X2>::type::value,
      "Cannot perform binary operations between the two Tensors with the "
      "given data types because at least one of the data types is not "
      "supported by TensorExpressions.");
  using type = typename tensor_binop_datatypes_are_supported_impl<X1, X2>::type;
};

template <typename T1, typename T2>
struct tensorexpression_binop_datatypes_are_supported_impl {
  using type =
      typename tensor_binop_datatypes_are_supported<typename T1::type,
                                                    typename T2::type>::type;
};

template <typename TensorExpressionType, typename NumberType>
struct tensorexpression_binop_datatypes_are_supported_impl<
    TensorExpressionType, NumberAsExpression<NumberType>> {
  using type = std::bool_constant<(binop_datatypes_are_supported<
      typename TensorExpressionType::type, NumberType>::type::value and
      not (is_real<typename TensorExpressionType::type>::value and is_complex<NumberType>::value))>;
};
template <typename NumberType, typename TensorExpressionType>
struct tensorexpression_binop_datatypes_are_supported_impl<
    NumberAsExpression<NumberType>, TensorExpressionType> {
  using type = typename tensorexpression_binop_datatypes_are_supported_impl<
      TensorExpressionType, NumberAsExpression<NumberType>>::type;
};

template <typename T1, typename T2>
struct tensorexpression_binop_datatypes_are_supported {
  static_assert(
      is_supported_tensorexpression_datatype<typename T1::type>::type::value and
          is_supported_tensorexpression_datatype<
              typename T2::type>::type::value,
      "Cannot perform binary operations between the two TensorExpressions with "
      "the given data types because at least one of the data types is not "
      "supported by TensorExpressions.");
  using type =
      typename tensorexpression_binop_datatypes_are_supported_impl<T1,
                                                                   T2>::type;
};
}  // namespace detail
}  // namespace tenex
