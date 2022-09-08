// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ArbitraryType {};

template <typename ValueType>
using number_expression = tenex::NumberAsExpression<ValueType>;
template <typename ValueType>
using tensor_expression =
    tenex::TensorAsExpression<Scalar<ValueType>, tmpl::list<>>;

template <typename T>
void test_is_real_number(const bool real_number_expected) {
  CHECK(tenex::detail::is_real_number<T>::value == real_number_expected);
}

template <typename T>
void test_is_complex_number(const bool complex_number_expected) {
  CHECK(tenex::detail::is_complex_number<T>::value == complex_number_expected);
}

template <typename T>
void test_is_real_vector(const bool real_vector_expected) {
  CHECK(tenex::detail::is_real_vector<T>::value == real_vector_expected);
}

template <typename T>
void test_is_complex_vector(const bool complex_vector_expected) {
  CHECK(tenex::detail::is_complex_vector<T>::value == complex_vector_expected);
}

template <typename T, typename Expected>
void test_upcast_if_derived_vector_type() {
  static_assert(
      std::is_same_v<
          typename tenex::detail::upcast_if_derived_vector_type<T>::type,
          Expected>,
      "Test for tenex::detail::upcast_if_derived_vector_type failed.");
}

template <typename T>
void test_is_supported_tensorexpression_datatype(const bool support_expected) {
  CHECK(tenex::detail::is_supported_tensorexpression_datatype<T>::type::value ==
        support_expected);
}

template <typename LhsDataType, typename RhsDataType>
void test_lhs_datatype_is_assignable_to_rhs_datatype(
    const bool support_expected) {
  CHECK(tenex::detail::lhs_datatype_is_assignable_to_rhs_datatype_impl<
            typename tenex::detail::upcast_if_derived_vector_type<
                LhsDataType>::type,
            typename tenex::detail::upcast_if_derived_vector_type<
                RhsDataType>::type>::type::value == support_expected);
}

template <typename X1, typename X2>
void test_binop_datatypes_are_supported(const bool support_expected) {
  CHECK(tenex::detail::binop_datatypes_are_supported<X1, X2>::type::value ==
        support_expected);
}

template <typename X1, typename X2, typename ExpectedBinOpDataType>
void test_get_binop_datatype() {
  CHECK(std::is_same_v<
        typename tenex::detail::get_binop_datatype_impl<
            typename tenex::detail::upcast_if_derived_vector_type<X1>::type,
            typename tenex::detail::upcast_if_derived_vector_type<X2>::type>::
            type,
        ExpectedBinOpDataType>);
}

template <typename X1, typename X2>
void test_tensor_binop_datatypes_are_supported(const bool support_expected) {
  CHECK(tenex::detail::tensor_binop_datatypes_are_supported_impl<
            X1, X2>::type::value == support_expected);
}

template <typename T1, typename T2>
void test_tensorexpression_binop_datatypes_are_supported(
    const bool support_expected) {
  CHECK(tenex::detail::tensorexpression_binop_datatypes_are_supported_impl<
            T1, T2>::type::value == support_expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.DataTypeSupport",
                  "[DataStructures][Unit]") {
  // Test
  test_is_real_number<double>(true);
  test_is_real_number<int>(true);
  test_is_real_number<float>(true);
  test_is_real_number<std::complex<double>>(false);
  test_is_real_number<std::complex<int>>(false);
  test_is_real_number<std::complex<float>>(false);
  test_is_real_number<DataVector>(false);
  test_is_real_number<ComplexDataVector>(false);
  test_is_real_number<ModalVector>(false);
  test_is_real_number<ComplexModalVector>(false);
  test_is_real_number<std::string>(false);
  test_is_real_number<ArbitraryType>(false);

  test_is_complex_number<double>(false);
  test_is_complex_number<int>(false);
  test_is_complex_number<float>(false);
  test_is_complex_number<std::complex<double>>(true);
  test_is_complex_number<std::complex<int>>(true);
  test_is_complex_number<std::complex<float>>(true);
  test_is_complex_number<DataVector>(false);
  test_is_complex_number<ComplexDataVector>(false);
  test_is_complex_number<ModalVector>(false);
  test_is_complex_number<ComplexModalVector>(false);
  test_is_complex_number<std::string>(false);
  test_is_complex_number<ArbitraryType>(false);

  test_is_real_vector<double>(false);
  test_is_real_vector<int>(false);
  test_is_real_vector<float>(false);
  test_is_real_vector<std::complex<double>>(false);
  test_is_real_vector<std::complex<int>>(false);
  test_is_real_vector<std::complex<float>>(false);
  test_is_real_vector<DataVector>(true);
  test_is_real_vector<ComplexDataVector>(false);
  test_is_real_vector<ModalVector>(true);
  test_is_real_vector<ComplexModalVector>(false);
  test_is_real_vector<std::string>(false);
  test_is_real_vector<ArbitraryType>(false);

  test_is_complex_vector<double>(false);
  test_is_complex_vector<int>(false);
  test_is_complex_vector<float>(false);
  test_is_complex_vector<std::complex<double>>(false);
  test_is_complex_vector<std::complex<int>>(false);
  test_is_complex_vector<std::complex<float>>(false);
  test_is_complex_vector<DataVector>(false);
  test_is_complex_vector<ComplexDataVector>(true);
  test_is_complex_vector<ModalVector>(false);
  test_is_complex_vector<ComplexModalVector>(true);
  test_is_complex_vector<std::string>(false);
  test_is_complex_vector<ArbitraryType>(false);

  test_upcast_if_derived_vector_type<double, double>();
  test_upcast_if_derived_vector_type<int, int>();
  test_upcast_if_derived_vector_type<float, float>();
  test_upcast_if_derived_vector_type<std::complex<double>,
                                     std::complex<double>>();
  test_upcast_if_derived_vector_type<std::complex<int>, std::complex<int>>();
  test_upcast_if_derived_vector_type<std::complex<float>,
                                     std::complex<float>>();
  test_upcast_if_derived_vector_type<DataVector,
                                     VectorImpl<double, DataVector>>();
  test_upcast_if_derived_vector_type<VectorImpl<double, DataVector>,
                                     VectorImpl<double, DataVector>>();
  test_upcast_if_derived_vector_type<
      ComplexDataVector, VectorImpl<std::complex<double>, ComplexDataVector>>();
  test_upcast_if_derived_vector_type<
      VectorImpl<std::complex<double>, ComplexDataVector>,
      VectorImpl<std::complex<double>, ComplexDataVector>>();
  test_upcast_if_derived_vector_type<ModalVector,
                                     VectorImpl<double, ModalVector>>();
  test_upcast_if_derived_vector_type<VectorImpl<double, ModalVector>,
                                     VectorImpl<double, ModalVector>>();
  test_upcast_if_derived_vector_type<
      ComplexModalVector,
      VectorImpl<std::complex<double>, ComplexModalVector>>();
  test_upcast_if_derived_vector_type<
      VectorImpl<std::complex<double>, ComplexModalVector>,
      VectorImpl<std::complex<double>, ComplexModalVector>>();
  test_upcast_if_derived_vector_type<std::string, std::string>();
  test_upcast_if_derived_vector_type<ArbitraryType, ArbitraryType>();

  //   CHECK(not tenex::detail::is_vector<double>::value);
  //   CHECK(tenex::detail::is_vector<DataVector>::value);
  //   CHECK(tenex::detail::is_vector<VectorImpl<double, DataVector>>::value);

  // Test is_supported_tensorexpression_datatype
  test_is_supported_tensorexpression_datatype<double>(true);
  test_is_supported_tensorexpression_datatype<std::complex<double>>(true);
  test_is_supported_tensorexpression_datatype<DataVector>(true);
  test_is_supported_tensorexpression_datatype<ComplexDataVector>(true);
  test_is_supported_tensorexpression_datatype<ArbitraryType>(false);

  // Test lhs_datatype_is_assignable_to_rhs_datatype
  test_lhs_datatype_is_assignable_to_rhs_datatype<double, double>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<double, std::complex<double>>(
      false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<double, DataVector>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<double, ComplexDataVector>(
      false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<double, ArbitraryType>(false);

  test_lhs_datatype_is_assignable_to_rhs_datatype<std::complex<double>, double>(
      true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<std::complex<double>,
                                                  std::complex<double>>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<std::complex<double>,
                                                  DataVector>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<std::complex<double>,
                                                  ComplexDataVector>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<std::complex<double>,
                                                  ArbitraryType>(false);

  test_lhs_datatype_is_assignable_to_rhs_datatype<DataVector, double>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<DataVector,
                                                  std::complex<double>>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<DataVector, DataVector>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<DataVector,
                                                  ComplexDataVector>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<DataVector, ArbitraryType>(
      false);

  test_lhs_datatype_is_assignable_to_rhs_datatype<ComplexDataVector, double>(
      true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ComplexDataVector,
                                                  std::complex<double>>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ComplexDataVector,
                                                  DataVector>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ComplexDataVector,
                                                  ComplexDataVector>(true);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ComplexDataVector,
                                                  ArbitraryType>(false);

  test_lhs_datatype_is_assignable_to_rhs_datatype<ArbitraryType, double>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ArbitraryType,
                                                  std::complex<double>>(false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ArbitraryType, DataVector>(
      false);
  test_lhs_datatype_is_assignable_to_rhs_datatype<ArbitraryType,
                                                  ComplexDataVector>(false);
  // true because lhs_datatype_is_assignable_to_rhs_datatype_impl does not check
  // if the types are supported types. That check is instead done by its caller,
  // lhs_datatype_is_assignable_to_rhs_datatype.
  test_lhs_datatype_is_assignable_to_rhs_datatype<ArbitraryType, ArbitraryType>(
      true);

  // Test binop_datatypes_are_supported
  test_binop_datatypes_are_supported<double, double>(true);
  test_binop_datatypes_are_supported<double, std::complex<double>>(true);
  test_binop_datatypes_are_supported<double, DataVector>(true);
  test_binop_datatypes_are_supported<double, ComplexDataVector>(true);
  test_binop_datatypes_are_supported<double, ArbitraryType>(false);

  test_binop_datatypes_are_supported<std::complex<double>, double>(true);
  test_binop_datatypes_are_supported<std::complex<double>,
                                     std::complex<double>>(true);
  test_binop_datatypes_are_supported<std::complex<double>, DataVector>(false);
  test_binop_datatypes_are_supported<std::complex<double>, ComplexDataVector>(
      true);
  test_binop_datatypes_are_supported<std::complex<double>, ArbitraryType>(
      false);

  test_binop_datatypes_are_supported<DataVector, double>(true);
  test_binop_datatypes_are_supported<DataVector, std::complex<double>>(false);
  test_binop_datatypes_are_supported<DataVector, DataVector>(true);
  test_binop_datatypes_are_supported<DataVector, ComplexDataVector>(true);
  test_binop_datatypes_are_supported<DataVector, ArbitraryType>(false);

  test_binop_datatypes_are_supported<ComplexDataVector, double>(true);
  test_binop_datatypes_are_supported<ComplexDataVector, std::complex<double>>(
      true);
  test_binop_datatypes_are_supported<ComplexDataVector, DataVector>(true);
  test_binop_datatypes_are_supported<ComplexDataVector, ComplexDataVector>(
      true);
  test_binop_datatypes_are_supported<ComplexDataVector, ArbitraryType>(false);

  test_binop_datatypes_are_supported<ArbitraryType, double>(false);
  test_binop_datatypes_are_supported<ArbitraryType, std::complex<double>>(
      false);
  test_binop_datatypes_are_supported<ArbitraryType, DataVector>(false);
  test_binop_datatypes_are_supported<ArbitraryType, ComplexDataVector>(false);
  // true because binop_datatypes_are_supported_impl does not check if the types
  // are supported types. That check is instead done by its caller,
  // binop_datatypes_are_supported.
  test_binop_datatypes_are_supported<ArbitraryType, ArbitraryType>(true);

  // Test get_binop_datatype
  test_get_binop_datatype<double, double, double>();
  test_get_binop_datatype<double, std::complex<double>, std::complex<double>>();
  test_get_binop_datatype<double, DataVector, DataVector>();
  test_get_binop_datatype<double, ComplexDataVector, ComplexDataVector>();
  test_get_binop_datatype<double, ArbitraryType, std::bool_constant<false>>();

  test_get_binop_datatype<std::complex<double>, double, std::complex<double>>();
  test_get_binop_datatype<std::complex<double>, std::complex<double>,
                          std::complex<double>>();
  test_get_binop_datatype<std::complex<double>, DataVector,
                          std::bool_constant<false>>();
  test_get_binop_datatype<std::complex<double>, ComplexDataVector,
                          ComplexDataVector>();
  test_get_binop_datatype<std::complex<double>, ArbitraryType,
                          std::bool_constant<false>>();

  test_get_binop_datatype<DataVector, double, DataVector>();
  test_get_binop_datatype<DataVector, std::complex<double>,
                          std::bool_constant<false>>();
  test_get_binop_datatype<DataVector, DataVector, DataVector>();
  test_get_binop_datatype<DataVector, ComplexDataVector, ComplexDataVector>();
  test_get_binop_datatype<DataVector, ArbitraryType,
                          std::bool_constant<false>>();

  test_get_binop_datatype<ComplexDataVector, double, ComplexDataVector>();
  test_get_binop_datatype<ComplexDataVector, std::complex<double>,
                          ComplexDataVector>();
  test_get_binop_datatype<ComplexDataVector, DataVector, ComplexDataVector>();
  test_get_binop_datatype<ComplexDataVector, ComplexDataVector,
                          ComplexDataVector>();
  test_get_binop_datatype<ComplexDataVector, ArbitraryType,
                          std::bool_constant<false>>();

  test_get_binop_datatype<ArbitraryType, double, std::bool_constant<false>>();
  test_get_binop_datatype<ArbitraryType, std::complex<double>,
                          std::bool_constant<false>>();
  test_get_binop_datatype<ArbitraryType, DataVector,
                          std::bool_constant<false>>();
  test_get_binop_datatype<ArbitraryType, ComplexDataVector,
                          std::bool_constant<false>>();
  // ArbitraryType is result because get_binop_datatype_impl does not check if
  // the types are supported types. That check is instead done by its caller,
  // get_binop_datatype.
  test_get_binop_datatype<ArbitraryType, ArbitraryType, ArbitraryType>();

  // Test tensor_binop_datatypes_are_supported
  test_tensor_binop_datatypes_are_supported<double, double>(true);
  test_tensor_binop_datatypes_are_supported<double, std::complex<double>>(true);
  test_tensor_binop_datatypes_are_supported<double, DataVector>(false);
  test_tensor_binop_datatypes_are_supported<double, ComplexDataVector>(false);
  test_tensor_binop_datatypes_are_supported<double, ArbitraryType>(false);

  test_tensor_binop_datatypes_are_supported<std::complex<double>, double>(true);
  test_tensor_binop_datatypes_are_supported<std::complex<double>,
                                            std::complex<double>>(true);
  test_tensor_binop_datatypes_are_supported<std::complex<double>, DataVector>(
      false);
  test_tensor_binop_datatypes_are_supported<std::complex<double>,
                                            ComplexDataVector>(false);
  test_tensor_binop_datatypes_are_supported<std::complex<double>,
                                            ArbitraryType>(false);

  test_tensor_binop_datatypes_are_supported<DataVector, double>(false);
  test_tensor_binop_datatypes_are_supported<DataVector, std::complex<double>>(
      false);
  test_tensor_binop_datatypes_are_supported<DataVector, DataVector>(true);
  test_tensor_binop_datatypes_are_supported<DataVector, ComplexDataVector>(
      true);
  test_tensor_binop_datatypes_are_supported<DataVector, ArbitraryType>(false);

  test_tensor_binop_datatypes_are_supported<ComplexDataVector, double>(false);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector,
                                            std::complex<double>>(false);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector, DataVector>(
      true);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector,
                                            ComplexDataVector>(true);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector, ArbitraryType>(
      false);

  test_tensor_binop_datatypes_are_supported<ArbitraryType, double>(false);
  test_tensor_binop_datatypes_are_supported<ArbitraryType,
                                            std::complex<double>>(false);
  test_tensor_binop_datatypes_are_supported<ArbitraryType, DataVector>(false);
  test_tensor_binop_datatypes_are_supported<ArbitraryType, ComplexDataVector>(
      false);
  // true because tensor_binop_datatypes_are_supported_impl does not check if
  // the types are supported types. That check is instead done by its caller,
  // tensor_binop_datatypes_are_supported.
  test_tensor_binop_datatypes_are_supported<ArbitraryType, ArbitraryType>(true);

  // Test tensorexpression_binop_datatypes_are_supported
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<DataVector>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<ComplexDataVector>>(true);

    test_tensorexpression_binop_datatypes_are_supported<
        number_expression<std::complex<double>>, tensor_expression<double>>(
        false);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>,
      tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>, tensor_expression<DataVector>>(
      false);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>,
      tensor_expression<ComplexDataVector>>(true);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, number_expression<double>>(true);
    test_tensorexpression_binop_datatypes_are_supported<
        tensor_expression<double>, number_expression<std::complex<double>>>(
        false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<DataVector>>(false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<ComplexDataVector>>(false);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>, number_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>,
      number_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>,
      tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>, tensor_expression<DataVector>>(
      false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>,
      tensor_expression<ComplexDataVector>>(false);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, number_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, number_expression<std::complex<double>>>(
      false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<double>>(false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<std::complex<double>>>(
      false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<DataVector>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<ComplexDataVector>>(
      true);
}
