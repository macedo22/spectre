// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <random>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename T1, typename T2>
void check_values_equal(const T1& lhs_value, const T2& rhs_value) {
  CHECK(lhs_value == rhs_value);
}

template <>
void check_values_equal<std::complex<double>, double>(
    const std::complex<double>& lhs_value, const double& rhs_value) {
  CHECK(std::imag(lhs_value) == 0.0);
  CHECK(std::real(lhs_value) == rhs_value);
}

template <>
void check_values_equal<ComplexDataVector, double>(
    const ComplexDataVector& lhs_value, const double& rhs_value) {
  for (size_t i = 0; i < lhs_value.size(); i++) {
    CHECK(std::imag(lhs_value[i]) == 0.0);
    CHECK(std::real(lhs_value[i]) == rhs_value);
  }
}

template <>
void check_values_equal<ComplexDataVector, DataVector>(
    const ComplexDataVector& lhs_value, const DataVector& rhs_value) {
  for (size_t i = 0; i < lhs_value.size(); i++) {
    CHECK(std::imag(lhs_value[i]) == 0.0);
    CHECK(std::real(lhs_value[i]) == rhs_value[i]);
  }
}

template <typename Generator, typename LhsDataType, typename RhsDataType>
void test_evaluate_assignment(const gsl::not_null<Generator*> generator,
                              const LhsDataType& used_for_size_lhs,
                              const RhsDataType& used_for_size_rhs) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  if constexpr (std::is_same_v<RhsDataType, double> or
                std::is_same_v<RhsDataType, std::complex<double>>) {
    // assign number
    const auto R1 = make_with_random_values<RhsDataType>(
        generator, distribution, used_for_size_rhs);
    Scalar<LhsDataType> L1{used_for_size_lhs};
    tenex::evaluate(make_not_null(&L1), R1);
    check_values_equal(get(L1), R1);
  }

  // assign Scalar<RhsDataType>
  const auto R2 = make_with_random_values<Scalar<RhsDataType>>(
      generator, distribution, used_for_size_rhs);
  Scalar<LhsDataType> L2{used_for_size_lhs};
  tenex::evaluate(make_not_null(&L2), R2());
  check_values_equal(get(L2), get(R2));

  // assign Tensor<RhsDataType, ...> rank > 0
  const auto R3 = make_with_random_values<tnsr::ij<RhsDataType, 3>>(
      generator, distribution, used_for_size_rhs);
  tnsr::ii<LhsDataType, 3> L3{used_for_size_lhs};
  tenex::evaluate<ti::j, ti::i>(make_not_null(&L3), R3(ti::i, ti::j));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = i; j < 3; j++) {
      check_values_equal(L3.get(j, i), R3.get(i, j));
    }
  }
}

template <typename Generator, typename LhsDataType, typename RhsDataType>
void test_evaluate_ops(const gsl::not_null<Generator*> generator,
                       const LhsDataType& used_for_size_lhs,
                       const RhsDataType& used_for_size_rhs) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R =
      make_with_random_values<tnsr::Ab<RhsDataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size_rhs);
  const auto S =
      make_with_random_values<tnsr::aa<RhsDataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size_rhs);
  const auto T = make_with_random_values<Scalar<RhsDataType>>(
      generator, distribution, used_for_size_rhs);

  // test evaluation of unary ops
  Scalar<LhsDataType> L_contraction{used_for_size_lhs};
  tenex::evaluate(make_not_null(&L_contraction), R(ti::A, ti::a));
  RhsDataType expected_L_contraction =
      make_with_value<RhsDataType>(used_for_size_rhs, 0.0);
  for (size_t a = 0; a < 4; a++) {
    expected_L_contraction += R.get(a, a);
  }
  check_values_equal(get(L_contraction), expected_L_contraction);

  tnsr::aa<LhsDataType, 3, Frame::Inertial> L_negation{used_for_size_lhs};
  tenex::evaluate<ti::b, ti::a>(make_not_null(&L_negation), -S(ti::a, ti::b));
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      check_values_equal(L_negation.get(b, a), -S.get(a, b));
    }
  }

  Scalar<LhsDataType> L_square_root{used_for_size_lhs};
  tenex::evaluate(make_not_null(&L_square_root), sqrt(T()));
  check_values_equal(get(L_square_root), sqrt(get(T)));

  // test evaluation of binary ops
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateComplex",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  const size_t vector_size = 5;

  const double used_for_size_real_double =
      std::numeric_limits<double>::signaling_NaN();
  const std::complex<double> used_for_size_complex_double =
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN());
  const DataVector used_for_size_real_datavector =
      DataVector(vector_size, std::numeric_limits<double>::signaling_NaN());
  const ComplexDataVector used_for_size_complex_datavector = ComplexDataVector(
      vector_size, std::numeric_limits<double>::signaling_NaN());

  test_evaluate_assignment(make_not_null(&generator),
                           used_for_size_complex_double,
                           used_for_size_real_double);
  test_evaluate_assignment(make_not_null(&generator),
                           used_for_size_complex_double,
                           used_for_size_complex_double);
  test_evaluate_assignment(make_not_null(&generator),
                           used_for_size_complex_datavector,
                           used_for_size_real_double);
  test_evaluate_assignment(make_not_null(&generator),
                           used_for_size_complex_datavector,
                           used_for_size_real_datavector);
  test_evaluate_assignment(make_not_null(&generator),
                           used_for_size_complex_datavector,
                           used_for_size_complex_datavector);

  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_double,
                    used_for_size_real_double);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_double,
                    used_for_size_complex_double);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_datavector,
                    used_for_size_real_double);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_datavector,
                    used_for_size_real_datavector);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_datavector,
                    used_for_size_complex_datavector);
}
