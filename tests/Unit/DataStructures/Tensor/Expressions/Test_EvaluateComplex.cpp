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

namespace {
void check_values_equal(const std::complex<double>& lhs_value,
                        const double rhs_value) {
  CHECK(std::imag(lhs_value) == 0.0);
  CHECK(std::real(lhs_value) == rhs_value);
}

void check_values_equal(const std::complex<double>& lhs_value,
                        const std::complex<double>& rhs_value) {
  CHECK(lhs_value == rhs_value);
}

void check_values_equal(const ComplexDataVector& lhs_value,
                        const double rhs_value) {
  for (size_t i = 0; i < lhs_value.size(); i++) {
    CHECK(std::imag(lhs_value[i]) == 0.0);
    CHECK(std::real(lhs_value[i]) == rhs_value);
  }
}

void check_values_equal(const ComplexDataVector& lhs_value,
                        const DataVector& rhs_value) {
  for (size_t i = 0; i < lhs_value.size(); i++) {
    CHECK(std::imag(lhs_value[i]) == 0.0);
    CHECK(std::real(lhs_value[i]) == rhs_value[i]);
  }
}

void check_values_equal(const ComplexDataVector& lhs_value,
                        const ComplexDataVector& rhs_value) {
  CHECK(lhs_value == rhs_value);
}

template <typename Generator, typename LhsDataType, typename RhsDataType>
void test_evaluate(const gsl::not_null<Generator*> generator,
                   const LhsDataType& used_for_size_lhs,
                   const RhsDataType& used_for_size_rhs) {
  std::uniform_real_distribution<> distribution(1.0, 1.0);

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

  test_evaluate(make_not_null(&generator), used_for_size_complex_double,
                used_for_size_real_double);
  test_evaluate(make_not_null(&generator), used_for_size_complex_double,
                used_for_size_complex_double);
  test_evaluate(make_not_null(&generator), used_for_size_complex_datavector,
                used_for_size_real_double);
//   test_evaluate(make_not_null(&generator), used_for_size_complex_datavector,
//                 used_for_size_real_datavector);
  test_evaluate(make_not_null(&generator), used_for_size_complex_datavector,
                used_for_size_complex_datavector);
}
