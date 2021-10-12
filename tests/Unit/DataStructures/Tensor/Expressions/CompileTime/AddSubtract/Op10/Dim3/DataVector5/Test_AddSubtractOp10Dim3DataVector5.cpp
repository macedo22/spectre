// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename Generator, typename DataType>
void test(const gsl::not_null<Generator*> generator,
          const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  const auto R =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto S =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto T =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto A =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto B =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto C =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto D =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto E =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto F =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto G =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto H =
      make_with_random_values<tnsr::ab<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  tnsr::ab<DataType, Dim, Frame::Inertial> L(used_for_size);

  TensorExpressions::evaluate<ti_a, ti_b>(
      make_not_null(&L), R(ti_a, ti_b) + S(ti_a, ti_b) + T(ti_a, ti_b) +
                             A(ti_a, ti_b) + B(ti_a, ti_b) + C(ti_a, ti_b) +
                             D(ti_a, ti_b) + E(ti_a, ti_b) + F(ti_a, ti_b) +
                             G(ti_a, ti_b) + H(ti_a, ti_b));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.CompileTime.AddSubtract.Op10Dim3DataVector5",
    "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test<3>(make_not_null(&generator),
          DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
