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
      make_with_random_values<tnsr::AbC<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  const auto S =
      make_with_random_values<tnsr::Abc<DataType, Dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  Scalar<DataType> L(used_for_size);

  TensorExpressions::evaluate(make_not_null(&L),
                              R(ti_A, ti_b, ti_C) * S(ti_B, ti_c, ti_a));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.CompileTime.InnerProduct.3x3to0Dim3DataVector5",
    "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test<3>(make_not_null(&generator),
          DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
