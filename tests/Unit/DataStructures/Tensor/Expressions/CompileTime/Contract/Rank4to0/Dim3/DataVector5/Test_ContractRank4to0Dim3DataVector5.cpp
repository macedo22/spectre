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

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>>(
      generator, make_not_null(&distribution), used_for_size);

  Scalar<DataType> L(used_for_size);

  TensorExpressions::evaluate(make_not_null(&L), R(ti_B, ti_a, ti_b, ti_A));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.CompileTime.Contract.Rank4to0Dim3DataVector5",
    "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test<3>(make_not_null(&generator),
          DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
