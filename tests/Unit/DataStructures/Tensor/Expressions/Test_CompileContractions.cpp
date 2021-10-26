// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename Generator>
void test(const gsl::not_null<Generator*> generator,
          const DataVector& used_for_size) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto R = make_with_random_values<
      Tensor<DataVector, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>>(
      generator, distribution, used_for_size);
  const auto S = make_with_random_values<
      Tensor<DataVector, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // Rank 4 x Rank 4 inner product
  // 3D, contract over spatial dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    0m16.151s
  // user    0m15.860s
  // sys     0m0.290s
  // const Scalar<DataVector> T = TensorExpressions::evaluate(
  //     R(ti_I, ti_J, ti_K, ti_L) * S(ti_i, ti_j, ti_k, ti_l));

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    0m16.151s
  // user    0m15.860s
  // sys     0m0.290s
  const Scalar<DataVector> result = TensorExpressions::evaluate(
      R(ti_A, ti_B, ti_C, ti_D) * S(ti_d, ti_c, ti_b, ti_a));

  auto expected_result =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 4; d++) {
          get(expected_result) += R.get(a, b, c, d) * S.get(d, c, b, a);
        }
      }
    }
  }

  CHECK_ITERABLE_APPROX(expected_result, result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.CompileContractions",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator),
       DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
