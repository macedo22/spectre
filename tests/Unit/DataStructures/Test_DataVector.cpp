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

namespace {
template <typename Generator>
void test_large_datavector_expression(const gsl::not_null<Generator*> generator,
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
  // const Scalar<DataVector> L = TensorExpressions::evaluate(
  //     R(ti_A, ti_B, ti_C, ti_D) * S(ti_d, ti_c, ti_b, ti_a));

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // real    1m27.180s
  // user    1m25.824s
  // sys     0m1.341s
  // const DataVector result = R.get(0, 0, 0, 0) * S.get(0, 0, 0, 0) +
  //                           R.get(0, 0, 0, 1) * S.get(1, 0, 0, 0) +
  //                           R.get(0, 0, 0, 2) * S.get(2, 0, 0, 0) +
  //                           R.get(0, 0, 0, 3) * S.get(3, 0, 0, 0) +
  //                           R.get(0, 0, 1, 0) * S.get(0, 1, 0, 0) +
  //                           R.get(0, 0, 1, 1) * S.get(1, 1, 0, 0) +
  //                           R.get(0, 0, 1, 2) * S.get(2, 1, 0, 0) +
  //                           R.get(0, 0, 1, 3) * S.get(3, 1, 0, 0) +
  //                           R.get(0, 0, 2, 0) * S.get(0, 2, 0, 0) +
  //                           R.get(0, 0, 2, 1) * S.get(1, 2, 0, 0) +
  //                           R.get(0, 0, 2, 2) * S.get(2, 2, 0, 0) +
  //                           R.get(0, 0, 2, 3) * S.get(3, 2, 0, 0) +
  //                           R.get(0, 0, 3, 0) * S.get(0, 3, 0, 0) +
  //                           R.get(0, 0, 3, 1) * S.get(1, 3, 0, 0) +
  //                           R.get(0, 0, 3, 2) * S.get(2, 3, 0, 0) +
  //                           R.get(0, 0, 3, 3) * S.get(3, 3, 0, 0) +
  //                           R.get(0, 1, 0, 0) * S.get(0, 0, 1, 0) +
  //                           R.get(0, 1, 0, 1) * S.get(1, 0, 1, 0) +
  //                           R.get(0, 1, 0, 2) * S.get(2, 0, 1, 0) +
  //                           R.get(0, 1, 0, 3) * S.get(3, 0, 1, 0) +
  //                           R.get(0, 1, 1, 0) * S.get(0, 1, 1, 0) +
  //                           R.get(0, 1, 1, 1) * S.get(1, 1, 1, 0) +
  //                           R.get(0, 1, 1, 2) * S.get(2, 1, 1, 0) +
  //                           R.get(0, 1, 1, 3) * S.get(3, 1, 1, 0) +
  //                           R.get(0, 1, 2, 0) * S.get(0, 2, 1, 0) +
  //                           R.get(0, 1, 2, 1) * S.get(1, 2, 1, 0) +
  //                           R.get(0, 1, 2, 2) * S.get(2, 2, 1, 0) +
  //                           R.get(0, 1, 2, 3) * S.get(3, 2, 1, 0) +
  //                           R.get(0, 1, 3, 0) * S.get(0, 3, 1, 0) +
  //                           R.get(0, 1, 3, 1) * S.get(1, 3, 1, 0) +
  //                           R.get(0, 1, 3, 2) * S.get(2, 3, 1, 0) +
  //                           R.get(0, 1, 3, 3) * S.get(3, 3, 1, 0) +
  //                           R.get(0, 2, 0, 0) * S.get(0, 0, 2, 0) +
  //                           R.get(0, 2, 0, 1) * S.get(1, 0, 2, 0) +
  //                           R.get(0, 2, 0, 2) * S.get(2, 0, 2, 0) +
  //                           R.get(0, 2, 0, 3) * S.get(3, 0, 2, 0) +
  //                           R.get(0, 2, 1, 0) * S.get(0, 1, 2, 0) +
  //                           R.get(0, 2, 1, 1) * S.get(1, 1, 2, 0) +
  //                           R.get(0, 2, 1, 2) * S.get(2, 1, 2, 0) +
  //                           R.get(0, 2, 1, 3) * S.get(3, 1, 2, 0) +
  //                           R.get(0, 2, 2, 0) * S.get(0, 2, 2, 0) +
  //                           R.get(0, 2, 2, 1) * S.get(1, 2, 2, 0) +
  //                           R.get(0, 2, 2, 2) * S.get(2, 2, 2, 0) +
  //                           R.get(0, 2, 2, 3) * S.get(3, 2, 2, 0) +
  //                           R.get(0, 2, 3, 0) * S.get(0, 3, 2, 0) +
  //                           R.get(0, 2, 3, 1) * S.get(1, 3, 2, 0) +
  //                           R.get(0, 2, 3, 2) * S.get(2, 3, 2, 0) +
  //                           R.get(0, 2, 3, 3) * S.get(3, 3, 2, 0) +
  //                           R.get(0, 3, 0, 0) * S.get(0, 0, 3, 0) +
  //                           R.get(0, 3, 0, 1) * S.get(1, 0, 3, 0) +
  //                           R.get(0, 3, 0, 2) * S.get(2, 0, 3, 0) +
  //                           R.get(0, 3, 0, 3) * S.get(3, 0, 3, 0) +
  //                           R.get(0, 3, 1, 0) * S.get(0, 1, 3, 0) +
  //                           R.get(0, 3, 1, 1) * S.get(1, 1, 3, 0) +
  //                           R.get(0, 3, 1, 2) * S.get(2, 1, 3, 0) +
  //                           R.get(0, 3, 1, 3) * S.get(3, 1, 3, 0) +
  //                           R.get(0, 3, 2, 0) * S.get(0, 2, 3, 0) +
  //                           R.get(0, 3, 2, 1) * S.get(1, 2, 3, 0) +
  //                           R.get(0, 3, 2, 2) * S.get(2, 2, 3, 0) +
  //                           R.get(0, 3, 2, 3) * S.get(3, 2, 3, 0) +
  //                           R.get(0, 3, 3, 0) * S.get(0, 3, 3, 0) +
  //                           R.get(0, 3, 3, 1) * S.get(1, 3, 3, 0) +
  //                           R.get(0, 3, 3, 2) * S.get(2, 3, 3, 0) +
  //                           R.get(0, 3, 3, 3) * S.get(3, 3, 3, 0) +
  //                           R.get(1, 0, 0, 0) * S.get(0, 0, 0, 1) +
  //                           R.get(1, 0, 0, 1) * S.get(1, 0, 0, 1) +
  //                           R.get(1, 0, 0, 2) * S.get(2, 0, 0, 1) +
  //                           R.get(1, 0, 0, 3) * S.get(3, 0, 0, 1) +
  //                           R.get(1, 0, 1, 0) * S.get(0, 1, 0, 1) +
  //                           R.get(1, 0, 1, 1) * S.get(1, 1, 0, 1) +
  //                           R.get(1, 0, 1, 2) * S.get(2, 1, 0, 1) +
  //                           R.get(1, 0, 1, 3) * S.get(3, 1, 0, 1) +
  //                           R.get(1, 0, 2, 0) * S.get(0, 2, 0, 1) +
  //                           R.get(1, 0, 2, 1) * S.get(1, 2, 0, 1) +
  //                           R.get(1, 0, 2, 2) * S.get(2, 2, 0, 1) +
  //                           R.get(1, 0, 2, 3) * S.get(3, 2, 0, 1) +
  //                           R.get(1, 0, 3, 0) * S.get(0, 3, 0, 1) +
  //                           R.get(1, 0, 3, 1) * S.get(1, 3, 0, 1) +
  //                           R.get(1, 0, 3, 2) * S.get(2, 3, 0, 1) +
  //                           R.get(1, 0, 3, 3) * S.get(3, 3, 0, 1) +
  //                           R.get(1, 1, 0, 0) * S.get(0, 0, 1, 1) +
  //                           R.get(1, 1, 0, 1) * S.get(1, 0, 1, 1) +
  //                           R.get(1, 1, 0, 2) * S.get(2, 0, 1, 1) +
  //                           R.get(1, 1, 0, 3) * S.get(3, 0, 1, 1) +
  //                           R.get(1, 1, 1, 0) * S.get(0, 1, 1, 1) +
  //                           R.get(1, 1, 1, 1) * S.get(1, 1, 1, 1) +
  //                           R.get(1, 1, 1, 2) * S.get(2, 1, 1, 1) +
  //                           R.get(1, 1, 1, 3) * S.get(3, 1, 1, 1) +
  //                           R.get(1, 1, 2, 0) * S.get(0, 2, 1, 1) +
  //                           R.get(1, 1, 2, 1) * S.get(1, 2, 1, 1) +
  //                           R.get(1, 1, 2, 2) * S.get(2, 2, 1, 1) +
  //                           R.get(1, 1, 2, 3) * S.get(3, 2, 1, 1) +
  //                           R.get(1, 1, 3, 0) * S.get(0, 3, 1, 1) +
  //                           R.get(1, 1, 3, 1) * S.get(1, 3, 1, 1) +
  //                           R.get(1, 1, 3, 2) * S.get(2, 3, 1, 1) +
  //                           R.get(1, 1, 3, 3) * S.get(3, 3, 1, 1) +
  //                           R.get(1, 2, 0, 0) * S.get(0, 0, 2, 1) +
  //                           R.get(1, 2, 0, 1) * S.get(1, 0, 2, 1) +
  //                           R.get(1, 2, 0, 2) * S.get(2, 0, 2, 1) +
  //                           R.get(1, 2, 0, 3) * S.get(3, 0, 2, 1) +
  //                           R.get(1, 2, 1, 0) * S.get(0, 1, 2, 1) +
  //                           R.get(1, 2, 1, 1) * S.get(1, 1, 2, 1) +
  //                           R.get(1, 2, 1, 2) * S.get(2, 1, 2, 1) +
  //                           R.get(1, 2, 1, 3) * S.get(3, 1, 2, 1) +
  //                           R.get(1, 2, 2, 0) * S.get(0, 2, 2, 1) +
  //                           R.get(1, 2, 2, 1) * S.get(1, 2, 2, 1) +
  //                           R.get(1, 2, 2, 2) * S.get(2, 2, 2, 1) +
  //                           R.get(1, 2, 2, 3) * S.get(3, 2, 2, 1) +
  //                           R.get(1, 2, 3, 0) * S.get(0, 3, 2, 1) +
  //                           R.get(1, 2, 3, 1) * S.get(1, 3, 2, 1) +
  //                           R.get(1, 2, 3, 2) * S.get(2, 3, 2, 1) +
  //                           R.get(1, 2, 3, 3) * S.get(3, 3, 2, 1) +
  //                           R.get(1, 3, 0, 0) * S.get(0, 0, 3, 1) +
  //                           R.get(1, 3, 0, 1) * S.get(1, 0, 3, 1) +
  //                           R.get(1, 3, 0, 2) * S.get(2, 0, 3, 1) +
  //                           R.get(1, 3, 0, 3) * S.get(3, 0, 3, 1) +
  //                           R.get(1, 3, 1, 0) * S.get(0, 1, 3, 1) +
  //                           R.get(1, 3, 1, 1) * S.get(1, 1, 3, 1) +
  //                           R.get(1, 3, 1, 2) * S.get(2, 1, 3, 1) +
  //                           R.get(1, 3, 1, 3) * S.get(3, 1, 3, 1) +
  //                           R.get(1, 3, 2, 0) * S.get(0, 2, 3, 1) +
  //                           R.get(1, 3, 2, 1) * S.get(1, 2, 3, 1) +
  //                           R.get(1, 3, 2, 2) * S.get(2, 2, 3, 1) +
  //                           R.get(1, 3, 2, 3) * S.get(3, 2, 3, 1) +
  //                           R.get(1, 3, 3, 0) * S.get(0, 3, 3, 1) +
  //                           R.get(1, 3, 3, 1) * S.get(1, 3, 3, 1) +
  //                           R.get(1, 3, 3, 2) * S.get(2, 3, 3, 1) +
  //                           R.get(1, 3, 3, 3) * S.get(3, 3, 3, 1) +
  //                           R.get(2, 0, 0, 0) * S.get(0, 0, 0, 2) +
  //                           R.get(2, 0, 0, 1) * S.get(1, 0, 0, 2) +
  //                           R.get(2, 0, 0, 2) * S.get(2, 0, 0, 2) +
  //                           R.get(2, 0, 0, 3) * S.get(3, 0, 0, 2) +
  //                           R.get(2, 0, 1, 0) * S.get(0, 1, 0, 2) +
  //                           R.get(2, 0, 1, 1) * S.get(1, 1, 0, 2) +
  //                           R.get(2, 0, 1, 2) * S.get(2, 1, 0, 2) +
  //                           R.get(2, 0, 1, 3) * S.get(3, 1, 0, 2) +
  //                           R.get(2, 0, 2, 0) * S.get(0, 2, 0, 2) +
  //                           R.get(2, 0, 2, 1) * S.get(1, 2, 0, 2) +
  //                           R.get(2, 0, 2, 2) * S.get(2, 2, 0, 2) +
  //                           R.get(2, 0, 2, 3) * S.get(3, 2, 0, 2) +
  //                           R.get(2, 0, 3, 0) * S.get(0, 3, 0, 2) +
  //                           R.get(2, 0, 3, 1) * S.get(1, 3, 0, 2) +
  //                           R.get(2, 0, 3, 2) * S.get(2, 3, 0, 2) +
  //                           R.get(2, 0, 3, 3) * S.get(3, 3, 0, 2) +
  //                           R.get(2, 1, 0, 0) * S.get(0, 0, 1, 2) +
  //                           R.get(2, 1, 0, 1) * S.get(1, 0, 1, 2) +
  //                           R.get(2, 1, 0, 2) * S.get(2, 0, 1, 2) +
  //                           R.get(2, 1, 0, 3) * S.get(3, 0, 1, 2) +
  //                           R.get(2, 1, 1, 0) * S.get(0, 1, 1, 2) +
  //                           R.get(2, 1, 1, 1) * S.get(1, 1, 1, 2) +
  //                           R.get(2, 1, 1, 2) * S.get(2, 1, 1, 2) +
  //                           R.get(2, 1, 1, 3) * S.get(3, 1, 1, 2) +
  //                           R.get(2, 1, 2, 0) * S.get(0, 2, 1, 2) +
  //                           R.get(2, 1, 2, 1) * S.get(1, 2, 1, 2) +
  //                           R.get(2, 1, 2, 2) * S.get(2, 2, 1, 2) +
  //                           R.get(2, 1, 2, 3) * S.get(3, 2, 1, 2) +
  //                           R.get(2, 1, 3, 0) * S.get(0, 3, 1, 2) +
  //                           R.get(2, 1, 3, 1) * S.get(1, 3, 1, 2) +
  //                           R.get(2, 1, 3, 2) * S.get(2, 3, 1, 2) +
  //                           R.get(2, 1, 3, 3) * S.get(3, 3, 1, 2) +
  //                           R.get(2, 2, 0, 0) * S.get(0, 0, 2, 2) +
  //                           R.get(2, 2, 0, 1) * S.get(1, 0, 2, 2) +
  //                           R.get(2, 2, 0, 2) * S.get(2, 0, 2, 2) +
  //                           R.get(2, 2, 0, 3) * S.get(3, 0, 2, 2) +
  //                           R.get(2, 2, 1, 0) * S.get(0, 1, 2, 2) +
  //                           R.get(2, 2, 1, 1) * S.get(1, 1, 2, 2) +
  //                           R.get(2, 2, 1, 2) * S.get(2, 1, 2, 2) +
  //                           R.get(2, 2, 1, 3) * S.get(3, 1, 2, 2) +
  //                           R.get(2, 2, 2, 0) * S.get(0, 2, 2, 2) +
  //                           R.get(2, 2, 2, 1) * S.get(1, 2, 2, 2) +
  //                           R.get(2, 2, 2, 2) * S.get(2, 2, 2, 2) +
  //                           R.get(2, 2, 2, 3) * S.get(3, 2, 2, 2) +
  //                           R.get(2, 2, 3, 0) * S.get(0, 3, 2, 2) +
  //                           R.get(2, 2, 3, 1) * S.get(1, 3, 2, 2) +
  //                           R.get(2, 2, 3, 2) * S.get(2, 3, 2, 2) +
  //                           R.get(2, 2, 3, 3) * S.get(3, 3, 2, 2) +
  //                           R.get(2, 3, 0, 0) * S.get(0, 0, 3, 2) +
  //                           R.get(2, 3, 0, 1) * S.get(1, 0, 3, 2) +
  //                           R.get(2, 3, 0, 2) * S.get(2, 0, 3, 2) +
  //                           R.get(2, 3, 0, 3) * S.get(3, 0, 3, 2) +
  //                           R.get(2, 3, 1, 0) * S.get(0, 1, 3, 2) +
  //                           R.get(2, 3, 1, 1) * S.get(1, 1, 3, 2) +
  //                           R.get(2, 3, 1, 2) * S.get(2, 1, 3, 2) +
  //                           R.get(2, 3, 1, 3) * S.get(3, 1, 3, 2) +
  //                           R.get(2, 3, 2, 0) * S.get(0, 2, 3, 2) +
  //                           R.get(2, 3, 2, 1) * S.get(1, 2, 3, 2) +
  //                           R.get(2, 3, 2, 2) * S.get(2, 2, 3, 2) +
  //                           R.get(2, 3, 2, 3) * S.get(3, 2, 3, 2) +
  //                           R.get(2, 3, 3, 0) * S.get(0, 3, 3, 2) +
  //                           R.get(2, 3, 3, 1) * S.get(1, 3, 3, 2) +
  //                           R.get(2, 3, 3, 2) * S.get(2, 3, 3, 2) +
  //                           R.get(2, 3, 3, 3) * S.get(3, 3, 3, 2) +
  //                           R.get(3, 0, 0, 0) * S.get(0, 0, 0, 3) +
  //                           R.get(3, 0, 0, 1) * S.get(1, 0, 0, 3) +
  //                           R.get(3, 0, 0, 2) * S.get(2, 0, 0, 3) +
  //                           R.get(3, 0, 0, 3) * S.get(3, 0, 0, 3) +
  //                           R.get(3, 0, 1, 0) * S.get(0, 1, 0, 3) +
  //                           R.get(3, 0, 1, 1) * S.get(1, 1, 0, 3) +
  //                           R.get(3, 0, 1, 2) * S.get(2, 1, 0, 3) +
  //                           R.get(3, 0, 1, 3) * S.get(3, 1, 0, 3) +
  //                           R.get(3, 0, 2, 0) * S.get(0, 2, 0, 3) +
  //                           R.get(3, 0, 2, 1) * S.get(1, 2, 0, 3) +
  //                           R.get(3, 0, 2, 2) * S.get(2, 2, 0, 3) +
  //                           R.get(3, 0, 2, 3) * S.get(3, 2, 0, 3) +
  //                           R.get(3, 0, 3, 0) * S.get(0, 3, 0, 3) +
  //                           R.get(3, 0, 3, 1) * S.get(1, 3, 0, 3) +
  //                           R.get(3, 0, 3, 2) * S.get(2, 3, 0, 3) +
  //                           R.get(3, 0, 3, 3) * S.get(3, 3, 0, 3) +
  //                           R.get(3, 1, 0, 0) * S.get(0, 0, 1, 3) +
  //                           R.get(3, 1, 0, 1) * S.get(1, 0, 1, 3) +
  //                           R.get(3, 1, 0, 2) * S.get(2, 0, 1, 3) +
  //                           R.get(3, 1, 0, 3) * S.get(3, 0, 1, 3) +
  //                           R.get(3, 1, 1, 0) * S.get(0, 1, 1, 3) +
  //                           R.get(3, 1, 1, 1) * S.get(1, 1, 1, 3) +
  //                           R.get(3, 1, 1, 2) * S.get(2, 1, 1, 3) +
  //                           R.get(3, 1, 1, 3) * S.get(3, 1, 1, 3) +
  //                           R.get(3, 1, 2, 0) * S.get(0, 2, 1, 3) +
  //                           R.get(3, 1, 2, 1) * S.get(1, 2, 1, 3) +
  //                           R.get(3, 1, 2, 2) * S.get(2, 2, 1, 3) +
  //                           R.get(3, 1, 2, 3) * S.get(3, 2, 1, 3) +
  //                           R.get(3, 1, 3, 0) * S.get(0, 3, 1, 3) +
  //                           R.get(3, 1, 3, 1) * S.get(1, 3, 1, 3) +
  //                           R.get(3, 1, 3, 2) * S.get(2, 3, 1, 3) +
  //                           R.get(3, 1, 3, 3) * S.get(3, 3, 1, 3) +
  //                           R.get(3, 2, 0, 0) * S.get(0, 0, 2, 3) +
  //                           R.get(3, 2, 0, 1) * S.get(1, 0, 2, 3) +
  //                           R.get(3, 2, 0, 2) * S.get(2, 0, 2, 3) +
  //                           R.get(3, 2, 0, 3) * S.get(3, 0, 2, 3) +
  //                           R.get(3, 2, 1, 0) * S.get(0, 1, 2, 3) +
  //                           R.get(3, 2, 1, 1) * S.get(1, 1, 2, 3) +
  //                           R.get(3, 2, 1, 2) * S.get(2, 1, 2, 3) +
  //                           R.get(3, 2, 1, 3) * S.get(3, 1, 2, 3) +
  //                           R.get(3, 2, 2, 0) * S.get(0, 2, 2, 3) +
  //                           R.get(3, 2, 2, 1) * S.get(1, 2, 2, 3) +
  //                           R.get(3, 2, 2, 2) * S.get(2, 2, 2, 3) +
  //                           R.get(3, 2, 2, 3) * S.get(3, 2, 2, 3) +
  //                           R.get(3, 2, 3, 0) * S.get(0, 3, 2, 3) +
  //                           R.get(3, 2, 3, 1) * S.get(1, 3, 2, 3) +
  //                           R.get(3, 2, 3, 2) * S.get(2, 3, 2, 3) +
  //                           R.get(3, 2, 3, 3) * S.get(3, 3, 2, 3) +
  //                           R.get(3, 3, 0, 0) * S.get(0, 0, 3, 3) +
  //                           R.get(3, 3, 0, 1) * S.get(1, 0, 3, 3) +
  //                           R.get(3, 3, 0, 2) * S.get(2, 0, 3, 3) +
  //                           R.get(3, 3, 0, 3) * S.get(3, 0, 3, 3) +
  //                           R.get(3, 3, 1, 0) * S.get(0, 1, 3, 3) +
  //                           R.get(3, 3, 1, 1) * S.get(1, 1, 3, 3) +
  //                           R.get(3, 3, 1, 2) * S.get(2, 1, 3, 3) +
  //                           R.get(3, 3, 1, 3) * S.get(3, 1, 3, 3) +
  //                           R.get(3, 3, 2, 0) * S.get(0, 2, 3, 3) +
  //                           R.get(3, 3, 2, 1) * S.get(1, 2, 3, 3) +
  //                           R.get(3, 3, 2, 2) * S.get(2, 2, 3, 3) +
  //                           R.get(3, 3, 2, 3) * S.get(3, 2, 3, 3) +
  //                           R.get(3, 3, 3, 0) * S.get(0, 3, 3, 3) +
  //                           R.get(3, 3, 3, 1) * S.get(1, 3, 3, 3) +
  //                           R.get(3, 3, 3, 2) * S.get(2, 3, 3, 3) +
  //                           R.get(3, 3, 3, 3) * S.get(3, 3, 3, 3);

  // CHECK_ITERABLE_APPROX(get(L), result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector", "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test_large_datavector_expression(
      make_not_null(&generator),
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
