// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <iostream>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename Generator, typename DataType>
void test_inner_product(const gsl::not_null<Generator*> generator,
                        const DataType& used_for_size) {
  using R_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>;

  using S_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>;

  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto R =
      make_with_random_values<R_type>(generator, distribution, used_for_size);
  const auto S =
      make_with_random_values<S_type>(generator, distribution, used_for_size);

  // Rank 4 x Rank 4 inner product
  // 3D, contract over all 4 dimensions
  //
  // Compiled with clang-10 compile_commands.json command. See
  // compile_command.txt in this directory
  //
  // Before removing inlining
  // ------------------------------------------------
  // real    5m12.886s
  // user    5m11.945s
  // sys     0m0.935s
  // ------------------------------------------------
  // After removing inlining
  // ------------------------------------------------
  // real    0m45.191s
  // user    0m44.672s
  // sys     0m0.518s
  // ------------------------------------------------
  const Scalar<DataType> L = TensorExpressions::evaluate(
      R(ti_A, ti_B, ti_C, ti_D) * S(ti_d, ti_c, ti_b, ti_a));
  std::cout << L << std::endl;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.InnerProduct",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  test_inner_product(
      make_not_null(&generator),
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
