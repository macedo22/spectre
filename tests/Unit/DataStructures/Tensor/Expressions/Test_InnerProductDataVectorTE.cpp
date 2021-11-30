// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.InnerProductDataVectorTE",
    "[Unit][DataStructures]") {
  // User runtime using $ time ./bin/RunSingleTest
  // 2 x 2 spatial : 0m0.024s
  // 3 x 3 spatial : 0m0.038s
  // 3 (+ 1) x 3 (+ 1) spacetime : 0m0.190s

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  using DataType = DataVector;
  constexpr size_t Dim = 3;
  using FrameType = Frame::Inertial;
  using R_type = Tensor<DataType, Symmetry<4, 3, 2, 1>,
                        index_list<SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                   SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                   SpacetimeIndex<Dim, UpLo::Up, FrameType>,
                                   SpacetimeIndex<Dim, UpLo::Up, FrameType>>>;
  using S_type = Tensor<DataType, Symmetry<4, 3, 2, 1>,
                        index_list<SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                   SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                   SpacetimeIndex<Dim, UpLo::Lo, FrameType>,
                                   SpacetimeIndex<Dim, UpLo::Lo, FrameType>>>;

  constexpr size_t num_points = 1000;
  const DataVector used_for_size =
      DataVector(num_points, std::numeric_limits<double>::signaling_NaN());

  const auto R = make_with_random_values<R_type>(make_not_null(&generator),
                                                 distribution, used_for_size);
  const auto S = make_with_random_values<S_type>(make_not_null(&generator),
                                                 distribution, used_for_size);

  Scalar<DataType> L{};

  TensorExpressions::evaluate(
      make_not_null(&L), R(ti_A, ti_B, ti_C, ti_D) * S(ti_a, ti_b, ti_c, ti_d));
}
