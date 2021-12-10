// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using DataType = DataVector;
constexpr size_t Dim = 1;
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

constexpr size_t num_points = 8;
const DataVector used_for_size =
    DataVector(num_points, std::numeric_limits<double>::signaling_NaN());
}  // namespace

namespace TestNamespace {
void test_function(gsl::not_null<Scalar<DataType>*> L, const R_type& R,
                   const S_type& S) {
  get(*L) = 0.0;
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t c = 0; c < Dim + 1; c++) {
        for (size_t d = 0; d < Dim + 1; d++) {
          get(*L) += R.get(a, b, c, d) * S.get(a, b, c, d);
        }
      }
    }
  }
}
}  // namespace TestNamespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.InnerProduct4x4DataVectorLoop",
    "[Unit][DataStructures]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R = make_with_random_values<R_type>(make_not_null(&generator),
                                                 distribution, used_for_size);
  const auto S = make_with_random_values<S_type>(make_not_null(&generator),
                                                 distribution, used_for_size);

  Scalar<DataType> L(used_for_size);
  TestNamespace::test_function(make_not_null(&L), R, S);
}
