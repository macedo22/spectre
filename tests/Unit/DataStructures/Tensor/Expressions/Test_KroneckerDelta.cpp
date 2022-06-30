// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
template <typename Frame>
void test_kronecker_delta_properties() {
  // check dimension value
  static_assert(kronecker_delta<1, Frame>.dim == 1,
                "kronecker_delta<1> dimension should be 1");
  static_assert(kronecker_delta<2, Frame>.dim == 2,
                "kronecker_delta<2> dimension should be 2");
  static_assert(kronecker_delta<3, Frame>.dim == 3,
                "kronecker_delta<3> dimension should be 3");
  static_assert(kronecker_delta<4, Frame>.dim == 4,
                "kronecker_delta<4> dimension should be 4");

  // check component values

  // dimension 1
  const auto kdeltaIj = kronecker_delta<1, Frame>(ti::I, ti::j);
  CHECK(kdeltaIj.get({{0}}) == 1.0);

  // dimension 2
  const auto kdeltakJ = kronecker_delta<2, Frame>(ti::K, ti::J);
  CHECK(kdeltakJ.get({{0, 0}}) == 1.0);
  CHECK(kdeltakJ.get({{1, 0}}) == 0.0);
  CHECK(kdeltakJ.get({{0, 1}}) == 0.0);
  CHECK(kdeltakJ.get({{1, 1}}) == 1.0);

  // dimension 3 contracted
  const auto kdeltaJj = kronecker_delta<3, Frame>(ti::J, ti::j);
  CHECK(kdeltaJj.get({{}}) == 3.0);

  // dimension 4, spacetime
  const auto kdeltaAb = kronecker_delta<4, Frame>(ti::A, ti::b);
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      if (a == b) {
        CHECK(kdeltaAb.get({{a, b}}) == 1.0);
      } else {
        CHECK(kdeltaAb.get({{a, b}}) == 0.0);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.KroneckerDelta",
                  "[DataStructures][Unit]") {
  test_kronecker_delta_properties<Frame::Inertial>();
  test_kronecker_delta_properties<Frame::Grid>();
}
