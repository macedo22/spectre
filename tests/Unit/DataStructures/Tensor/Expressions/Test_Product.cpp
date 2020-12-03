// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Product",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sll{};
  std::iota(Sll.begin(), Sll.end(), 0.0);
  //   auto L_abcd = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_bacd = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_c_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_abdc = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_d_t, ti_c_t>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_badc = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_d_t,
  //   ti_c_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 4; d++) {
          //   CHECK(L_abcd.get(a, b, c, d) == Rll.get(a, b) * Sll.get(c, d));
          //   CHECK(L_bacd.get(b, a, c, d) == Rll.get(a, b) * Sll.get(c, d));
          // std::array<size_t, 4> index = {a, b, d, c};
          // std::cout << "Current index: " << index << std::endl;
          // std::cout << "R(a, b) : " << Rll.get(a, b) << std::endl;
          // std::cout << "S(d, c) : " << Sll.get(d, c) << std::endl;
          //   CHECK(L_abdc.get(a, b, d, c) == Rll.get(a, b) * Sll.get(c, d));
          // CHECK(L_badc.get(b, a, d, c) == Rll.get(a, b) * Sll.get(c, d));
        }
      }
    }
  }
}
