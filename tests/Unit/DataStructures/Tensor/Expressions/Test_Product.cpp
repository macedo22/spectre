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

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct2By2",
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
    auto L_abcd = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t,
    ti_d_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_abdc = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_d_t, ti_c_t>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_acbd = TensorExpressions::evaluate<ti_a_t, ti_c_t, ti_b_t,
    ti_d_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_acdb = TensorExpressions::evaluate<ti_a_t, ti_c_t, ti_d_t,
    ti_b_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_adbc = TensorExpressions::evaluate<ti_a_t, ti_d_t, ti_b_t,
    ti_c_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_adcb = TensorExpressions::evaluate<ti_a_t, ti_d_t, ti_c_t,
    ti_b_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

    auto L_bacd = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_c_t,
    ti_d_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_badc = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_d_t,
    ti_c_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_bcad = TensorExpressions::evaluate<ti_b_t, ti_c_t, ti_a_t,
    ti_d_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_bcda = TensorExpressions::evaluate<ti_b_t, ti_c_t, ti_d_t,
    ti_a_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_bdac = TensorExpressions::evaluate<ti_b_t, ti_d_t, ti_a_t,
    ti_c_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_bdca = TensorExpressions::evaluate<ti_b_t, ti_d_t, ti_c_t,
    ti_a_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  auto L_cabd = TensorExpressions::evaluate<ti_c_t, ti_a_t, ti_b_t,
    ti_d_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_cadb = TensorExpressions::evaluate<ti_c_t, ti_a_t, ti_d_t, ti_b_t>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_cbad = TensorExpressions::evaluate<ti_c_t, ti_b_t, ti_a_t,
    ti_d_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_cbda = TensorExpressions::evaluate<ti_c_t, ti_b_t, ti_d_t,
    ti_a_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_cdab = TensorExpressions::evaluate<ti_c_t, ti_d_t, ti_a_t,
    ti_b_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_cdba = TensorExpressions::evaluate<ti_c_t, ti_d_t, ti_b_t,
    ti_a_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  auto L_dabc = TensorExpressions::evaluate<ti_d_t, ti_a_t, ti_b_t,
    ti_c_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_dacb = TensorExpressions::evaluate<ti_d_t, ti_a_t, ti_c_t, ti_b_t>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_dbac = TensorExpressions::evaluate<ti_d_t, ti_b_t, ti_a_t,
    ti_c_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_dbca = TensorExpressions::evaluate<ti_d_t, ti_b_t, ti_c_t,
    ti_a_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_dcab = TensorExpressions::evaluate<ti_d_t, ti_c_t, ti_a_t,
    ti_b_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
    auto L_dcba = TensorExpressions::evaluate<ti_d_t, ti_c_t, ti_b_t,
    ti_a_t>(
        Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 4; d++) {
            // std::array<size_t, 4> index = {a, b, d, c};
            // std::cout << "Current index: " << index << std::endl;
            // std::cout << "R(a, b) : " << Rll.get(a, b) << std::endl;
            // std::cout << "S(d, c) : " << Sll.get(d, c) << std::endl;

            CHECK(L_abcd.get(a, b, c, d) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_abdc.get(a, b, d, c) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_acbd.get(a, c, b, d) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_acdb.get(a, c, d, b) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_adbc.get(a, d, b, c) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_adcb.get(a, d, c, b) == Rll.get(a, b) * Sll.get(c, d));

            CHECK(L_bacd.get(b, a, c, d) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_badc.get(b, a, d, c) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_bcad.get(b, c, a, d) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_bcda.get(b, c, d, a) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_bdac.get(b, d, a, c) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_bdca.get(b, d, c, a) == Rll.get(a, b) * Sll.get(c, d));

            CHECK(L_cabd.get(c, a, b, d) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_cadb.get(c, a, d, b) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_cbad.get(c, b, a, d) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_cbda.get(c, b, d, a) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_cdab.get(c, d, a, b) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_cdba.get(c, d, b, a) == Rll.get(a, b) * Sll.get(c, d));

            CHECK(L_dabc.get(d, a, b, c) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_dacb.get(d, a, c, b) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_dbac.get(d, b, a, c) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_dbca.get(d, b, c, a) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_dcab.get(d, c, a, b) == Rll.get(a, b) * Sll.get(c, d));
            CHECK(L_dcba.get(d, c, b, a) == Rll.get(a, b) * Sll.get(c, d));
        }
      }
    }
  }
}
