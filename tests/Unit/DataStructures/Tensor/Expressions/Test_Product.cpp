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

template <class... T>
struct td;

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct1By1",
                  "[DataStructures][Unit]") {
  // Tensor<double, Symmetry<1>,
  //        index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
  //     Ru{};
  // std::iota(Ru.begin(), Ru.end(), 0.0);
  // Tensor<double, Symmetry<1>,
  //        index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
  //     Su{};
  // std::iota(Su.begin(), Su.end(), 0.0);
  // auto L_ab = TensorExpressions::evaluate<ti_A_t, ti_B_t>(Ru(ti_A) *
  // Su(ti_B));

  // for (size_t a = 0; a < 4; a++) {
  //   for (size_t b = 0; b < 4; b++) {
  //     CHECK(L_ab.get(a, b) == Ru.get(a) * Su.get(b));
  //   }
  // }

  // Tensor<double, Symmetry<1>,
  //        index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
  //     Sl{};
  // std::iota(Sl.begin(), Sl.end(), 0.0);

  // // auto L_expr = Ru(ti_A) * Sl(ti_a);
  // // td<decltype(L_expr)::args_list>idk1;
  // // td<decltype(L_expr)>idk2;

  // // inner product
  // auto L = TensorExpressions::evaluate(Ru(ti_A) * Sl(ti_a));

  // double expected_sum = 0.0;
  // for (size_t a = 0; a < 4; a++) {
  //   expected_sum += (Ru.get(a) * Su.get(a));
  // }
  // CHECK(L.get() == expected_sum);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct2By2",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sll{};
  std::iota(Sll.begin(), Sll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ruu{};
  std::iota(Ruu.begin(), Ruu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Suu{};
  std::iota(Suu.begin(), Suu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Rlu{};
  std::iota(Rlu.begin(), Rlu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Slu{};
  std::iota(Slu.begin(), Slu.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rul{};
  std::iota(Rul.begin(), Rul.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sul{};
  std::iota(Sul.begin(), Sul.end(), 0.0);
  auto L = TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_A, ti_B));

  double expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      expected_sum += (Rll.get(a, b) * Suu.get(a, b));
    }
  }
  CHECK(L.get() == expected_sum);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct2By2",
                  "[DataStructures][Unit]") {
  // Tensor<double, Symmetry<1, 1>,
  //        index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
  //                   SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
  //     Rll{};
  // std::iota(Rll.begin(), Rll.end(), 0.0);
  // Tensor<double, Symmetry<2, 1>,
  //        index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
  //                   SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
  //     Sll{};
  // std::iota(Sll.begin(), Sll.end(), 0.0);
  //   auto L_abcd = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_abdc = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_d_t,
  //   ti_c_t>(
  //     Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_acbd = TensorExpressions::evaluate<ti_a_t, ti_c_t, ti_b_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_acdb = TensorExpressions::evaluate<ti_a_t, ti_c_t, ti_d_t,
  //   ti_b_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_adbc = TensorExpressions::evaluate<ti_a_t, ti_d_t, ti_b_t,
  //   ti_c_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_adcb = TensorExpressions::evaluate<ti_a_t, ti_d_t, ti_c_t,
  //   ti_b_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  //   auto L_bacd = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_c_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_badc = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_d_t,
  //   ti_c_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_bcad = TensorExpressions::evaluate<ti_b_t, ti_c_t, ti_a_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_bcda = TensorExpressions::evaluate<ti_b_t, ti_c_t, ti_d_t,
  //   ti_a_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_bdac = TensorExpressions::evaluate<ti_b_t, ti_d_t, ti_a_t,
  //   ti_c_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_bdca = TensorExpressions::evaluate<ti_b_t, ti_d_t, ti_c_t,
  //   ti_a_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  // auto L_cabd = TensorExpressions::evaluate<ti_c_t, ti_a_t, ti_b_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_cadb = TensorExpressions::evaluate<ti_c_t, ti_a_t, ti_d_t,
  //   ti_b_t>(
  //     Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_cbad = TensorExpressions::evaluate<ti_c_t, ti_b_t, ti_a_t,
  //   ti_d_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_cbda = TensorExpressions::evaluate<ti_c_t, ti_b_t, ti_d_t,
  //   ti_a_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_cdab = TensorExpressions::evaluate<ti_c_t, ti_d_t, ti_a_t,
  //   ti_b_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_cdba = TensorExpressions::evaluate<ti_c_t, ti_d_t, ti_b_t,
  //   ti_a_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  // auto L_dabc = TensorExpressions::evaluate<ti_d_t, ti_a_t, ti_b_t,
  //   ti_c_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_dacb = TensorExpressions::evaluate<ti_d_t, ti_a_t, ti_c_t,
  //   ti_b_t>(
  //     Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_dbac = TensorExpressions::evaluate<ti_d_t, ti_b_t, ti_a_t,
  //   ti_c_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_dbca = TensorExpressions::evaluate<ti_d_t, ti_b_t, ti_c_t,
  //   ti_a_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_dcab = TensorExpressions::evaluate<ti_d_t, ti_c_t, ti_a_t,
  //   ti_b_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  //   auto L_dcba = TensorExpressions::evaluate<ti_d_t, ti_c_t, ti_b_t,
  //   ti_a_t>(
  //       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  // for (size_t a = 0; a < 4; a++) {
  //   for (size_t b = 0; b < 4; b++) {
  //     for (size_t c = 0; c < 4; c++) {
  //       for (size_t d = 0; d < 4; d++) {
  //           // std::array<size_t, 4> index = {a, b, d, c};
  //           // std::cout << "Current index: " << index << std::endl;
  //           // std::cout << "R(a, b) : " << Rll.get(a, b) << std::endl;
  //           // std::cout << "S(d, c) : " << Sll.get(d, c) << std::endl;

  //           CHECK(L_abcd.get(a, b, c, d) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_abdc.get(a, b, d, c) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_acbd.get(a, c, b, d) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_acdb.get(a, c, d, b) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_adbc.get(a, d, b, c) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_adcb.get(a, d, c, b) == Rll.get(a, b) * Sll.get(c, d));

  //           CHECK(L_bacd.get(b, a, c, d) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_badc.get(b, a, d, c) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_bcad.get(b, c, a, d) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_bcda.get(b, c, d, a) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_bdac.get(b, d, a, c) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_bdca.get(b, d, c, a) == Rll.get(a, b) * Sll.get(c, d));

  //           CHECK(L_cabd.get(c, a, b, d) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_cadb.get(c, a, d, b) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_cbad.get(c, b, a, d) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_cbda.get(c, b, d, a) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_cdab.get(c, d, a, b) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_cdba.get(c, d, b, a) == Rll.get(a, b) * Sll.get(c, d));

  //           CHECK(L_dabc.get(d, a, b, c) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_dacb.get(d, a, c, b) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_dbac.get(d, b, a, c) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_dbca.get(d, b, c, a) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_dcab.get(d, c, a, b) == Rll.get(a, b) * Sll.get(c, d));
  //           CHECK(L_dcba.get(d, c, b, a) == Rll.get(a, b) * Sll.get(c, d));
  //       }
  //     }
  //   }
  // }
}
