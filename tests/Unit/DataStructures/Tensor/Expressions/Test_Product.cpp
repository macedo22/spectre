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

// SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct1By1",
//                   "[DataStructures][Unit]") {
//   Tensor<double, Symmetry<1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Ru{};
//   std::iota(Ru.begin(), Ru.end(), 0.0);
//   Tensor<double, Symmetry<1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Su{};
//   std::iota(Su.begin(), Su.end(), 0.0);
//   auto L_ab = TensorExpressions::evaluate<ti_A_t, ti_B_t>(Ru(ti_A) *
//   Su(ti_B));

//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       CHECK(L_ab.get(a, b) == Ru.get(a) * Su.get(b));
//     }
//   }

//   Tensor<double, Symmetry<1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Sl{};
//   std::iota(Sl.begin(), Sl.end(), 0.0);

//   // auto L_expr = Ru(ti_A) * Sl(ti_a);
//   // td<decltype(L_expr)::args_list>idk1;
//   // td<decltype(L_expr)>idk2;

//   // inner product
//   auto L = TensorExpressions::evaluate(Ru(ti_A) * Sl(ti_a));

//   double expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     expected_sum += (Ru.get(a) * Su.get(a));
//   }
//   CHECK(L.get() == expected_sum);
// }

// SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct2By2",
//                   "[DataStructures][Unit]") {
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Rll{};
//   std::iota(Rll.begin(), Rll.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Sll{};
//   std::iota(Sll.begin(), Sll.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Ruu{};
//   std::iota(Ruu.begin(), Ruu.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Suu{};
//   std::iota(Suu.begin(), Suu.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Rlu{};
//   std::iota(Rlu.begin(), Rlu.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Slu{};
//   std::iota(Slu.begin(), Slu.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Rul{};
//   std::iota(Rul.begin(), Rul.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Sul{};
//   std::iota(Sul.begin(), Sul.end(), 0.0);

//   auto L_abAB_product =
//       TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_A, ti_B));
//   double L_abAB_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_abAB_expected_sum += (Rll.get(a, b) * Suu.get(a, b));
//     }
//   }
//   CHECK(L_abAB_product.get() == L_abAB_expected_sum);

//   auto L_abBA_product =
//       TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_B, ti_A));
//   double L_abBA_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_abBA_expected_sum += (Rll.get(a, b) * Suu.get(b, a));
//     }
//   }
//   CHECK(L_abBA_product.get() == L_abBA_expected_sum);

//   auto L_baAB_product =
//       TensorExpressions::evaluate(Rll(ti_b, ti_a) * Suu(ti_A, ti_B));
//   double L_baAB_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_baAB_expected_sum += (Rll.get(b, a) * Suu.get(a, b));
//     }
//   }
//   CHECK(L_baAB_product.get() == L_baAB_expected_sum);

//   auto L_baBA_product =
//       TensorExpressions::evaluate(Rll(ti_b, ti_a) * Suu(ti_B, ti_A));
//   double L_baBA_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_baBA_expected_sum += (Rll.get(b, a) * Suu.get(b, a));
//     }
//   }
//   CHECK(L_baBA_product.get() == L_baBA_expected_sum);

//   auto L_ABab_product =
//       TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Sll(ti_a, ti_b));
//   double L_ABab_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_ABab_expected_sum += (Ruu.get(a, b) * Sll.get(a, b));
//     }
//   }
//   CHECK(L_ABab_product.get() == L_ABab_expected_sum);

//   auto L_ABba_product =
//       TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Sll(ti_b, ti_a));
//   double L_ABba_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_ABba_expected_sum += (Ruu.get(a, b) * Sll.get(b, a));
//     }
//   }
//   CHECK(L_ABba_product.get() == L_ABba_expected_sum);

//   auto L_BAab_product =
//       TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Sll(ti_a, ti_b));
//   double L_BAab_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_BAab_expected_sum += (Ruu.get(b, a) * Sll.get(a, b));
//     }
//   }
//   CHECK(L_BAab_product.get() == L_BAab_expected_sum);

//   auto L_BAba_product =
//       TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Sll(ti_b, ti_a));
//   double L_BAba_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_BAba_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
//     }
//   }
//   CHECK(L_BAba_product.get() == L_BAba_expected_sum);

//   auto L_aBAb_product =
//       TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Sul(ti_A, ti_b));
//   double L_aBAb_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_aBAb_expected_sum += (Rlu.get(a, b) * Sul.get(a, b));
//     }
//   }
//   CHECK(L_aBAb_product.get() == L_aBAb_expected_sum);

//   auto L_AbaB_product =
//       TensorExpressions::evaluate(Rul(ti_A, ti_b) * Slu(ti_a, ti_B));
//   double L_AbaB_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_AbaB_expected_sum += (Rul.get(a, b) * Slu.get(a, b));
//     }
//   }
//   CHECK(L_AbaB_product.get() == L_AbaB_expected_sum);

//   auto L_aBbA_product =
//       TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Slu(ti_b, ti_A));
//   double L_aBbA_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_aBbA_expected_sum += (Rlu.get(a, b) * Slu.get(b, a));
//     }
//   }
//   CHECK(L_aBbA_product.get() == L_aBbA_expected_sum);

//   auto L_BaAb_product =
//       TensorExpressions::evaluate(Rul(ti_B, ti_a) * Sul(ti_A, ti_b));
//   double L_BaAb_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_BaAb_expected_sum += (Rlu.get(b, a) * Sul.get(a, b));
//     }
//   }
//   CHECK(L_BaAb_product.get() == L_BaAb_expected_sum);

//   auto L_AbBa_product =
//       TensorExpressions::evaluate(Rul(ti_A, ti_b) * Sul(ti_B, ti_a));
//   double L_AbBa_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_AbBa_expected_sum += (Rul.get(a, b) * Sul.get(b, a));
//     }
//   }
//   CHECK(L_AbBa_product.get() == L_AbBa_expected_sum);

//   auto L_BabA_product =
//       TensorExpressions::evaluate(Rul(ti_B, ti_a) * Slu(ti_b, ti_A));
//   double L_BabA_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_BabA_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
//     }
//   }
//   CHECK(L_BabA_product.get() == L_BabA_expected_sum);

//   auto L_bAaB_product =
//       TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Slu(ti_a, ti_B));
//   double L_bAaB_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_bAaB_expected_sum += (Ruu.get(b, a) * Sll.get(a, b));
//     }
//   }
//   CHECK(L_bAaB_product.get() == L_bAaB_expected_sum);

//   auto L_bABa_product =
//       TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Sul(ti_B, ti_a));
//   double L_bABa_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       L_bABa_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
//     }
//   }
//   CHECK(L_bABa_product.get() == L_bABa_expected_sum);
// }

// SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct3By3",
//                   "[DataStructures][Unit]") {
//   Tensor<double, Symmetry<3, 2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Rlul{};
//   std::iota(Rlul.begin(), Rlul.end(), 0.0);
//   Tensor<double, Symmetry<3, 2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Suul{};
//   std::iota(Suul.begin(), Suul.end(), 0.0);

//   auto L_aBcCAb_product =
//       TensorExpressions::evaluate(Rlul(ti_a, ti_B, ti_c) * Suul(ti_C, ti_A,
//       ti_b));
//   double L_aBcCAb_expected_sum = 0.0;
//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       for (size_t c = 0; c < 4; c++) {
//         L_aBcCAb_expected_sum += (Rlul.get(a, b, c) * Suul.get(c, a, b));
//       }
//     }
//   }
//   CHECK(L_aBcCAb_product.get() == L_aBcCAb_expected_sum);
// }

// SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct2By2",
//                   "[DataStructures][Unit]") {
//   Tensor<double, Symmetry<1, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Rll{};
//   std::iota(Rll.begin(), Rll.end(), 0.0);
//   Tensor<double, Symmetry<2, 1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
//                     SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Sll{};
//   std::iota(Sll.begin(), Sll.end(), 0.0);
//     auto L_abcd = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t,
//     ti_d_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_abdc = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_d_t,
//     ti_c_t>(
//       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_acbd = TensorExpressions::evaluate<ti_a_t, ti_c_t, ti_b_t,
//     ti_d_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_acdb = TensorExpressions::evaluate<ti_a_t, ti_c_t, ti_d_t,
//     ti_b_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_adbc = TensorExpressions::evaluate<ti_a_t, ti_d_t, ti_b_t,
//     ti_c_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_adcb = TensorExpressions::evaluate<ti_a_t, ti_d_t, ti_c_t,
//     ti_b_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

//     auto L_bacd = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_c_t,
//     ti_d_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_badc = TensorExpressions::evaluate<ti_b_t, ti_a_t, ti_d_t,
//     ti_c_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_bcad = TensorExpressions::evaluate<ti_b_t, ti_c_t, ti_a_t,
//     ti_d_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_bcda = TensorExpressions::evaluate<ti_b_t, ti_c_t, ti_d_t,
//     ti_a_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_bdac = TensorExpressions::evaluate<ti_b_t, ti_d_t, ti_a_t,
//     ti_c_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_bdca = TensorExpressions::evaluate<ti_b_t, ti_d_t, ti_c_t,
//     ti_a_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

//   auto L_cabd = TensorExpressions::evaluate<ti_c_t, ti_a_t, ti_b_t,
//     ti_d_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_cadb = TensorExpressions::evaluate<ti_c_t, ti_a_t, ti_d_t,
//     ti_b_t>(
//       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_cbad = TensorExpressions::evaluate<ti_c_t, ti_b_t, ti_a_t,
//     ti_d_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_cbda = TensorExpressions::evaluate<ti_c_t, ti_b_t, ti_d_t,
//     ti_a_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_cdab = TensorExpressions::evaluate<ti_c_t, ti_d_t, ti_a_t,
//     ti_b_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_cdba = TensorExpressions::evaluate<ti_c_t, ti_d_t, ti_b_t,
//     ti_a_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

//   auto L_dabc = TensorExpressions::evaluate<ti_d_t, ti_a_t, ti_b_t,
//     ti_c_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_dacb = TensorExpressions::evaluate<ti_d_t, ti_a_t, ti_c_t,
//     ti_b_t>(
//       Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_dbac = TensorExpressions::evaluate<ti_d_t, ti_b_t, ti_a_t,
//     ti_c_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_dbca = TensorExpressions::evaluate<ti_d_t, ti_b_t, ti_c_t,
//     ti_a_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_dcab = TensorExpressions::evaluate<ti_d_t, ti_c_t, ti_a_t,
//     ti_b_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
//     auto L_dcba = TensorExpressions::evaluate<ti_d_t, ti_c_t, ti_b_t,
//     ti_a_t>(
//         Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       for (size_t c = 0; c < 4; c++) {
//         for (size_t d = 0; d < 4; d++) {
//             // std::array<size_t, 4> index = {a, b, d, c};
//             // std::cout << "Current index: " << index << std::endl;
//             // std::cout << "R(a, b) : " << Rll.get(a, b) << std::endl;
//             // std::cout << "S(d, c) : " << Sll.get(d, c) << std::endl;

//             CHECK(L_abcd.get(a, b, c, d) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_abdc.get(a, b, d, c) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_acbd.get(a, c, b, d) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_acdb.get(a, c, d, b) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_adbc.get(a, d, b, c) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_adcb.get(a, d, c, b) == Rll.get(a, b) * Sll.get(c, d));

//             CHECK(L_bacd.get(b, a, c, d) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_badc.get(b, a, d, c) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_bcad.get(b, c, a, d) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_bcda.get(b, c, d, a) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_bdac.get(b, d, a, c) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_bdca.get(b, d, c, a) == Rll.get(a, b) * Sll.get(c, d));

//             CHECK(L_cabd.get(c, a, b, d) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_cadb.get(c, a, d, b) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_cbad.get(c, b, a, d) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_cbda.get(c, b, d, a) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_cdab.get(c, d, a, b) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_cdba.get(c, d, b, a) == Rll.get(a, b) * Sll.get(c, d));

//             CHECK(L_dabc.get(d, a, b, c) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_dacb.get(d, a, c, b) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_dbac.get(d, b, a, c) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_dbca.get(d, b, c, a) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_dcab.get(d, c, a, b) == Rll.get(a, b) * Sll.get(c, d));
//             CHECK(L_dcba.get(d, c, b, a) == Rll.get(a, b) * Sll.get(c, d));
//         }
//       }
//     }
//   }
// }

// SPECTRE_TEST_CASE(
//   "Unit.DataStructures.Tensor.Expression.OuterProduct1By1By1",
//                   "[DataStructures][Unit]") {
//   Tensor<double, Symmetry<1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Ru{};
//   std::iota(Ru.begin(), Ru.end(), 0.0);
//   Tensor<double, Symmetry<1>,
//          index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
//       Sl{};
//   std::iota(Sl.begin(), Sl.end(), 0.0);
//   Tensor<double, Symmetry<1>,
//          index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
//       Tu{};
//   std::iota(Tu.begin(), Tu.end(), 0.0);

//   auto L_AbC = TensorExpressions::evaluate<ti_A_t, ti_b_t, ti_C_t>(
//       Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
//   auto L_ACb = TensorExpressions::evaluate<ti_A_t, ti_C_t, ti_b_t>(
//       Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
//   auto L_bAC = TensorExpressions::evaluate<ti_b_t, ti_A_t, ti_C_t>(
//       Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
//   auto L_bCA = TensorExpressions::evaluate<ti_b_t, ti_C_t, ti_A_t>(
//       Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
//   auto L_CAb = TensorExpressions::evaluate<ti_C_t, ti_A_t, ti_b_t>(
//       Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
//   auto L_CbA = TensorExpressions::evaluate<ti_C_t, ti_b_t, ti_A_t>(
//       Ru(ti_A) * Sl(ti_b) * Tu(ti_C));

//   for (size_t a = 0; a < 4; a++) {
//     for (size_t b = 0; b < 4; b++) {
//       for (size_t c = 0; c < 4; c++) {
//         CHECK(L_AbC.get(a, b, c) == Ru.get(a) * Sl.get(b) * Tu.get(c));
//         CHECK(L_ACb.get(a, c, b) == Ru.get(a) * Sl.get(b) * Tu.get(c));
//         CHECK(L_bAC.get(b, a, c) == Ru.get(a) * Sl.get(b) * Tu.get(c));
//         CHECK(L_bCA.get(b, c, a) == Ru.get(a) * Sl.get(b) * Tu.get(c));
//         CHECK(L_CAb.get(c, a, b) == Ru.get(a) * Sl.get(b) * Tu.get(c));
//         CHECK(L_CbA.get(c, b, a) == Ru.get(a) * Sl.get(b) * Tu.get(c));
//       }
//     }
//   }
// }

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerAndOuterProduct",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sul{};
  std::iota(Sul.begin(), Sul.end(), 0.0);
  auto L_abBc_to_ac = TensorExpressions::evaluate<ti_a_t, ti_c_t>(
      Rll(ti_a, ti_b) * Sul(ti_B, ti_c));
  auto L_abBc_to_ca = TensorExpressions::evaluate<ti_c_t, ti_a_t>(
      Rll(ti_a, ti_b) * Sul(ti_B, ti_c));

  for (size_t a = 0; a < 4; a++) {
    for (size_t c = 0; c < 4; c++) {
      double expected_sum = 0.0;
      for (size_t b = 0; b < 4; b++) {
        expected_sum += (Rll.get(a, b) * Sul.get(b, c));
      }
      CHECK(L_abBc_to_ac.get(a, c) == expected_sum);
      CHECK(L_abBc_to_ca.get(c, a) == expected_sum);
    }
  }
}
