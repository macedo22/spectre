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
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

template <typename DataType>
void test_rank_0_outer_product(const DataType& used_for_size) noexcept {
  Tensor<DataType> R{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Instead of the tensor's value being the whole number, `used_for_size`
    R.get() = -3.7;
  } else {
    // Instead of the tensor's `DataVector` having elements with all the same
    // whole number value
    create_tensor(make_not_null(&R));
  }

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Su(used_for_size);
  create_tensor(make_not_null(&Su));

  // \f$L^{a} = R * S^{a}\f$
  const decltype(Su) LA_from_R_SA =
      TensorExpressions::evaluate<ti_A>(R() * Su(ti_A));
  // \f$L^{a} = S^{a} * R\f$
  const decltype(Su) LA_from_SA_R =
      TensorExpressions::evaluate<ti_A>(Su(ti_A) * R());

  for (size_t a = 0; a < 4; a++) {
    CHECK(LA_from_R_SA.get(a) == R.get() * Su.get(a));
    CHECK(LA_from_SA_R.get(a) == Su.get(a) * R.get());
  }
}

template <typename DataType>
void test_products(const DataType& used_for_size) noexcept {
  test_rank_0_outer_product(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Product",
                  "[DataStructures][Unit]") {
  test_products(std::numeric_limits<double>::signaling_NaN());
  test_products(DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct0By1",
                  "[DataStructures][Unit]") {
  const Tensor<double> R{{{3.7}}};
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Su{};
  std::iota(Su.begin(), Su.end(), 0.0);
  auto L_A = TensorExpressions::evaluate<ti_A>(R() * Su(ti_A));

  for (size_t a = 0; a < 4; a++) {
    CHECK(L_A.get(a) == 3.7 * Su.get(a));
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct0By2",
                  "[DataStructures][Unit]") {
  const Tensor<double> R{{{4.2}}};
  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sll{};
  std::iota(Sll.begin(), Sll.end(), 0.0);
  auto L_ab =
      TensorExpressions::evaluate<ti_a, ti_b>(R() * Sll(ti_a, ti_b));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      CHECK(L_ab.get(a, b) == 4.2 * Sll.get(a, b));
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct1By1",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Su{};
  std::iota(Su.begin(), Su.end(), 0.0);
  auto L_ab = TensorExpressions::evaluate<ti_A, ti_B>(Ru(ti_A) * Su(ti_B));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      CHECK(L_ab.get(a, b) == Ru.get(a) * Su.get(b));
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct1By1",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sl{};
  std::iota(Sl.begin(), Sl.end(), 0.0);
  auto L = TensorExpressions::evaluate(Ru(ti_A) * Sl(ti_a));

  double expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    expected_sum += (Ru.get(a) * Sl.get(a));
  }
  CHECK(L.get() == expected_sum);
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

  auto L_abAB_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_A, ti_B));
  double L_abAB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_abAB_expected_sum += (Rll.get(a, b) * Suu.get(a, b));
    }
  }
  CHECK(L_abAB_product.get() == L_abAB_expected_sum);

  auto L_abBA_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_b) * Suu(ti_B, ti_A));
  double L_abBA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_abBA_expected_sum += (Rll.get(a, b) * Suu.get(b, a));
    }
  }
  CHECK(L_abBA_product.get() == L_abBA_expected_sum);

  auto L_baAB_product =
      TensorExpressions::evaluate(Rll(ti_b, ti_a) * Suu(ti_A, ti_B));
  double L_baAB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_baAB_expected_sum += (Rll.get(b, a) * Suu.get(a, b));
    }
  }
  CHECK(L_baAB_product.get() == L_baAB_expected_sum);

  auto L_baBA_product =
      TensorExpressions::evaluate(Rll(ti_b, ti_a) * Suu(ti_B, ti_A));
  double L_baBA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_baBA_expected_sum += (Rll.get(b, a) * Suu.get(b, a));
    }
  }
  CHECK(L_baBA_product.get() == L_baBA_expected_sum);

  auto L_ABab_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Sll(ti_a, ti_b));
  double L_ABab_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_ABab_expected_sum += (Ruu.get(a, b) * Sll.get(a, b));
    }
  }
  CHECK(L_ABab_product.get() == L_ABab_expected_sum);

  auto L_ABba_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_B) * Sll(ti_b, ti_a));
  double L_ABba_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_ABba_expected_sum += (Ruu.get(a, b) * Sll.get(b, a));
    }
  }
  CHECK(L_ABba_product.get() == L_ABba_expected_sum);

  auto L_BAab_product =
      TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Sll(ti_a, ti_b));
  double L_BAab_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BAab_expected_sum += (Ruu.get(b, a) * Sll.get(a, b));
    }
  }
  CHECK(L_BAab_product.get() == L_BAab_expected_sum);

  auto L_BAba_product =
      TensorExpressions::evaluate(Ruu(ti_B, ti_A) * Sll(ti_b, ti_a));
  double L_BAba_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BAba_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
    }
  }
  CHECK(L_BAba_product.get() == L_BAba_expected_sum);

  auto L_aBAb_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Sul(ti_A, ti_b));
  double L_aBAb_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_aBAb_expected_sum += (Rlu.get(a, b) * Sul.get(a, b));
    }
  }
  CHECK(L_aBAb_product.get() == L_aBAb_expected_sum);

  auto L_AbaB_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_b) * Slu(ti_a, ti_B));
  double L_AbaB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_AbaB_expected_sum += (Rul.get(a, b) * Slu.get(a, b));
    }
  }
  CHECK(L_AbaB_product.get() == L_AbaB_expected_sum);

  auto L_aBbA_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_B) * Slu(ti_b, ti_A));
  double L_aBbA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_aBbA_expected_sum += (Rlu.get(a, b) * Slu.get(b, a));
    }
  }
  CHECK(L_aBbA_product.get() == L_aBbA_expected_sum);

  auto L_BaAb_product =
      TensorExpressions::evaluate(Rul(ti_B, ti_a) * Sul(ti_A, ti_b));
  double L_BaAb_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BaAb_expected_sum += (Rlu.get(b, a) * Sul.get(a, b));
    }
  }
  CHECK(L_BaAb_product.get() == L_BaAb_expected_sum);

  auto L_AbBa_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_b) * Sul(ti_B, ti_a));
  double L_AbBa_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_AbBa_expected_sum += (Rul.get(a, b) * Sul.get(b, a));
    }
  }
  CHECK(L_AbBa_product.get() == L_AbBa_expected_sum);

  auto L_BabA_product =
      TensorExpressions::evaluate(Rul(ti_B, ti_a) * Slu(ti_b, ti_A));
  double L_BabA_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_BabA_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
    }
  }
  CHECK(L_BabA_product.get() == L_BabA_expected_sum);

  auto L_bAaB_product =
      TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Slu(ti_a, ti_B));
  double L_bAaB_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_bAaB_expected_sum += (Ruu.get(b, a) * Sll.get(a, b));
    }
  }
  CHECK(L_bAaB_product.get() == L_bAaB_expected_sum);

  auto L_bABa_product =
      TensorExpressions::evaluate(Rlu(ti_b, ti_A) * Sul(ti_B, ti_a));
  double L_bABa_expected_sum = 0.0;
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      L_bABa_expected_sum += (Ruu.get(b, a) * Sll.get(b, a));
    }
  }
  CHECK(L_bABa_product.get() == L_bABa_expected_sum);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerProduct2By2By2",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rll{};
  std::iota(Rll.begin(), Rll.end(), 0.0);
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Suu{};
  std::iota(Suu.begin(), Suu.end(), 0.0);
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Tll{};
  std::iota(Tll.begin(), Tll.end(), 0.0);

  auto L_klKLij_to_ij = TensorExpressions::evaluate<ti_i, ti_j>(
      Rll(ti_k, ti_l) * Suu(ti_K, ti_L) * Tll(ti_i, ti_j));

  double trace = 0.0;
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 3; l++) {
      trace += (Rll.get(k, l) * Suu.get(k, l));
    }
  }

  CHECK(TensorExpressions::evaluate(Rll(ti_k, ti_l) * Suu(ti_K, ti_L)).get() ==
        trace);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      double expected_sum = 0.0;
      for (size_t k = 0; k < 3; k++) {
        for (size_t l = 0; l < 3; l++) {
          expected_sum += Rll.get(k, l) * Suu.get(k, l) * Tll.get(i, j);
        }
      }
      CHECK(expected_sum == (trace * Tll.get(i, j)));
    }
  }

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      CHECK(L_klKLij_to_ij.get(i, j) == (trace * Tll.get(i, j)));
    }
  }
}

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
  auto L_abcd = TensorExpressions::evaluate<ti_a, ti_b, ti_c, ti_d>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_abdc = TensorExpressions::evaluate<ti_a, ti_b, ti_d, ti_c>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_acbd = TensorExpressions::evaluate<ti_a, ti_c, ti_b, ti_d>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_acdb = TensorExpressions::evaluate<ti_a, ti_c, ti_d, ti_b>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_adbc = TensorExpressions::evaluate<ti_a, ti_d, ti_b, ti_c>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_adcb = TensorExpressions::evaluate<ti_a, ti_d, ti_c, ti_b>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  auto L_bacd = TensorExpressions::evaluate<ti_b, ti_a, ti_c, ti_d>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_badc = TensorExpressions::evaluate<ti_b, ti_a, ti_d, ti_c>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_bcad = TensorExpressions::evaluate<ti_b, ti_c, ti_a, ti_d>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_bcda = TensorExpressions::evaluate<ti_b, ti_c, ti_d, ti_a>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_bdac = TensorExpressions::evaluate<ti_b, ti_d, ti_a, ti_c>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_bdca = TensorExpressions::evaluate<ti_b, ti_d, ti_c, ti_a>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  auto L_cabd = TensorExpressions::evaluate<ti_c, ti_a, ti_b, ti_d>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_cadb = TensorExpressions::evaluate<ti_c, ti_a, ti_d, ti_b>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_cbad = TensorExpressions::evaluate<ti_c, ti_b, ti_a, ti_d>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_cbda = TensorExpressions::evaluate<ti_c, ti_b, ti_d, ti_a>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_cdab = TensorExpressions::evaluate<ti_c, ti_d, ti_a, ti_b>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_cdba = TensorExpressions::evaluate<ti_c, ti_d, ti_b, ti_a>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  auto L_dabc = TensorExpressions::evaluate<ti_d, ti_a, ti_b, ti_c>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_dacb = TensorExpressions::evaluate<ti_d, ti_a, ti_c, ti_b>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_dbac = TensorExpressions::evaluate<ti_d, ti_b, ti_a, ti_c>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_dbca = TensorExpressions::evaluate<ti_d, ti_b, ti_c, ti_a>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_dcab = TensorExpressions::evaluate<ti_d, ti_c, ti_a, ti_b>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));
  auto L_dcba = TensorExpressions::evaluate<ti_d, ti_c, ti_b, ti_a>(
      Rll(ti_a, ti_b) * Sll(ti_c, ti_d));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 4; d++) {
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

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct1By1By1",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sl{};
  std::iota(Sl.begin(), Sl.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Tu{};
  std::iota(Tu.begin(), Tu.end(), 0.0);

  auto L_AbC = TensorExpressions::evaluate<ti_A, ti_b, ti_C>(
      Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
  auto L_ACb = TensorExpressions::evaluate<ti_A, ti_C, ti_b>(
      Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
  auto L_bAC = TensorExpressions::evaluate<ti_b, ti_A, ti_C>(
      Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
  auto L_bCA = TensorExpressions::evaluate<ti_b, ti_C, ti_A>(
      Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
  auto L_CAb = TensorExpressions::evaluate<ti_C, ti_A, ti_b>(
      Ru(ti_A) * Sl(ti_b) * Tu(ti_C));
  auto L_CbA = TensorExpressions::evaluate<ti_C, ti_b, ti_A>(
      Ru(ti_A) * Sl(ti_b) * Tu(ti_C));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        CHECK(L_AbC.get(a, b, c) == Ru.get(a) * Sl.get(b) * Tu.get(c));
        CHECK(L_ACb.get(a, c, b) == Ru.get(a) * Sl.get(b) * Tu.get(c));
        CHECK(L_bAC.get(b, a, c) == Ru.get(a) * Sl.get(b) * Tu.get(c));
        CHECK(L_bCA.get(b, c, a) == Ru.get(a) * Sl.get(b) * Tu.get(c));
        CHECK(L_CAb.get(c, a, b) == Ru.get(a) * Sl.get(b) * Tu.get(c));
        CHECK(L_CbA.get(c, b, a) == Ru.get(a) * Sl.get(b) * Tu.get(c));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.OuterProduct1By2By1",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Slu{};
  std::iota(Slu.begin(), Slu.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tl{};
  std::iota(Tl.begin(), Tl.end(), 0.0);

  auto L_aBCd = TensorExpressions::evaluate<ti_a, ti_B, ti_C, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_aBdC = TensorExpressions::evaluate<ti_a, ti_B, ti_d, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_aCBd = TensorExpressions::evaluate<ti_a, ti_C, ti_B, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_aCdB = TensorExpressions::evaluate<ti_a, ti_C, ti_d, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_adBC = TensorExpressions::evaluate<ti_a, ti_d, ti_B, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_adCB = TensorExpressions::evaluate<ti_a, ti_d, ti_C, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  auto L_BaCd = TensorExpressions::evaluate<ti_B, ti_a, ti_C, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BadC = TensorExpressions::evaluate<ti_B, ti_a, ti_d, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BCad = TensorExpressions::evaluate<ti_B, ti_C, ti_a, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BCda = TensorExpressions::evaluate<ti_B, ti_C, ti_d, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BdaC = TensorExpressions::evaluate<ti_B, ti_d, ti_a, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_BdCa = TensorExpressions::evaluate<ti_B, ti_d, ti_C, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  auto L_CaBd = TensorExpressions::evaluate<ti_C, ti_a, ti_B, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CadB = TensorExpressions::evaluate<ti_C, ti_a, ti_d, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CBad = TensorExpressions::evaluate<ti_C, ti_B, ti_a, ti_d>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CBda = TensorExpressions::evaluate<ti_C, ti_B, ti_d, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CdaB = TensorExpressions::evaluate<ti_C, ti_d, ti_a, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_CdBa = TensorExpressions::evaluate<ti_C, ti_d, ti_B, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  auto L_daBC = TensorExpressions::evaluate<ti_d, ti_a, ti_B, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_daCB = TensorExpressions::evaluate<ti_d, ti_a, ti_C, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dBaC = TensorExpressions::evaluate<ti_d, ti_B, ti_a, ti_C>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dBCa = TensorExpressions::evaluate<ti_d, ti_B, ti_C, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dCaB = TensorExpressions::evaluate<ti_d, ti_C, ti_a, ti_B>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));
  auto L_dCBa = TensorExpressions::evaluate<ti_d, ti_C, ti_B, ti_a>(
      Ru(ti_B) * Slu(ti_d, ti_C) * Tl(ti_a));

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 4; d++) {
          double expected_product = Ru.get(b) * Slu.get(d, c) * Tl.get(a);
          CHECK(L_aBCd.get(a, b, c, d) == expected_product);
          CHECK(L_aBdC.get(a, b, d, c) == expected_product);
          CHECK(L_aCBd.get(a, c, b, d) == expected_product);
          CHECK(L_aCdB.get(a, c, d, b) == expected_product);
          CHECK(L_adBC.get(a, d, b, c) == expected_product);
          CHECK(L_adCB.get(a, d, c, b) == expected_product);

          CHECK(L_BaCd.get(b, a, c, d) == expected_product);
          CHECK(L_BadC.get(b, a, d, c) == expected_product);
          CHECK(L_BCad.get(b, c, a, d) == expected_product);
          CHECK(L_BCda.get(b, c, d, a) == expected_product);
          CHECK(L_BdaC.get(b, d, a, c) == expected_product);
          CHECK(L_BdCa.get(b, d, c, a) == expected_product);

          CHECK(L_CaBd.get(c, a, b, d) == expected_product);
          CHECK(L_CadB.get(c, a, d, b) == expected_product);
          CHECK(L_CBad.get(c, b, a, d) == expected_product);
          CHECK(L_CBda.get(c, b, d, a) == expected_product);
          CHECK(L_CdaB.get(c, d, a, b) == expected_product);
          CHECK(L_CdBa.get(c, d, b, a) == expected_product);

          CHECK(L_daBC.get(d, a, b, c) == expected_product);
          CHECK(L_daCB.get(d, a, c, b) == expected_product);
          CHECK(L_dBaC.get(d, b, a, c) == expected_product);
          CHECK(L_dBCa.get(d, b, c, a) == expected_product);
          CHECK(L_dCaB.get(d, c, a, b) == expected_product);
          CHECK(L_dCBa.get(d, c, b, a) == expected_product);
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.InnerOuterProduct",
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
  auto L_abBc_to_ac = TensorExpressions::evaluate<ti_a, ti_c>(
      Rll(ti_a, ti_b) * Sul(ti_B, ti_c));
  auto L_abBc_to_ca = TensorExpressions::evaluate<ti_c, ti_a>(
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

  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru{};
  std::iota(Ru.begin(), Ru.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sl{};
  std::iota(Sl.begin(), Sl.end(), 0.0);
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tl{};
  std::iota(Tl.begin(), Tl.end(), 0.0);

  auto L_Aab_to_b =
      TensorExpressions::evaluate<ti_b>(Ru(ti_A) * Sl(ti_a) * Tl(ti_b));

  for (size_t b = 0; b < 4; b++) {
    double expected_sum = 0.0;
    for (size_t a = 0; a < 4; a++) {
      expected_sum += (Ru.get(a) * Sl.get(a) * Tl.get(b));
    }
    CHECK(L_Aab_to_b.get(b) == expected_sum);
  }

  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tll{};
  std::iota(Tll.begin(), Tll.end(), 0.0);

  auto L_Aabc_to_bc = TensorExpressions::evaluate<ti_b, ti_c>(
      Ru(ti_A) * Sl(ti_a) * Tll(ti_b, ti_c));

  for (size_t c = 0; c < 4; c++) {
    for (size_t b = 0; b < 4; b++) {
      double expected_sum = 0.0;
      for (size_t a = 0; a < 4; a++) {
        expected_sum += (Ru.get(a) * Sl.get(a) * Tll.get(b, c));
      }
      CHECK(L_Aabc_to_bc.get(b, c) == expected_sum);
    }
  }
}
