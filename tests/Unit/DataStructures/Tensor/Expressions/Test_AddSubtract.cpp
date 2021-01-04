// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtract",
                  "[DataStructures][Unit]") {
  // Test adding and subtracting rank 0 tensors
  const double R_value = 2.5;
  const double S_value = -1.25;
  const Tensor<double> R{{{R_value}}};
  const Tensor<double> S{{{S_value}}};
  const double expected_R_S_sum = R_value + S_value;
  const double expected_R_S_difference = R_value - S_value;

  const Tensor<double> R_S_sum_1 = TensorExpressions::evaluate(R() + S());
  const Tensor<double> R_S_sum_2 = TensorExpressions::evaluate(R() + S_value);
  const Tensor<double> R_S_sum_3 = TensorExpressions::evaluate(R_value + S());
  const Tensor<double> R_S_difference_1 =
      TensorExpressions::evaluate(R() - S());
  const Tensor<double> R_S_difference_2 =
      TensorExpressions::evaluate(R() - S_value);
  const Tensor<double> R_S_difference_3 =
      TensorExpressions::evaluate(R_value - S());

  CHECK(R_S_sum_1.get() == expected_R_S_sum);
  CHECK(R_S_sum_2.get() == expected_R_S_sum);
  CHECK(R_S_sum_3.get() == expected_R_S_sum);
  CHECK(R_S_difference_1.get() == expected_R_S_difference);
  CHECK(R_S_difference_2.get() == expected_R_S_difference);
  CHECK(R_S_difference_3.get() == expected_R_S_difference);

  // Test adding and subtracting rank 2 tensors
  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);
  /// [use_tensor_index]
  auto Gll = TensorExpressions::evaluate<ti_a, ti_b>(All(ti_a, ti_b) +
                                                     Hll(ti_a, ti_b));
  auto Gll2 = TensorExpressions::evaluate<ti_a, ti_b>(All(ti_a, ti_b) +
                                                      Hll(ti_b, ti_a));
  auto Gll3 = TensorExpressions::evaluate<ti_a, ti_b>(
      All(ti_a, ti_b) + Hll(ti_b, ti_a) + All(ti_b, ti_a) - Hll(ti_b, ti_a));
  /// [use_tensor_index]
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      CHECK(Gll.get(i, j) == All.get(i, j) + Hll.get(i, j));
      CHECK(Gll2.get(i, j) == All.get(i, j) + Hll.get(j, i));
      CHECK(Gll3.get(i, j) == 2.0 * All.get(i, j));
    }
  }

  // Test adding and subtracting rank 3 tensors
  Tensor<double, Symmetry<1, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll{};
  std::iota(Alll.begin(), Alll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Hlll{};
  std::iota(Hlll.begin(), Hlll.end(), 0.0);
  auto Glll = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_a, ti_b, ti_c));
  auto Glll2 = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c));
  auto Glll3 = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c) + Alll(ti_b, ti_a, ti_c) -
      Hlll(ti_b, ti_a, ti_c));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        CHECK(Glll.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(i, j, k));
        CHECK(Glll2.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(j, i, k));
        CHECK(Glll3.get(i, j, k) == 2.0 * Alll.get(i, j, k));
      }
    }
  }

  // Test adding and subtracting a scalar with a contraction
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Tul{};
  std::iota(Tul.begin(), Tul.end(), 0.0);

  double expected_trace = 0.0;
  for (size_t a = 0; a < 4; a++) {
    expected_trace += Tul.get(a, a);
  }
  const double expected_R_TAa_sum = R_value + expected_trace;
  const double expected_R_TAa_difference_1 = R_value - expected_trace;
  const double expected_R_TAa_difference_2 = expected_trace - R_value;

  const Tensor<double> scalar_with_contraction_sum_1 =
      TensorExpressions::evaluate(R() + Tul(ti_A, ti_a));
  const Tensor<double> scalar_with_contraction_sum_2 =
      TensorExpressions::evaluate(Tul(ti_A, ti_a) + R());
  const Tensor<double> scalar_with_contraction_sum_3 =
      TensorExpressions::evaluate(R_value + Tul(ti_A, ti_a));
  const Tensor<double> scalar_with_contraction_sum_4 =
      TensorExpressions::evaluate(Tul(ti_A, ti_a) + R_value);
  const Tensor<double> scalar_with_contraction_difference_1 =
      TensorExpressions::evaluate(R() - Tul(ti_A, ti_a));
  const Tensor<double> scalar_with_contraction_difference_2 =
      TensorExpressions::evaluate(Tul(ti_A, ti_a) - R());
  const Tensor<double> scalar_with_contraction_difference_3 =
      TensorExpressions::evaluate(R_value - Tul(ti_A, ti_a));
  const Tensor<double> scalar_with_contraction_difference_4 =
      TensorExpressions::evaluate(Tul(ti_A, ti_a) - R_value);

  CHECK(scalar_with_contraction_sum_1.get() == expected_R_TAa_sum);
  CHECK(scalar_with_contraction_sum_2.get() == expected_R_TAa_sum);
  CHECK(scalar_with_contraction_sum_3.get() == expected_R_TAa_sum);
  CHECK(scalar_with_contraction_sum_4.get() == expected_R_TAa_sum);
  CHECK(scalar_with_contraction_difference_1.get() ==
        expected_R_TAa_difference_1);
  CHECK(scalar_with_contraction_difference_2.get() ==
        expected_R_TAa_difference_2);
  CHECK(scalar_with_contraction_difference_3.get() ==
        expected_R_TAa_difference_1);
  CHECK(scalar_with_contraction_difference_4.get() ==
        expected_R_TAa_difference_2);
}
