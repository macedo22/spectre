// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
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

  const Tensor<double> R_S_sum = TensorExpressions::evaluate(R() + S());
  const Tensor<double> R_S_difference = TensorExpressions::evaluate(R() - S());

  CHECK(R_S_sum.get() == expected_R_S_sum);
  CHECK(R_S_difference.get() == expected_R_S_difference);

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
}
