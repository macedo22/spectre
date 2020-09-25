// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  // Contract rank 2 (upper, lower) tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Aul{};
  std::iota(Aul.begin(), Aul.end(), 0.0);

  auto Ii_contracted = TensorExpressions::evaluate(Aul(ti_I, ti_i));

  double expected_Ii_sum = 0.0;
  for (size_t i = 0; i < 3; i++) {
    expected_Ii_sum += Aul.get(i, i);
  }
  CHECK(Ii_contracted.get() == expected_Ii_sum);

  // Contract rank 2 (lower, upper) tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Alu{};
  std::iota(Alu.begin(), Alu.end(), 0.0);

  auto gG_contracted = TensorExpressions::evaluate(Alu(ti_g, ti_G));

  double expected_gG_sum = 0.0;
  for (size_t g = 0; g < 4; g++) {
    expected_gG_sum += Alu.get(g, g);
  }
  CHECK(gG_contracted.get() == expected_gG_sum);

  // Contract first and second indices of nonsymmetric rank 3 (upper, lower,
  // lower) tensor to rank 1 tensor
  Tensor<double, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Aull{};
  std::iota(Aull.begin(), Aull.end(), 0.0);

  auto Iij_contracted =
      TensorExpressions::evaluate<ti_j_t>(Aull(ti_I, ti_i, ti_j));

  for (size_t j = 0; j < 3; j++) {
    double expected_sum = 0.0;
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Aull.get(i, i, j);
    }
    CHECK(Iij_contracted.get(j) == expected_sum);
  }

  // Contract second and third indices of <2, 1, 2> symmetry rank 3 (upper,
  // lower, upper) tensor to rank 1 tensor
  Tensor<double, Symmetry<2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Aulu{};
  std::iota(Aulu.begin(), Aulu.end(), 0.0);

  auto BfF_contracted =
      TensorExpressions::evaluate<ti_B_t>(Aulu(ti_B, ti_f, ti_F));

  for (size_t b = 0; b < 4; b++) {
    double expected_sum = 0.0;
    for (size_t f = 0; f < 4; f++) {
      expected_sum += Aulu.get(b, f, f);
    }
    CHECK(BfF_contracted.get(b) == expected_sum);
  }

  // Contract first and third indices of <2, 2, 1> symmetry rank 3 (upper,
  // upper, lower) tensor to rank 1 tensor
  Tensor<double, Symmetry<2, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Auul{};
  std::iota(Auul.begin(), Auul.end(), 0.0);

  auto JLj_contracted =
      TensorExpressions::evaluate<ti_L_t>(Auul(ti_J, ti_L, ti_j));

  for (size_t l = 0; l < 3; l++) {
    double expected_sum = 0.0;
    for (size_t j = 0; j < 3; j++) {
      expected_sum += Auul.get(j, l, j);
    }
    CHECK(JLj_contracted.get(l) == expected_sum);
  }

  // Contract second and fourth indices of nonsymmetric rank 4 (lower, upper,
  // lower, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Alull{};
  std::iota(Alull.begin(), Alull.end(), 0.0);

  auto kJij_contracted = TensorExpressions::evaluate<ti_k_t, ti_i_t>(
      Alull(ti_k, ti_J, ti_i, ti_j));

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 3; i++) {
      double expected_sum = 0.0;
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Alull.get(k, j, i, j);
      }
      CHECK(kJij_contracted.get(k, i) == expected_sum);
    }
  }

  // Contract second and third indices of <2, 2, 1, 2> symmetry rank 4 (upper,
  // upper, lower, upper) tensor to rank 2 tensor
  Tensor<double, Symmetry<2, 2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Auulu{};
  std::iota(Auulu.begin(), Auulu.end(), 0.0);

  auto EDdA_contracted = TensorExpressions::evaluate<ti_E_t, ti_A_t>(
      Auulu(ti_E, ti_D, ti_d, ti_A));

  for (size_t e = 0; e < 4; e++) {
    for (size_t a = 0; a < 4; a++) {
      double expected_sum = 0.0;
      for (size_t d = 0; d < 4; d++) {
        expected_sum += Auulu.get(e, d, d, a);
      }
      CHECK(EDdA_contracted.get(e, a) == expected_sum);
    }
  }

  // Contract first and second indices and third and fourth indices of
  // nonsymmetric rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Aulul{};
  std::iota(Aulul.begin(), Aulul.end(), 0.0);

  auto KkLl_contracted =
      TensorExpressions::evaluate(Aulul(ti_K, ti_k, ti_L, ti_l));

  double expected_KkLl_sum = 0.0;
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 3; l++) {
      expected_KkLl_sum += Aulul.get(k, k, l, l);
    }
  }
  CHECK(KkLl_contracted.get() == expected_KkLl_sum);

  // Contract first and third indices and second and fourth indices of<2, 1, 1,
  // 2> symmetry rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      Aluul{};
  std::iota(Aluul.begin(), Aluul.end(), 0.0);

  auto cACa_contracted = TensorExpressions::evaluate(Aluul(ti_c, ti_A, ti_C,
  ti_a));

  double expected_cACa_sum = 0.0;
  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      expected_cACa_sum += Aluul.get(c, a, c, a);
    }
  }
  CHECK(cACa_contracted.get() == expected_cACa_sum);
}
