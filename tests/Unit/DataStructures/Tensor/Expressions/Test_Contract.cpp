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
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Aul{};
  std::iota(Aul.begin(), Aul.end(), 0.0);

  const Tensor<double> Ii_contracted =
      TensorExpressions::evaluate(Aul(ti_I, ti_i));

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

  const Tensor<double> gG_contracted =
      TensorExpressions::evaluate(Alu(ti_g, ti_G));

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
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Aull{};
  std::iota(Aull.begin(), Aull.end(), 0.0);

  const Tensor<double, Symmetry<1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Iij_contracted =
          TensorExpressions::evaluate<ti_j_t>(Aull(ti_I, ti_i, ti_j));

  for (size_t j = 0; j < 4; j++) {
    double expected_sum = 0.0;
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Aull.get(i, i, j);
    }
    CHECK(Iij_contracted.get(j) == expected_sum);
  }

  // Contract first and third indices of <2, 2, 1> symmetry rank 3 (upper,
  // upper, lower) tensor to rank 1 tensor
  Tensor<double, Symmetry<2, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Auul{};
  std::iota(Auul.begin(), Auul.end(), 0.0);

  const Tensor<double, Symmetry<1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      JLj_contracted =
          TensorExpressions::evaluate<ti_L_t>(Auul(ti_J, ti_L, ti_j));

  for (size_t l = 0; l < 3; l++) {
    double expected_sum = 0.0;
    for (size_t j = 0; j < 3; j++) {
      expected_sum += Auul.get(j, l, j);
    }
    CHECK(JLj_contracted.get(l) == expected_sum);
  }

  // Contract second and third indices of <2, 1, 2> symmetry rank 3 (upper,
  // lower, upper) tensor to rank 1 tensor
  Tensor<double, Symmetry<2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Aulu{};
  std::iota(Aulu.begin(), Aulu.end(), 0.0);

  const Tensor<double, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      BfF_contracted =
          TensorExpressions::evaluate<ti_B_t>(Aulu(ti_B, ti_f, ti_F));

  for (size_t b = 0; b < 4; b++) {
    double expected_sum = 0.0;
    for (size_t f = 0; f < 4; f++) {
      expected_sum += Aulu.get(b, f, f);
    }
    CHECK(BfF_contracted.get(b) == expected_sum);
  }

  // Contract first and second indices of nonsymmetric rank 4 (lower, upper,
  // upper, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Aluul{};
  std::iota(Aluul.begin(), Aluul.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      iIKj_contracted = TensorExpressions::evaluate<ti_K_t, ti_j_t>(
          Aluul(ti_i, ti_I, ti_K, ti_j));

  for (size_t k = 0; k < 4; k++) {
    for (size_t j = 0; j < 3; j++) {
      double expected_sum = 0.0;
      for (size_t i = 0; i < 3; i++) {
        expected_sum += Aluul.get(i, i, k, j);
      }
      CHECK(iIKj_contracted.get(k, j) == expected_sum);
    }
  }

  // Contract first and third indices of nonsymmetric rank 4 (upper, upper,
  // lower, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      Auull{};
  std::iota(Auull.begin(), Auull.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      ABac_contracted = TensorExpressions::evaluate<ti_B_t, ti_c_t>(
          Auull(ti_A, ti_B, ti_a, ti_c));

  for (size_t b = 0; b < 4; b++) {
    for (size_t c = 0; c < 5; c++) {
      double expected_sum = 0.0;
      for (size_t a = 0; a < 5; a++) {
        expected_sum += Auull.get(a, b, a, c);
      }
      CHECK(ABac_contracted.get(b, c) == expected_sum);
    }
  }

  // Contract first and fourth indices of <3, 2, 3, 1> symmetry rank 4 (upper,
  // upper, upper, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<3, 2, 3, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Auuul{};
  std::iota(Auuul.begin(), Auuul.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      LJIl_contracted = TensorExpressions::evaluate<ti_J_t, ti_I_t>(
          Auuul(ti_L, ti_J, ti_I, ti_l));

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 3; i++) {
      double expected_sum = 0.0;
      for (size_t l = 0; l < 3; l++) {
        expected_sum += Auuul.get(l, j, i, l);
      }
      CHECK(LJIl_contracted.get(j, i) == expected_sum);
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

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      EDdA_contracted = TensorExpressions::evaluate<ti_E_t, ti_A_t>(
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

  // Contract second and fourth indices of nonsymmetric rank 4 (lower, upper,
  // lower, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Alull{};
  std::iota(Alull.begin(), Alull.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      kJij_contracted = TensorExpressions::evaluate<ti_k_t, ti_i_t>(
          Alull(ti_k, ti_J, ti_i, ti_j));

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 4; i++) {
      double expected_sum = 0.0;
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Alull.get(k, j, i, j);
      }
      CHECK(kJij_contracted.get(k, i) == expected_sum);
    }
  }

  // Contract third and fourth indices of <3, 2, 2, 1> symmetry rank 4 (upper,
  // lower, lower, upper) tensor to rank 2 tensor
  Tensor<double, Symmetry<3, 2, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Aullu{};
  std::iota(Aullu.begin(), Aullu.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      FcgG_contracted = TensorExpressions::evaluate<ti_F_t, ti_c_t>(
          Aullu(ti_F, ti_c, ti_g, ti_G));

  for (size_t f = 0; f < 5; f++) {
    for (size_t c = 0; c < 4; c++) {
      double expected_sum = 0.0;
      for (size_t g = 0; g < 4; g++) {
        expected_sum += Aullu.get(f, c, g, g);
      }
      CHECK(FcgG_contracted.get(f, c) == expected_sum);
    }
  }

  // Contract first and fourth indices of <2, 1, 1, 1> symmetry rank 4 (upper,
  // lower, lower, lower) tensor to rank 2 tensor and reorder indices
  Tensor<double, Symmetry<2, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Aulll{};
  std::iota(Aulll.begin(), Aulll.end(), 0.0);

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Adba_contracted_to_bd = TensorExpressions::evaluate<ti_b_t, ti_d_t>(
          Aulll(ti_A, ti_d, ti_b, ti_a));

  for (size_t b = 0; b < 4; b++) {
    for (size_t d = 0; d < 4; d++) {
      double expected_sum = 0.0;
      for (size_t a = 0; a < 4; a++) {
        expected_sum += Aulll.get(a, d, b, a);
      }
      CHECK(Adba_contracted_to_bd.get(b, d) == expected_sum);
    }
  }

  // Contract second and third indices of <4, 3, 2, 1> symmetry rank 4 (lower,
  // lower, upper, lower) tensor to rank 2 tensor and reorder indices
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Allul{};
  std::iota(Allul.begin(), Allul.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      ljJi_contracted_to_il = TensorExpressions::evaluate<ti_i_t, ti_l_t>(
          Allul(ti_l, ti_j, ti_J, ti_i));

  for (size_t i = 0; i < 4; i++) {
    for (size_t l = 0; l < 3; l++) {
      double expected_sum = 0.0;
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Allul.get(l, j, j, i);
      }
      CHECK(ljJi_contracted_to_il.get(i, l) == expected_sum);
    }
  }

  // Contract first and second indices and third and fourth indices of
  // nonsymmetric rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Aulul{};
  std::iota(Aulul.begin(), Aulul.end(), 0.0);

  const Tensor<double> KkLl_contracted =
      TensorExpressions::evaluate(Aulul(ti_K, ti_k, ti_L, ti_l));

  double expected_KkLl_sum = 0.0;
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 4; l++) {
      expected_KkLl_sum += Aulul.get(k, k, l, l);
    }
  }
  CHECK(KkLl_contracted.get() == expected_KkLl_sum);

  // Contract first and third indices and second and fourth indices of
  // <2, 1, 1, 2> symmetry rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      Bluul{};
  std::iota(Bluul.begin(), Bluul.end(), 0.0);

  const Tensor<double> cACa_contracted =
      TensorExpressions::evaluate(Bluul(ti_c, ti_A, ti_C, ti_a));

  double expected_cACa_sum = 0.0;
  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      expected_cACa_sum += Bluul.get(c, a, c, a);
    }
  }
  CHECK(cACa_contracted.get() == expected_cACa_sum);

  // Contract first and fourth indices and second and third indices of
  // <2, 1, 2, 1> symmetry rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Alulu{};
  std::iota(Alulu.begin(), Alulu.end(), 0.0);

  const Tensor<double> jIiJ_contracted =
      TensorExpressions::evaluate(Alulu(ti_j, ti_I, ti_i, ti_J));

  double expected_jIiJ_sum = 0.0;
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_jIiJ_sum += Alulu.get(j, i, i, j);
    }
  }
  CHECK(jIiJ_contracted.get() == expected_jIiJ_sum);
}
