// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

template <typename T,
          Requires<std::is_same_v<typename T::type, DataVector>> = nullptr>
void assign_unique_datavector_tensor_values(T& tensor) {
  double value = 0.0;
  for (auto index_it = tensor.begin(); index_it != tensor.end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  // Contract rank 2 (upper, lower) tensor to rank 0 tensor
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rul{};
  std::iota(Rul.begin(), Rul.end(), 0.0);

  const Tensor<double> RIi_contracted =
      TensorExpressions::evaluate(Rul(ti_I, ti_i));

  double expected_RIi_sum = 0.0;
  for (size_t i = 0; i < 3; i++) {
    expected_RIi_sum += Rul.get(i, i);
  }
  CHECK(RIi_contracted.get() == expected_RIi_sum);

  Tensor<DataVector, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Sul(4_st);
  assign_unique_datavector_tensor_values(Sul);

  const Tensor<DataVector> SIi_contracted =
      TensorExpressions::evaluate(Sul(ti_I, ti_i));

  DataVector expected_SIi_sum(4, 0.0);
  for (size_t i = 0; i < 3; i++) {
    expected_SIi_sum += Sul.get(i, i);
  }
  CHECK(SIi_contracted.get() == expected_SIi_sum);

  // Contract rank 2 (lower, upper) tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Rlu{};
  std::iota(Rlu.begin(), Rlu.end(), 0.0);

  const Tensor<double> RgG_contracted =
      TensorExpressions::evaluate(Rlu(ti_g, ti_G));

  double expected_RgG_sum = 0.0;
  for (size_t g = 0; g < 4; g++) {
    expected_RgG_sum += Rlu.get(g, g);
  }
  CHECK(RgG_contracted.get() == expected_RgG_sum);

  Tensor<DataVector, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Slu(2_st);
  assign_unique_datavector_tensor_values(Slu);

  const Tensor<DataVector> SgG_contracted =
      TensorExpressions::evaluate(Slu(ti_g, ti_G));

  DataVector expected_SgG_sum(2, 0.0);
  for (size_t g = 0; g < 4; g++) {
    expected_SgG_sum += Slu.get(g, g);
  }
  CHECK(SgG_contracted.get() == expected_SgG_sum);

  // Contract first and second indices of nonsymmetric rank 3 (upper, lower,
  // lower) tensor to rank 1 tensor
  Tensor<double, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rlul{};
  std::iota(Rlul.begin(), Rlul.end(), 0.0);

  const Tensor<double, Symmetry<1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      RiIj_contracted =
          TensorExpressions::evaluate<ti_j_t>(Rlul(ti_i, ti_I, ti_j));

  for (size_t j = 0; j < 4; j++) {
    double expected_sum = 0.0;
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Rlul.get(i, i, j);
    }
    CHECK(RiIj_contracted.get(j) == expected_sum);
  }

  Tensor<DataVector, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Slul(4_st);
  assign_unique_datavector_tensor_values(Slul);

  const Tensor<DataVector, Symmetry<1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      SiIj_contracted =
          TensorExpressions::evaluate<ti_j_t>(Slul(ti_i, ti_I, ti_j));

  for (size_t j = 0; j < 4; j++) {
    DataVector expected_sum(4, 0.0);
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Slul.get(i, i, j);
    }
    CHECK(SiIj_contracted.get(j) == expected_sum);
  }

  // Contract first and third indices of <2, 2, 1> symmetry rank 3 (upper,
  // upper, lower) tensor to rank 1 tensor
  Tensor<double, Symmetry<2, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Ruul{};
  std::iota(Ruul.begin(), Ruul.end(), 0.0);

  const Tensor<double, Symmetry<1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RJLj_contracted =
          TensorExpressions::evaluate<ti_L_t>(Ruul(ti_J, ti_L, ti_j));

  for (size_t l = 0; l < 3; l++) {
    double expected_sum = 0.0;
    for (size_t j = 0; j < 3; j++) {
      expected_sum += Ruul.get(j, l, j);
    }
    CHECK(RJLj_contracted.get(l) == expected_sum);
  }

  Tensor<DataVector, Symmetry<2, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Suul(3_st);
  assign_unique_datavector_tensor_values(Suul);

  const Tensor<DataVector, Symmetry<1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      SJLj_contracted =
          TensorExpressions::evaluate<ti_L_t>(Suul(ti_J, ti_L, ti_j));

  for (size_t l = 0; l < 3; l++) {
    DataVector expected_sum(3_st, 0.0);
    for (size_t j = 0; j < 3; j++) {
      expected_sum += Suul.get(j, l, j);
    }
    CHECK(SJLj_contracted.get(l) == expected_sum);
  }

  // Contract second and third indices of <2, 1, 2> symmetry rank 3 (upper,
  // lower, upper) tensor to rank 1 tensor
  Tensor<double, Symmetry<2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Rulu{};
  std::iota(Rulu.begin(), Rulu.end(), 0.0);

  const Tensor<double, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      RBfF_contracted =
          TensorExpressions::evaluate<ti_B_t>(Rulu(ti_B, ti_f, ti_F));

  for (size_t b = 0; b < 4; b++) {
    double expected_sum = 0.0;
    for (size_t f = 0; f < 4; f++) {
      expected_sum += Rulu.get(b, f, f);
    }
    CHECK(RBfF_contracted.get(b) == expected_sum);
  }

  Tensor<DataVector, Symmetry<2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Sulu(2_st);
  assign_unique_datavector_tensor_values(Sulu);

  const Tensor<DataVector, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      SBfF_contracted =
          TensorExpressions::evaluate<ti_B_t>(Sulu(ti_B, ti_f, ti_F));

  for (size_t b = 0; b < 4; b++) {
    DataVector expected_sum(2, 0.0);
    for (size_t f = 0; f < 4; f++) {
      expected_sum += Sulu.get(b, f, f);
    }
    CHECK(SBfF_contracted.get(b) == expected_sum);
  }

  // Contract first and second indices of nonsymmetric rank 4 (lower, upper,
  // upper, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rluul{};
  std::iota(Rluul.begin(), Rluul.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      RiIKj_contracted = TensorExpressions::evaluate<ti_K_t, ti_j_t>(
          Rluul(ti_i, ti_I, ti_K, ti_j));

  for (size_t k = 0; k < 4; k++) {
    for (size_t j = 0; j < 3; j++) {
      double expected_sum = 0.0;
      for (size_t i = 0; i < 3; i++) {
        expected_sum += Rluul.get(i, i, k, j);
      }
      CHECK(RiIKj_contracted.get(k, j) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Sluul(3_st);
  assign_unique_datavector_tensor_values(Sluul);

  const Tensor<DataVector, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      SiIKj_contracted = TensorExpressions::evaluate<ti_K_t, ti_j_t>(
          Sluul(ti_i, ti_I, ti_K, ti_j));

  for (size_t k = 0; k < 4; k++) {
    for (size_t j = 0; j < 3; j++) {
      DataVector expected_sum(3, 0.0);
      for (size_t i = 0; i < 3; i++) {
        expected_sum += Sluul.get(i, i, k, j);
      }
      CHECK(SiIKj_contracted.get(k, j) == expected_sum);
    }
  }

  // Contract first and third indices of nonsymmetric rank 4 (upper, upper,
  // lower, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      Ruull{};
  std::iota(Ruull.begin(), Ruull.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      RABac_contracted = TensorExpressions::evaluate<ti_B_t, ti_c_t>(
          Ruull(ti_A, ti_B, ti_a, ti_c));

  for (size_t b = 0; b < 4; b++) {
    for (size_t c = 0; c < 5; c++) {
      double expected_sum = 0.0;
      for (size_t a = 0; a < 5; a++) {
        expected_sum += Ruull.get(a, b, a, c);
      }
      CHECK(RABac_contracted.get(b, c) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      Suull(1_st);
  assign_unique_datavector_tensor_values(Suull);

  const Tensor<DataVector, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      SABac_contracted = TensorExpressions::evaluate<ti_B_t, ti_c_t>(
          Suull(ti_A, ti_B, ti_a, ti_c));

  for (size_t b = 0; b < 4; b++) {
    for (size_t c = 0; c < 5; c++) {
      DataVector expected_sum(1, 0.0);
      for (size_t a = 0; a < 5; a++) {
        expected_sum += Suull.get(a, b, a, c);
      }
      CHECK(SABac_contracted.get(b, c) == expected_sum);
    }
  }

  // Contract first and fourth indices of <3, 2, 3, 1> symmetry rank 4 (upper,
  // upper, upper, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<3, 2, 3, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Ruuul{};
  std::iota(Ruuul.begin(), Ruuul.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RLJIl_contracted = TensorExpressions::evaluate<ti_J_t, ti_I_t>(
          Ruuul(ti_L, ti_J, ti_I, ti_l));

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 3; i++) {
      double expected_sum = 0.0;
      for (size_t l = 0; l < 3; l++) {
        expected_sum += Ruuul.get(l, j, i, l);
      }
      CHECK(RLJIl_contracted.get(j, i) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<3, 2, 3, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Suuul(2_st);
  assign_unique_datavector_tensor_values(Suuul);

  const Tensor<DataVector, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      SLJIl_contracted = TensorExpressions::evaluate<ti_J_t, ti_I_t>(
          Suuul(ti_L, ti_J, ti_I, ti_l));

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 3; i++) {
      DataVector expected_sum(2, 0.0);
      for (size_t l = 0; l < 3; l++) {
        expected_sum += Suuul.get(l, j, i, l);
      }
      CHECK(SLJIl_contracted.get(j, i) == expected_sum);
    }
  }

  // Contract second and third indices of <2, 2, 1, 2> symmetry rank 4 (upper,
  // upper, lower, upper) tensor to rank 2 tensor
  Tensor<double, Symmetry<2, 2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ruulu{};
  std::iota(Ruulu.begin(), Ruulu.end(), 0.0);

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      REDdA_contracted = TensorExpressions::evaluate<ti_E_t, ti_A_t>(
          Ruulu(ti_E, ti_D, ti_d, ti_A));

  for (size_t e = 0; e < 4; e++) {
    for (size_t a = 0; a < 4; a++) {
      double expected_sum = 0.0;
      for (size_t d = 0; d < 4; d++) {
        expected_sum += Ruulu.get(e, d, d, a);
      }
      CHECK(REDdA_contracted.get(e, a) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<2, 2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Suulu(3_st);
  assign_unique_datavector_tensor_values(Suulu);

  const Tensor<DataVector, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      SEDdA_contracted = TensorExpressions::evaluate<ti_E_t, ti_A_t>(
          Suulu(ti_E, ti_D, ti_d, ti_A));

  for (size_t e = 0; e < 4; e++) {
    for (size_t a = 0; a < 4; a++) {
      DataVector expected_sum(3, 0.0);
      for (size_t d = 0; d < 4; d++) {
        expected_sum += Suulu.get(e, d, d, a);
      }
      CHECK(SEDdA_contracted.get(e, a) == expected_sum);
    }
  }

  // Contract second and fourth indices of nonsymmetric rank 4 (lower, upper,
  // lower, lower) tensor to rank 2 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rlull{};
  std::iota(Rlull.begin(), Rlull.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      RkJij_contracted = TensorExpressions::evaluate<ti_k_t, ti_i_t>(
          Rlull(ti_k, ti_J, ti_i, ti_j));

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 4; i++) {
      double expected_sum = 0.0;
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Rlull.get(k, j, i, j);
      }
      CHECK(RkJij_contracted.get(k, i) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Slull(2_st);
  assign_unique_datavector_tensor_values(Slull);

  const Tensor<DataVector, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      SkJij_contracted = TensorExpressions::evaluate<ti_k_t, ti_i_t>(
          Slull(ti_k, ti_J, ti_i, ti_j));

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 4; i++) {
      DataVector expected_sum(2, 0.0);
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Slull.get(k, j, i, j);
      }
      CHECK(SkJij_contracted.get(k, i) == expected_sum);
    }
  }

  // Contract third and fourth indices of <3, 2, 2, 1> symmetry rank 4 (upper,
  // lower, lower, upper) tensor to rank 2 tensor
  Tensor<double, Symmetry<3, 2, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Rullu{};
  std::iota(Rullu.begin(), Rullu.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      RFcgG_contracted = TensorExpressions::evaluate<ti_F_t, ti_c_t>(
          Rullu(ti_F, ti_c, ti_g, ti_G));

  for (size_t f = 0; f < 5; f++) {
    for (size_t c = 0; c < 4; c++) {
      double expected_sum = 0.0;
      for (size_t g = 0; g < 4; g++) {
        expected_sum += Rullu.get(f, c, g, g);
      }
      CHECK(RFcgG_contracted.get(f, c) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<3, 2, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Sullu(1_st);
  assign_unique_datavector_tensor_values(Sullu);

  const Tensor<DataVector, Symmetry<2, 1>,
               index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      SFcgG_contracted = TensorExpressions::evaluate<ti_F_t, ti_c_t>(
          Sullu(ti_F, ti_c, ti_g, ti_G));

  for (size_t f = 0; f < 5; f++) {
    for (size_t c = 0; c < 4; c++) {
      DataVector expected_sum(1, 0.0);
      for (size_t g = 0; g < 4; g++) {
        expected_sum += Sullu.get(f, c, g, g);
      }
      CHECK(SFcgG_contracted.get(f, c) == expected_sum);
    }
  }

  // Contract first and fourth indices of <2, 1, 1, 1> symmetry rank 4 (upper,
  // lower, lower, lower) tensor to rank 2 tensor and reorder indices
  Tensor<double, Symmetry<2, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rulll{};
  std::iota(Rulll.begin(), Rulll.end(), 0.0);

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      RAdba_contracted_to_bd = TensorExpressions::evaluate<ti_b_t, ti_d_t>(
          Rulll(ti_A, ti_d, ti_b, ti_a));

  for (size_t b = 0; b < 4; b++) {
    for (size_t d = 0; d < 4; d++) {
      double expected_sum = 0.0;
      for (size_t a = 0; a < 4; a++) {
        expected_sum += Rulll.get(a, d, b, a);
      }
      CHECK(RAdba_contracted_to_bd.get(b, d) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<2, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sulll(2_st);
  assign_unique_datavector_tensor_values(Sulll);

  const Tensor<DataVector, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      SAdba_contracted_to_bd = TensorExpressions::evaluate<ti_b_t, ti_d_t>(
          Sulll(ti_A, ti_d, ti_b, ti_a));

  for (size_t b = 0; b < 4; b++) {
    for (size_t d = 0; d < 4; d++) {
      DataVector expected_sum(2, 0.0);
      for (size_t a = 0; a < 4; a++) {
        expected_sum += Sulll.get(a, d, b, a);
      }
      CHECK(SAdba_contracted_to_bd.get(b, d) == expected_sum);
    }
  }

  // Contract second and third indices of <4, 3, 2, 1> symmetry rank 4 (lower,
  // lower, upper, lower) tensor to rank 2 tensor and reorder indices
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rllul{};
  std::iota(Rllul.begin(), Rllul.end(), 0.0);

  const Tensor<double, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      RljJi_contracted_to_il = TensorExpressions::evaluate<ti_i_t, ti_l_t>(
          Rllul(ti_l, ti_j, ti_J, ti_i));

  for (size_t i = 0; i < 4; i++) {
    for (size_t l = 0; l < 3; l++) {
      double expected_sum = 0.0;
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Rllul.get(l, j, j, i);
      }
      CHECK(RljJi_contracted_to_il.get(i, l) == expected_sum);
    }
  }

  Tensor<DataVector, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Sllul(2_st);
  assign_unique_datavector_tensor_values(Sllul);

  const Tensor<DataVector, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      SljJi_contracted_to_il = TensorExpressions::evaluate<ti_i_t, ti_l_t>(
          Sllul(ti_l, ti_j, ti_J, ti_i));

  for (size_t i = 0; i < 4; i++) {
    for (size_t l = 0; l < 3; l++) {
      DataVector expected_sum(2, 0.0);
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Sllul.get(l, j, j, i);
      }
      CHECK(SljJi_contracted_to_il.get(i, l) == expected_sum);
    }
  }

  // Contract first and second indices and third and fourth indices of
  // nonsymmetric rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rulul{};
  std::iota(Rulul.begin(), Rulul.end(), 0.0);

  const Tensor<double> RKkLl_contracted =
      TensorExpressions::evaluate(Rulul(ti_K, ti_k, ti_L, ti_l));

  double expected_RKkLl_sum = 0.0;
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 4; l++) {
      expected_RKkLl_sum += Rulul.get(k, k, l, l);
    }
  }
  CHECK(RKkLl_contracted.get() == expected_RKkLl_sum);

  Tensor<DataVector, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Sulul(1_st);
  assign_unique_datavector_tensor_values(Sulul);

  const Tensor<DataVector> SKkLl_contracted =
      TensorExpressions::evaluate(Sulul(ti_K, ti_k, ti_L, ti_l));

  DataVector expected_SKkLl_sum(1, 0.0);
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 4; l++) {
      expected_SKkLl_sum += Sulul.get(k, k, l, l);
    }
  }
  CHECK(SKkLl_contracted.get() == expected_SKkLl_sum);

  // Contract first and third indices and second and fourth indices of
  // <2, 2, 1, 1> symmetry rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Rlluu{};
  std::iota(Rlluu.begin(), Rlluu.end(), 0.0);

  const Tensor<double> RcaCA_contracted =
      TensorExpressions::evaluate(Rlluu(ti_c, ti_a, ti_C, ti_A));

  double expected_RcaCA_sum = 0.0;
  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      expected_RcaCA_sum += Rlluu.get(c, a, c, a);
    }
  }
  CHECK(RcaCA_contracted.get() == expected_RcaCA_sum);

  Tensor<DataVector, Symmetry<2, 2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Slluu(2_st);
  assign_unique_datavector_tensor_values(Slluu);

  const Tensor<DataVector> ScaCA_contracted =
      TensorExpressions::evaluate(Slluu(ti_c, ti_a, ti_C, ti_A));

  DataVector expected_ScaCA_sum(2, 0.0);
  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      expected_ScaCA_sum += Slluu.get(c, a, c, a);
    }
  }
  CHECK(ScaCA_contracted.get() == expected_ScaCA_sum);

  // Contract first and fourth indices and second and third indices of
  // <2, 1, 2, 1> symmetry rank 4 tensor to rank 0 tensor
  Tensor<double, Symmetry<2, 1, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Rlulu{};
  std::iota(Rlulu.begin(), Rlulu.end(), 0.0);

  const Tensor<double> RjIiJ_contracted =
      TensorExpressions::evaluate(Rlulu(ti_j, ti_I, ti_i, ti_J));

  double expected_RjIiJ_sum = 0.0;
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_RjIiJ_sum += Rlulu.get(j, i, i, j);
    }
  }
  CHECK(RjIiJ_contracted.get() == expected_RjIiJ_sum);

  Tensor<DataVector, Symmetry<2, 1, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Slulu(2_st);
  assign_unique_datavector_tensor_values(Slulu);

  const Tensor<DataVector> SjIiJ_contracted =
      TensorExpressions::evaluate(Slulu(ti_j, ti_I, ti_i, ti_J));

  DataVector expected_SjIiJ_sum(2, 0.0);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_SjIiJ_sum += Slulu.get(j, i, i, j);
    }
  }
  CHECK(SjIiJ_contracted.get() == expected_SjIiJ_sum);
}
