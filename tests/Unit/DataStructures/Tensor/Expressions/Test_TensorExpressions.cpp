// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank1TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank3TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

// include on multiple lines just to avoid line length warning
//#include
//"tests/Unit/DataStructures/Tensor/Expressions/
// ComputeRhsTensorIndex_Test_Helpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.TryToDoTensorStuff",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);

  // try to get the TensorIndexTypeA::value and TensorIndexTypeB::value
  // can also try to get the dimension of both indices, but I can
  // also just pass those in...
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.StorageGet",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);

  auto result1 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b));

  auto result2 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_b, ti_a));

  auto result3 = TensorExpressions::evaluate<ti_b_t, ti_a_t>(All(ti_a, ti_b));

  auto result4 = TensorExpressions::evaluate<ti_b_t, ti_a_t>(All(ti_b, ti_a));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(result1.get(i, j) == All.get(i, j));
      CHECK(result2.get(j, i) == All.get(i, j));
      CHECK(result3.get(j, i) == All.get(i, j));
      CHECK(result4.get(i, j) == All.get(i, j));

      CHECK(result1.get(j, i) == result2.get(i, j));
      CHECK(result1.get(j, i) == result2.get(i, j));
      CHECK(result1.get(i, j) == result4.get(i, j));
      CHECK(result2.get(i, j) == result3.get(i, j));
      CHECK(result2.get(j, i) == result4.get(i, j));
      CHECK(result3.get(j, i) == result4.get(i, j));
    }
  }

  Tensor<double, Symmetry<2, 1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Auul{};

  auto result5 = TensorExpressions::evaluate<ti_G_t, ti_B_t, ti_d_t>(
      Auul(ti_G, ti_B, ti_d));
  auto result6 = TensorExpressions::evaluate<ti_G_t, ti_B_t, ti_d_t>(
      Auul(ti_d, ti_G, ti_B));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; k++) {
        CHECK(result5.get(i, j, k) == Auul.get(i, j, k));
        CHECK(result6.get(j, k, i) == Auul.get(i, j, k));
        CHECK(result5.get(i, j, k) == result6.get(j, k, i));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ComputeRhsTensorIndex",
                  "[DataStructures][Unit]") {

  // Rank 1 spacetime
  test_compute_rhs_tensor_index_rank_1<
      double, ti_a_t, SpacetimeIndex, UpLo::Lo>(ti_a);
  test_compute_rhs_tensor_index_rank_1<
      double, ti_b_t, SpacetimeIndex, UpLo::Lo>(ti_b);
  test_compute_rhs_tensor_index_rank_1<
      double, ti_A_t, SpacetimeIndex, UpLo::Up>(ti_A);
  test_compute_rhs_tensor_index_rank_1<
      double, ti_B_t, SpacetimeIndex, UpLo::Up>(ti_B);

  // Rank 1 spatial
  test_compute_rhs_tensor_index_rank_1<
      double, ti_i_t, SpatialIndex, UpLo::Lo>(ti_i);
  test_compute_rhs_tensor_index_rank_1<
      double, ti_j_t, SpatialIndex, UpLo::Lo>(ti_j);
  test_compute_rhs_tensor_index_rank_1<
      double, ti_I_t, SpatialIndex, UpLo::Up>(ti_I);
  test_compute_rhs_tensor_index_rank_1<
      double, ti_J_t, SpatialIndex, UpLo::Up>(ti_J);


  // Rank 2 nonsymmetric, spacetime only
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_a_t, ti_b_t, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Lo>(ti_a, ti_b);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_A_t, ti_B_t, SpacetimeIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Up>(ti_A, ti_B);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_d_t, ti_c_t, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Lo>(ti_d, ti_c);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_D_t, ti_C_t, SpacetimeIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Up>(ti_D, ti_C);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_G_t, ti_b_t, SpacetimeIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Lo>(ti_G, ti_b);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_f_t, ti_G_t, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Up>(ti_f, ti_G);

  // Rank 2 nonsymmetric, spatial only
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_i_t, ti_j_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_i, ti_j);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_I_t, ti_J_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_I, ti_J);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_j_t, ti_i_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_j, ti_i);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_J_t, ti_I_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_J, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_i_t, ti_J_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(
      ti_i, ti_J);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_I_t, ti_j_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(
      ti_I, ti_j);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_j_t, ti_I_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(
      ti_j, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_J_t, ti_i_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(
      ti_J, ti_i);

  // Rank 2 nonsymmetric, spacetime and spatial mixed
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_c_t, ti_I_t, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(
      ti_c, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_A_t, ti_i_t, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(
      ti_A, ti_i);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_J_t, ti_a_t, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(
      ti_J, ti_a);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_i_t, ti_B_t, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(
      ti_i, ti_B);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_e_t, ti_j_t, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_e, ti_j);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_i_t, ti_d_t, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(
      ti_i, ti_d);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_C_t, ti_I_t, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_C, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, ti_J_t, ti_A_t, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(
      ti_J, ti_A);

  // Rank 2 symmetric, spacetime
  test_compute_rhs_tensor_index_rank_2_symmetric<double, ti_a_t, ti_d_t,
                                                 SpacetimeIndex, SpacetimeIndex,
                                                 UpLo::Lo, UpLo::Lo>(ti_a,
                                                                     ti_d);
  test_compute_rhs_tensor_index_rank_2_symmetric<double, ti_G_t, ti_B_t,
                                                 SpacetimeIndex, SpacetimeIndex,
                                                 UpLo::Up, UpLo::Up>(ti_G,
                                                                     ti_B);

  // Rank 2 symmetric, spatial
  test_compute_rhs_tensor_index_rank_2_symmetric<
      double, ti_j_t, ti_i_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_j, ti_i);
  test_compute_rhs_tensor_index_rank_2_symmetric<
      double, ti_I_t, ti_J_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_I, ti_J);

  // Rank 3 nonsymmetric
  test_compute_rhs_tensor_index_rank_3_no_symmetry<
      double, ti_D_t, ti_j_t, ti_B_t, SpacetimeIndex, SpatialIndex,
      SpacetimeIndex, UpLo::Up, UpLo::Lo, UpLo::Up>(ti_D, ti_j, ti_B);

  // TODO: The below doesn't work due to ti_a and ti_A trying to contract?
  /*test_compute_rhs_tensor_index_rank_3_ab_symmetry<
      double, ti_b_t, ti_a_t, ti_A_t, SpacetimeIndex, SpacetimeIndex,
      SpacetimeIndex, UpLo::Lo, UpLo::Lo, UpLo::Up>(ti_b, ti_a, ti_A);*/

  // Rank 3 ab symmetry
  test_compute_rhs_tensor_index_rank_3_ab_symmetry<
      double, ti_b_t, ti_a_t, ti_C_t, SpacetimeIndex, SpacetimeIndex,
      SpacetimeIndex, UpLo::Lo, UpLo::Lo, UpLo::Up>(ti_b, ti_a, ti_C);

  // Rank 3 ac symmetry
  test_compute_rhs_tensor_index_rank_3_ac_symmetry<
      double, ti_i_t, ti_f_t, ti_j_t, SpatialIndex, SpacetimeIndex,
      SpatialIndex, UpLo::Lo, UpLo::Lo, UpLo::Lo>(ti_i, ti_f, ti_j);

  // Rank 3 bc symmetry
  test_compute_rhs_tensor_index_rank_3_bc_symmetry<
      double, ti_d_t, ti_J_t, ti_I_t, SpacetimeIndex, SpatialIndex,
      SpatialIndex, UpLo::Lo, UpLo::Up, UpLo::Up>(ti_d, ti_J, ti_I);

  // Rank 3 abc symmetry
  test_compute_rhs_tensor_index_rank_3_abc_symmetry<
      double, ti_f_t, ti_d_t, ti_a_t, SpacetimeIndex, SpacetimeIndex,
      SpacetimeIndex, UpLo::Lo, UpLo::Lo, UpLo::Lo>(ti_f, ti_d, ti_a);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtract",
                  "[DataStructures][Unit]") {
  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{}, HHll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);
  std::iota(HHll.rbegin(), HHll.rend(), -Hll.size());
  /// [use_tensor_index]
  auto Gll = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b) +
                                                         Hll(ti_a, ti_b));
  auto Gll2 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b) +
                                                          Hll(ti_b, ti_a));
  auto Gll3 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(
      All(ti_a, ti_b) + Hll(ti_b, ti_a) + All(ti_b, ti_a) - Hll(ti_b, ti_a));
  /// [use_tensor_index]
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(Gll.get(i, j) == All.get(i, j) + Hll.get(i, j));
      CHECK(Gll2.get(i, j) == All.get(i, j) + Hll.get(j, i));
      CHECK(Gll3.get(i, j) == 2.0 * All.get(i, j));
    }
  }
  // Test 3 indices add subtract
  Tensor<double, Symmetry<1, 1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll{};
  std::iota(Alll.begin(), Alll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hlll{}, HHlll{};
  std::iota(Hlll.begin(), Hlll.end(), 0.0);
  std::iota(HHlll.rbegin(), HHlll.rend(), -Hlll.size());
  auto Glll = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_a, ti_b, ti_c));
  auto Glll2 = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c));
  auto Glll3 = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c) + Alll(ti_b, ti_a, ti_c) -
      Hlll(ti_b, ti_a, ti_c));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        CHECK(Glll.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(i, j, k));
        CHECK(Glll2.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(j, i, k));
        CHECK(Glll3.get(i, j, k) == 2.0 * Alll.get(i, j, k));
      }
    }
  }
}
