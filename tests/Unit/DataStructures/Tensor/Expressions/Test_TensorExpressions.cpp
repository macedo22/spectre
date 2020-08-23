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
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank0TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank1TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank3TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexRank4TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank0TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank1TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank3TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank4TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ComputeRhsTensorIndex",
                  "[DataStructures][Unit]") {

  // Rank 0
  test_compute_rhs_tensor_index_rank_0<double>();

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

  // Rank 3 nonsymmetric, spacetime
  test_compute_rhs_tensor_index_rank_3_no_symmetry<
      double, ti_D_t, ti_j_t, ti_B_t, SpacetimeIndex, SpatialIndex,
      SpacetimeIndex, UpLo::Up, UpLo::Lo, UpLo::Up>(ti_D, ti_j, ti_B);

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

  // Rank 4 nonsymmetric
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>,
      ti_c_t, ti_J_t, ti_i_t, ti_A_t> (
          ti_c, ti_J, ti_i, ti_A, 3, 2, 1, 2);

  // Rank 4 bd symmetry
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<3, 2, 1, 2>,
      index_list<SpatialIndex<1, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti_I_t, ti_A_t, ti_l_t, ti_F_t> (
          ti_I, ti_A, ti_l, ti_F, 1, 3, 2, 3);

  // Rank 4 acd symmetry
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<2, 1, 2, 2>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti_l_t, ti_J_t, ti_i_t, ti_k_t> (
          ti_l, ti_J, ti_i, ti_k, 3, 3, 3, 3);

  // Rank 4 abcd symmetry
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<1, 1, 1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti_j_t, ti_i_t, ti_k_t, ti_l_t> (
          ti_j, ti_i, ti_k, ti_l, 3, 3, 3, 3);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.StorageGet",
                  "[DataStructures][Unit]") {
  // Rank 0
  test_storage_get_rank_0<double>(-7.31);

  // Rank 1 spacetime
  test_storage_get_rank_1<
      double, ti_a_t, SpacetimeIndex, UpLo::Lo>(ti_a);
  test_storage_get_rank_1<
      double, ti_b_t, SpacetimeIndex, UpLo::Lo>(ti_b);
  test_storage_get_rank_1<
      double, ti_A_t, SpacetimeIndex, UpLo::Up>(ti_A);
  test_storage_get_rank_1<
      double, ti_B_t, SpacetimeIndex, UpLo::Up>(ti_B);

  // Rank 1 spatial
  test_storage_get_rank_1<
      double, ti_i_t, SpatialIndex, UpLo::Lo>(ti_i);
  test_storage_get_rank_1<
      double, ti_j_t, SpatialIndex, UpLo::Lo>(ti_j);
  test_storage_get_rank_1<
      double, ti_I_t, SpatialIndex, UpLo::Up>(ti_I);
  test_storage_get_rank_1<
      double, ti_J_t, SpatialIndex, UpLo::Up>(ti_J);

  // Rank 2 nonsymmetric, spacetime only
  test_storage_get_rank_2_no_symmetry<double, ti_a_t, ti_b_t, SpacetimeIndex,
                                      SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_a,
                                                                          ti_b);
  test_storage_get_rank_2_no_symmetry<
      double, ti_A_t, ti_B_t, SpacetimeIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Up>(ti_A, ti_B);
  test_storage_get_rank_2_no_symmetry<
      double, ti_d_t, ti_c_t, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Lo>(ti_d, ti_c);
  test_storage_get_rank_2_no_symmetry<
      double, ti_D_t, ti_C_t, SpacetimeIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Up>(ti_D, ti_C);
  test_storage_get_rank_2_no_symmetry<
      double, ti_G_t, ti_b_t, SpacetimeIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Lo>(ti_G, ti_b);
  test_storage_get_rank_2_no_symmetry<
      double, ti_f_t, ti_G_t, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Up>(ti_f, ti_G);

  // Rank 2 nonsymmetric, spatial only
  test_storage_get_rank_2_no_symmetry<
      double, ti_i_t, ti_j_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_i, ti_j);
  test_storage_get_rank_2_no_symmetry<
      double, ti_I_t, ti_J_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_I, ti_J);
   test_storage_get_rank_2_no_symmetry<
      double, ti_j_t, ti_i_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_j, ti_i);
  test_storage_get_rank_2_no_symmetry<
      double, ti_J_t, ti_I_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_J, ti_I);
  test_storage_get_rank_2_no_symmetry<
      double, ti_i_t, ti_J_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(
      ti_i, ti_J);
  test_storage_get_rank_2_no_symmetry<
      double, ti_I_t, ti_j_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(
      ti_I, ti_j);
  test_storage_get_rank_2_no_symmetry<
      double, ti_j_t, ti_I_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(
      ti_j, ti_I);
  test_storage_get_rank_2_no_symmetry<
      double, ti_J_t, ti_i_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(
      ti_J, ti_i);

  // Rank 2 nonsymmetric, spacetime and spatial mixed
  test_storage_get_rank_2_no_symmetry<
      double, ti_c_t, ti_I_t, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(
      ti_c, ti_I);
  test_storage_get_rank_2_no_symmetry<
      double, ti_A_t, ti_i_t, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(
      ti_A, ti_i);
  test_storage_get_rank_2_no_symmetry<
      double, ti_J_t, ti_a_t, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(
      ti_J, ti_a);
  test_storage_get_rank_2_no_symmetry<
      double, ti_i_t, ti_B_t, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(
      ti_i, ti_B);
  test_storage_get_rank_2_no_symmetry<
      double, ti_e_t, ti_j_t, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_e, ti_j);
  test_storage_get_rank_2_no_symmetry<
      double, ti_i_t, ti_d_t, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(
      ti_i, ti_d);
  test_storage_get_rank_2_no_symmetry<
      double, ti_C_t, ti_I_t, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_C, ti_I);
  test_storage_get_rank_2_no_symmetry<
      double, ti_J_t, ti_A_t, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(
      ti_J, ti_A);

  // Rank 2 symmetric, spacetime
  test_storage_get_rank_2_symmetric<double, ti_a_t, ti_d_t,
                                                 SpacetimeIndex, SpacetimeIndex,
                                                 UpLo::Lo, UpLo::Lo>(ti_a,
                                                                     ti_d);
  test_storage_get_rank_2_symmetric<double, ti_G_t, ti_B_t,
                                                 SpacetimeIndex, SpacetimeIndex,
                                                 UpLo::Up, UpLo::Up>(ti_G,
                                                                     ti_B);

  // Rank 2 symmetric, spatial
  test_storage_get_rank_2_symmetric<
      double, ti_j_t, ti_i_t, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(
      ti_j, ti_i);
  test_storage_get_rank_2_symmetric<
      double, ti_I_t, ti_J_t, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(
      ti_I, ti_J);

  // Rank 3 nonsymmetric
  test_storage_get_rank_3_no_symmetry<
      double, ti_D_t, ti_j_t, ti_B_t, SpacetimeIndex, SpatialIndex,
      SpacetimeIndex, UpLo::Up, UpLo::Lo, UpLo::Up>(ti_D, ti_j, ti_B);

  // Rank 3 ab symmetry
  test_storage_get_rank_3_ab_symmetry<
      double, ti_b_t, ti_a_t, ti_C_t, SpacetimeIndex, SpacetimeIndex,
      SpacetimeIndex, UpLo::Lo, UpLo::Lo, UpLo::Up>(ti_b, ti_a, ti_C);

  // Rank 3 ac symmetry
  test_storage_get_rank_3_ac_symmetry<
      double, ti_i_t, ti_f_t, ti_j_t, SpatialIndex, SpacetimeIndex,
      SpatialIndex, UpLo::Lo, UpLo::Lo, UpLo::Lo>(ti_i, ti_f, ti_j);

  // Rank 3 bc symmetry
  test_storage_get_rank_3_bc_symmetry<
      double, ti_d_t, ti_J_t, ti_I_t, SpacetimeIndex, SpatialIndex,
      SpatialIndex, UpLo::Lo, UpLo::Up, UpLo::Up>(ti_d, ti_J, ti_I);

  // Rank 3 abc symmetry
  test_storage_get_rank_3_abc_symmetry<
      double, ti_f_t, ti_d_t, ti_a_t, SpacetimeIndex, SpacetimeIndex,
      SpacetimeIndex, UpLo::Lo, UpLo::Lo, UpLo::Lo>(ti_f, ti_d, ti_a);

  // Rank 4 nonsymmetric
  test_storage_get_rank_4<
      double, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti_b_t, ti_A_t, ti_k_t, ti_l_t> (
          ti_b, ti_A, ti_k, ti_l, 3, 3, 1, 2);

  // Rank 4 bc symmetry
  test_storage_get_rank_4<
      double, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      ti_G_t, ti_d_t, ti_a_t, ti_j_t> (
          ti_G, ti_d, ti_a, ti_j, 2, 3, 3, 1);

  // Rank 4 abd symmetry
  test_storage_get_rank_4<
      double, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti_j_t, ti_i_t, ti_k_t, ti_l_t> (
          ti_j, ti_i, ti_k, ti_l, 3, 3, 3, 3);

  // Rank 4 abcd symmetry
  test_storage_get_rank_4<
      double, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti_F_t, ti_A_t, ti_C_t, ti_D_t> (
          ti_F, ti_A, ti_C, ti_D, 3, 3, 3, 3);

}
