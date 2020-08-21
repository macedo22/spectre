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
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/StorageGetRank3TestHelpers.hpp"
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

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.StorageGet",
                  "[DataStructures][Unit]") {
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

  // TODO - define the methods below
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
}
