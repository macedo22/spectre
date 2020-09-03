// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/Tensor/Expressions/ComputeRhsTensorIndexTestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateTestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.ComputeRhsTensorIndex",
                  "[DataStructures][Unit]") {

  // Rank 0
  test_compute_rhs_tensor_index_rank_0<double>();

  // Rank 1 spacetime
  test_compute_rhs_tensor_index_rank_1<double, SpacetimeIndex, UpLo::Lo>(ti_a);
  test_compute_rhs_tensor_index_rank_1<double, SpacetimeIndex, UpLo::Lo>(ti_b);
  test_compute_rhs_tensor_index_rank_1<double, SpacetimeIndex, UpLo::Up>(ti_A);
  test_compute_rhs_tensor_index_rank_1<double, SpacetimeIndex, UpLo::Up>(ti_B);

  // Rank 1 spatial
  test_compute_rhs_tensor_index_rank_1<double, SpatialIndex, UpLo::Lo>(ti_i);
  test_compute_rhs_tensor_index_rank_1<double, SpatialIndex, UpLo::Lo>(ti_j);
  test_compute_rhs_tensor_index_rank_1<double, SpatialIndex, UpLo::Up>(ti_I);
  test_compute_rhs_tensor_index_rank_1<double, SpatialIndex, UpLo::Up>(ti_J);

  // Rank 2 nonsymmetric, spacetime only
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_a, ti_b);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_A, ti_B);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_d, ti_c);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_D, ti_C);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(ti_G, ti_b);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_f, ti_G);

  // Rank 2 nonsymmetric, spatial only
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_i, ti_j);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_I, ti_J);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_j, ti_i);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_J, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_i, ti_J);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(ti_I, ti_j);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_j, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(ti_J, ti_i);

  // Rank 2 nonsymmetric, spacetime and spatial mixed
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_c, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(ti_A, ti_i);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(ti_J, ti_a);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_i, ti_B);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_e, ti_j);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_i, ti_d);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_C, ti_I);
  test_compute_rhs_tensor_index_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_J, ti_A);

  // Rank 2 symmetric, spacetime
  test_compute_rhs_tensor_index_rank_2_symmetric<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_a, ti_d);
  test_compute_rhs_tensor_index_rank_2_symmetric<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_G, ti_B);

  // Rank 2 symmetric, spatial
  test_compute_rhs_tensor_index_rank_2_symmetric<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_j, ti_i);
  test_compute_rhs_tensor_index_rank_2_symmetric<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_I, ti_J);

  // Rank 3 nonsymmetric, spacetime
  test_compute_rhs_tensor_index_rank_3_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo,
      UpLo::Up>(ti_D, ti_j, ti_B);

  // Rank 3 ab symmetry
  test_compute_rhs_tensor_index_rank_3_ab_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Lo, UpLo::Up>(ti_b, ti_a, ti_C);

  // Rank 3 ac symmetry
  test_compute_rhs_tensor_index_rank_3_ac_symmetry<
      double, SpatialIndex, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Lo,
      UpLo::Lo>(ti_i, ti_f, ti_j);

  // Rank 3 bc symmetry
  test_compute_rhs_tensor_index_rank_3_bc_symmetry<
      double, SpacetimeIndex, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up,
      UpLo::Up>(ti_d, ti_J, ti_I);

  // Rank 3 abc symmetry
  test_compute_rhs_tensor_index_rank_3_abc_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, SpacetimeIndex, UpLo::Lo,
      UpLo::Lo, UpLo::Lo>(ti_f, ti_d, ti_a);

  // Rank 4 nonsymmetric
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>(ti_c, ti_J, ti_i,
                                                            ti_A);

  // Rank 4 bd symmetry
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<3, 2, 1, 2>,
      index_list<SpatialIndex<1, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>(ti_I, ti_A,
                                                                ti_l, ti_F);

  // Rank 4 acd symmetry
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<2, 1, 2, 2>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>(ti_l, ti_J, ti_i,
                                                            ti_k);

  // Rank 4 abcd symmetry
  test_compute_rhs_tensor_index_rank_4<
      double, Symmetry<1, 1, 1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>(ti_j, ti_i, ti_k,
                                                              ti_l);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  // Rank 0
  test_evaluate_rank_0<double>(-7.31);

  // Rank 1 spacetime
  test_evaluate_rank_1<double, SpacetimeIndex, UpLo::Lo>(ti_a);
  test_evaluate_rank_1<double, SpacetimeIndex, UpLo::Lo>(ti_b);
  test_evaluate_rank_1<double, SpacetimeIndex, UpLo::Up>(ti_A);
  test_evaluate_rank_1<double, SpacetimeIndex, UpLo::Up>(ti_B);

  // Rank 1 spatial
  test_evaluate_rank_1<double, SpatialIndex, UpLo::Lo>(ti_i);
  test_evaluate_rank_1<double, SpatialIndex, UpLo::Lo>(ti_j);
  test_evaluate_rank_1<double, SpatialIndex, UpLo::Up>(ti_I);
  test_evaluate_rank_1<double, SpatialIndex, UpLo::Up>(ti_J);

  // Rank 2 nonsymmetric, spacetime only
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   UpLo::Lo, UpLo::Lo>(ti_a, ti_b);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   UpLo::Up, UpLo::Up>(ti_A, ti_B);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   UpLo::Lo, UpLo::Lo>(ti_d, ti_c);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   UpLo::Up, UpLo::Up>(ti_D, ti_C);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   UpLo::Up, UpLo::Lo>(ti_G, ti_b);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   UpLo::Lo, UpLo::Up>(ti_f, ti_G);

  // Rank 2 nonsymmetric, spatial only
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Lo,
                                   UpLo::Lo>(ti_i, ti_j);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Up,
                                   UpLo::Up>(ti_I, ti_J);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Lo,
                                   UpLo::Lo>(ti_j, ti_i);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Up,
                                   UpLo::Up>(ti_J, ti_I);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Lo,
                                   UpLo::Up>(ti_i, ti_J);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Up,
                                   UpLo::Lo>(ti_I, ti_j);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Lo,
                                   UpLo::Up>(ti_j, ti_I);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpatialIndex, UpLo::Up,
                                   UpLo::Lo>(ti_J, ti_i);

  // Rank 2 nonsymmetric, spacetime and spatial mixed
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpatialIndex,
                                   UpLo::Lo, UpLo::Up>(ti_c, ti_I);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpatialIndex,
                                   UpLo::Up, UpLo::Lo>(ti_A, ti_i);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpacetimeIndex,
                                   UpLo::Up, UpLo::Lo>(ti_J, ti_a);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpacetimeIndex,
                                   UpLo::Lo, UpLo::Up>(ti_i, ti_B);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpatialIndex,
                                   UpLo::Lo, UpLo::Lo>(ti_e, ti_j);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpacetimeIndex,
                                   UpLo::Lo, UpLo::Lo>(ti_i, ti_d);
  test_evaluate_rank_2_no_symmetry<double, SpacetimeIndex, SpatialIndex,
                                   UpLo::Up, UpLo::Up>(ti_C, ti_I);
  test_evaluate_rank_2_no_symmetry<double, SpatialIndex, SpacetimeIndex,
                                   UpLo::Up, UpLo::Up>(ti_J, ti_A);

  // Rank 2 symmetric, spacetime
  test_evaluate_rank_2_symmetric<double, SpacetimeIndex, SpacetimeIndex,
                                 UpLo::Lo, UpLo::Lo>(ti_a, ti_d);
  test_evaluate_rank_2_symmetric<double, SpacetimeIndex, SpacetimeIndex,
                                 UpLo::Up, UpLo::Up>(ti_G, ti_B);

  // Rank 2 symmetric, spatial
  test_evaluate_rank_2_symmetric<double, SpatialIndex, SpatialIndex, UpLo::Lo,
                                 UpLo::Lo>(ti_j, ti_i);
  test_evaluate_rank_2_symmetric<double, SpatialIndex, SpatialIndex, UpLo::Up,
                                 UpLo::Up>(ti_I, ti_J);

  // Rank 3 nonsymmetric
  test_evaluate_rank_3_no_symmetry<double, SpacetimeIndex, SpatialIndex,
                                   SpacetimeIndex, UpLo::Up, UpLo::Lo,
                                   UpLo::Up>(ti_D, ti_j, ti_B);

  // Rank 3 first and second indices symmetric
  test_evaluate_rank_3_ab_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                   SpacetimeIndex, UpLo::Lo, UpLo::Lo,
                                   UpLo::Up>(ti_b, ti_a, ti_C);

  // Rank 3 first and third indices symmetric
  test_evaluate_rank_3_ac_symmetry<double, SpatialIndex, SpacetimeIndex,
                                   SpatialIndex, UpLo::Lo, UpLo::Lo, UpLo::Lo>(
      ti_i, ti_f, ti_j);

  // Rank 3 second and third indices symmetric
  test_evaluate_rank_3_bc_symmetry<double, SpacetimeIndex, SpatialIndex,
                                   SpatialIndex, UpLo::Lo, UpLo::Up, UpLo::Up>(
      ti_d, ti_J, ti_I);

  // Rank 3 symmetric
  test_evaluate_rank_3_abc_symmetry<double, SpacetimeIndex, SpacetimeIndex,
                                    SpacetimeIndex, UpLo::Lo, UpLo::Lo,
                                    UpLo::Lo>(ti_f, ti_d, ti_a);

  // Rank 4 nonsymmetric
  test_evaluate_rank_4<double, Symmetry<4, 3, 2, 1>,
                       index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                                  SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                                  SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                                  SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>(
      ti_b, ti_A, ti_k, ti_l);

  // Rank 4 second and third indices symmetric
  test_evaluate_rank_4<double, Symmetry<3, 2, 2, 1>,
                       index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                                  SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                                  SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                                  SpatialIndex<1, UpLo::Lo, Frame::Grid>>>(
      ti_G, ti_d, ti_a, ti_j);

  // Rank 4 first, second, and fourth indices symmetric
  test_evaluate_rank_4<double, Symmetry<2, 2, 1, 2>,
                       index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                                  SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                                  SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                                  SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>(
      ti_j, ti_i, ti_k, ti_l);

  // Rank 4 symmetric
  test_evaluate_rank_4<double, Symmetry<1, 1, 1, 1>,
                       index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                                  SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                                  SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                                  SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>(
      ti_F, ti_A, ti_C, ti_D);
}
