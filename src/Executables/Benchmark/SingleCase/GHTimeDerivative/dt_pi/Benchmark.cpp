// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
constexpr size_t seed = 17;
constexpr size_t Dim = 3;
std::mt19937 generator(seed);
constexpr size_t num_grid_points = 1000;

void bench_dt_pi(benchmark::State& state) {  // NOLINT
  using dt_pi_type = tnsr::aa<DataVector, Dim>;
  using spacetime_deriv_gauge_function_type = tnsr::ab<DataVector, Dim>;
  using pi_two_normals_type = Scalar<DataVector>;
  using pi_type = tnsr::aa<DataVector, Dim>;
  using gamma0_type = Scalar<DataVector>;
  using normal_spacetime_one_form_type = tnsr::a<DataVector, Dim>;
  using gauge_constraint_type = tnsr::a<DataVector, Dim>;
  using spacetime_metric_type = tnsr::aa<DataVector, Dim>;
  using normal_dot_gauge_constraint_type = Scalar<DataVector>;
  using christoffel_second_kind_type = tnsr::Abb<DataVector, Dim>;
  using gauge_function_type = tnsr::a<DataVector, Dim>;
  using pi_2_up_type = tnsr::aB<DataVector, Dim>;
  using phi_1_up_type = tnsr::Iaa<DataVector, Dim>;
  using phi_3_up_type = tnsr::iaB<DataVector, Dim>;
  using christoffel_first_kind_3_up_type = tnsr::abC<DataVector, Dim>;
  using pi_one_normal_type = tnsr::a<DataVector, Dim>;
  using inverse_spatial_metric_type = tnsr::II<DataVector, Dim>;
  using d_phi_type = tnsr::ijaa<DataVector, Dim>;
  using lapse_type = Scalar<DataVector>;
  using gamma1gamma2_type = Scalar<DataVector>;
  using shift_dot_three_index_constraint_type = tnsr::aa<DataVector, Dim>;
  using shift_type = tnsr::I<DataVector, Dim>;
  using d_pi_type = tnsr::iaa<DataVector, Dim>;

  const DataVector used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: spacetime_deriv_gauge_function
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<spacetime_deriv_gauge_function_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: pi_two_normals
  const pi_two_normals_type pi_two_normals =
      make_with_random_values<pi_two_normals_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma0
  const gamma0_type gamma0 = make_with_random_values<gamma0_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: normal_spacetime_one_form
  const normal_spacetime_one_form_type normal_spacetime_one_form =
      make_with_random_values<normal_spacetime_one_form_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: gauge_constraint
  const gauge_constraint_type gauge_constraint =
      make_with_random_values<gauge_constraint_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: spacetime_metric
  const spacetime_metric_type spacetime_metric =
      make_with_random_values<spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: normal_dot_gauge_constraint
  const normal_dot_gauge_constraint_type normal_dot_gauge_constraint =
      make_with_random_values<normal_dot_gauge_constraint_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: christoffel_second_kind
  const christoffel_second_kind_type christoffel_second_kind =
      make_with_random_values<christoffel_second_kind_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: pi_2_up
  const pi_2_up_type pi_2_up = make_with_random_values<pi_2_up_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi_1_up
  const phi_1_up_type phi_1_up = make_with_random_values<phi_1_up_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi_3_up
  const phi_3_up_type phi_3_up = make_with_random_values<phi_3_up_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: christoffel_first_kind_3_up
  const christoffel_first_kind_3_up_type christoffel_first_kind_3_up =
      make_with_random_values<christoffel_first_kind_3_up_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: pi_one_normal
  const pi_one_normal_type pi_one_normal =
      make_with_random_values<pi_one_normal_type>(make_not_null(&generator),
                                                  make_not_null(&distribution),
                                                  used_for_size);

  // RHS: inverse_spatial_metric
  const inverse_spatial_metric_type inverse_spatial_metric =
      make_with_random_values<inverse_spatial_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1gamma2
  const gamma1gamma2_type gamma1gamma2 =
      make_with_random_values<gamma1gamma2_type>(make_not_null(&generator),
                                                 make_not_null(&distribution),
                                                 used_for_size);

  // RHS: shift_dot_three_index_constraint
  const shift_dot_three_index_constraint_type shift_dot_three_index_constraint =
      make_with_random_values<shift_dot_three_index_constraint_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // LHS: dt_pi
  dt_pi_type dt_pi(used_for_size);

  for (auto _ : state) {
    TensorExpressions::evaluate<ti_a, ti_b>(
        make_not_null(&dt_pi),
        (-spacetime_deriv_gauge_function(ti_a, ti_b) -
         spacetime_deriv_gauge_function(ti_b, ti_a) -
         0.5 * pi_two_normals() * pi(ti_a, ti_b) +
         gamma0() * (normal_spacetime_one_form(ti_a) * gauge_constraint(ti_b) +
                     normal_spacetime_one_form(ti_b) * gauge_constraint(ti_a)) -
         gamma0() * spacetime_metric(ti_a, ti_b) *
             normal_dot_gauge_constraint() +
         2.0 * christoffel_second_kind(ti_C, ti_a, ti_b) *
             gauge_function(ti_c) -
         2.0 * pi(ti_a, ti_c) * pi_2_up(ti_b, ti_C) +
         // Note : flipped ti_b and ti_a in next product so test passes
         // (symmetry asusmption issues)
         2.0 * phi_3_up(ti_i, ti_a, ti_C) * phi_1_up(ti_I, ti_b, ti_c) -
         2.0 * christoffel_first_kind_3_up(ti_a, ti_d, ti_C) *
             christoffel_first_kind_3_up(ti_b, ti_c, ti_D) -
         pi_one_normal(ti_j) * phi_1_up(ti_J, ti_a, ti_b) -
         inverse_spatial_metric(ti_J, ti_K) * d_phi(ti_j, ti_k, ti_a, ti_b)) *
                lapse() +
            gamma1gamma2() * shift_dot_three_index_constraint(ti_a, ti_b) +
            shift(ti_J) * d_pi(ti_j, ti_a, ti_b));
    benchmark::DoNotOptimize(dt_pi);
    benchmark::ClobberMemory();
  }
}

BENCHMARK(bench_dt_pi);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
