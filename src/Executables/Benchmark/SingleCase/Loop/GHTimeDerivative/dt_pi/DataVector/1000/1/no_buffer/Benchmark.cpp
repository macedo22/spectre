// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Executables/Benchmark/SingleCase/Loop/GHTimeDerivative/dt_pi/BenchmarkImpl.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
constexpr size_t seed = 17;
std::mt19937 generator(seed);

// tensor types in tensor equation being benchmarked
using DataType = DataVector;
static constexpr size_t Dim = 1;
constexpr size_t num_grid_points = 1000;

using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
using spacetime_deriv_gauge_function_type =
    typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
using pi_type = typename BenchmarkImpl::pi_type;
using gamma0_type = typename BenchmarkImpl::gamma0_type;
using normal_spacetime_one_form_type =
    typename BenchmarkImpl::normal_spacetime_one_form_type;
using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
using normal_dot_gauge_constraint_type =
    typename BenchmarkImpl::normal_dot_gauge_constraint_type;
using christoffel_second_kind_type =
    typename BenchmarkImpl::christoffel_second_kind_type;
using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
using christoffel_first_kind_3_up_type =
    typename BenchmarkImpl::christoffel_first_kind_3_up_type;
using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
using inverse_spatial_metric_type =
    typename BenchmarkImpl::inverse_spatial_metric_type;
using d_phi_type = typename BenchmarkImpl::d_phi_type;
using lapse_type = typename BenchmarkImpl::lapse_type;
using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
using shift_dot_three_index_constraint_type =
    typename BenchmarkImpl::shift_dot_three_index_constraint_type;
using shift_type = typename BenchmarkImpl::shift_type;
using d_pi_type = typename BenchmarkImpl::d_pi_type;

void bench(benchmark::State& state) {  // NOLINT
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
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
    BenchmarkImpl::apply(make_not_null(&dt_pi), spacetime_deriv_gauge_function,
                         pi_two_normals, pi, gamma0, normal_spacetime_one_form,
                         gauge_constraint, spacetime_metric,
                         normal_dot_gauge_constraint, christoffel_second_kind,
                         gauge_function, pi_2_up, phi_1_up, phi_3_up,
                         christoffel_first_kind_3_up, pi_one_normal,
                         inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
                         shift_dot_three_index_constraint, shift, d_pi);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::ClobberMemory();
  }
}

BENCHMARK(bench);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
