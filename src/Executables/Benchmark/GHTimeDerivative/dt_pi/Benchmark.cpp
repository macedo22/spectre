// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Executables/Benchmark/GHTimeDerivative/dt_pi/BenchmarkedImpls.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
// set up shared stuff
constexpr size_t seed = 17;
std::mt19937 generator(seed);

// =============== BENCHMARKS TO INSTANTIATE ==============

// - manual implementation
// - LHS tensor is a function argument to implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_arg_without_buffer(
    benchmark::State& state) {
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
  // type not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
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

  // RHS: pi_one_normal_spatial
  const pi_one_normal_spatial_type pi_one_normal_spatial =
      make_with_random_values<pi_one_normal_spatial_type>(
          make_not_null(&generator), make_not_null(&distribution),
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
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_pi), spacetime_deriv_gauge_function, pi_two_normals,
        pi, gamma0, normal_spacetime_one_form, gauge_constraint,
        spacetime_metric, normal_dot_gauge_constraint, christoffel_second_kind,
        gauge_function, pi_2_up, phi_1_up, phi_3_up,
        christoffel_first_kind_3_up, pi_one_normal_spatial,
        inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
        shift_dot_three_index_constraint, shift, d_pi);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::ClobberMemory();
  }
}

// - manual implementation
// - LHS tensor is a function argument to implementation
// - equation terms are stored in buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
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
  // type not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_pi_type>,
      ::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<2, pi_two_normals_type>,
      ::Tags::TempTensor<3, pi_type>, ::Tags::TempTensor<4, gamma0_type>,
      ::Tags::TempTensor<5, normal_spacetime_one_form_type>,
      ::Tags::TempTensor<6, gauge_constraint_type>,
      ::Tags::TempTensor<7, spacetime_metric_type>,
      ::Tags::TempTensor<8, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<9, christoffel_second_kind_type>,
      ::Tags::TempTensor<10, gauge_function_type>,
      ::Tags::TempTensor<11, pi_2_up_type>,
      ::Tags::TempTensor<12, phi_1_up_type>,
      ::Tags::TempTensor<13, phi_3_up_type>,
      ::Tags::TempTensor<14, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<15, pi_one_normal_spatial_type>,
      ::Tags::TempTensor<16, inverse_spatial_metric_type>,
      ::Tags::TempTensor<17, d_phi_type>, ::Tags::TempTensor<18, lapse_type>,
      ::Tags::TempTensor<19, gamma1gamma2_type>,
      ::Tags::TempTensor<20, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<21, shift_type>, ::Tags::TempTensor<22, d_pi_type>>>
      vars{num_grid_points};

  // RHS: spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function =
      get<::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&spacetime_deriv_gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi_two_normals
  pi_two_normals_type& pi_two_normals =
      get<::Tags::TempTensor<2, pi_two_normals_type>>(vars);
  fill_with_random_values(make_not_null(&pi_two_normals),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<3, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<4, gamma0_type>>(vars);
  fill_with_random_values(make_not_null(&gamma0), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form =
      get<::Tags::TempTensor<5, normal_spacetime_one_form_type>>(vars);
  fill_with_random_values(make_not_null(&normal_spacetime_one_form),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<6, gauge_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<7, spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint =
      get<::Tags::TempTensor<8, normal_dot_gauge_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&normal_dot_gauge_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<9, christoffel_second_kind_type>>(vars);
  fill_with_random_values(make_not_null(&christoffel_second_kind),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gauge_function
  gauge_function_type& gauge_function =
      get<::Tags::TempTensor<10, gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi_2_up
  pi_2_up_type& pi_2_up = get<::Tags::TempTensor<11, pi_2_up_type>>(vars);
  fill_with_random_values(make_not_null(&pi_2_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<12, phi_1_up_type>>(vars);
  fill_with_random_values(make_not_null(&phi_1_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_3_up
  phi_3_up_type& phi_3_up = get<::Tags::TempTensor<13, phi_3_up_type>>(vars);
  fill_with_random_values(make_not_null(&phi_3_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up =
      get<::Tags::TempTensor<14, christoffel_first_kind_3_up_type>>(vars);
  fill_with_random_values(make_not_null(&christoffel_first_kind_3_up),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial =
      get<::Tags::TempTensor<15, pi_one_normal_spatial_type>>(vars);
  fill_with_random_values(make_not_null(&pi_one_normal_spatial),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric =
      get<::Tags::TempTensor<16, inverse_spatial_metric_type>>(vars);
  fill_with_random_values(make_not_null(&inverse_spatial_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<17, d_phi_type>>(vars);
  fill_with_random_values(make_not_null(&d_phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<18, lapse_type>>(vars);
  fill_with_random_values(make_not_null(&lapse), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma1gamma2
  gamma1gamma2_type& gamma1gamma2 =
      get<::Tags::TempTensor<19, gamma1gamma2_type>>(vars);
  fill_with_random_values(make_not_null(&gamma1gamma2),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<20, shift_dot_three_index_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&shift_dot_three_index_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift
  shift_type& shift = get<::Tags::TempTensor<21, shift_type>>(vars);
  fill_with_random_values(make_not_null(&shift), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<22, d_pi_type>>(vars);
  fill_with_random_values(make_not_null(&d_pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_pi
  dt_pi_type& dt_pi = get<::Tags::TempTensor<0, dt_pi_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_pi), spacetime_deriv_gauge_function, pi_two_normals,
        pi, gamma0, normal_spacetime_one_form, gauge_constraint,
        spacetime_metric, normal_dot_gauge_constraint, christoffel_second_kind,
        gauge_function, pi_2_up, phi_1_up, phi_3_up,
        christoffel_first_kind_3_up, pi_one_normal_spatial,
        inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
        shift_dot_three_index_constraint, shift, d_pi);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::ClobberMemory();
  }
}

// - TensorExpression implementation
// - LHS tensor is a function argument to implementation
// - equation terms are not stored in buffer
// - CaseNumber refers to a specific TE implementation variation
template <typename DataType, size_t Dim, size_t CaseNumber>
void bench_tensorexpression_lhs_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
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
  // type not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
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

  // RHS: pi_one_normal_spatial
  const pi_one_normal_spatial_type pi_one_normal_spatial =
      make_with_random_values<pi_one_normal_spatial_type>(
          make_not_null(&generator), make_not_null(&distribution),
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
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_pi), spacetime_deriv_gauge_function, pi_two_normals,
        pi, gamma0, normal_spacetime_one_form, gauge_constraint,
        spacetime_metric, normal_dot_gauge_constraint, christoffel_second_kind,
        gauge_function, pi_2_up, phi_1_up, phi_3_up,
        christoffel_first_kind_3_up, pi_one_normal_spatial,
        inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
        shift_dot_three_index_constraint, shift, d_pi);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::ClobberMemory();
  }
}

// - TensorExpression implementation
// - LHS tensor is a function argument to implementation
// - equation terms are stored in buffer
// - CaseNumber refers to a specific TE implementation variation
template <typename DataType, size_t Dim, size_t CaseNumber>
void bench_tensorexpression_lhs_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
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
  // type not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_pi_type>,
      ::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<2, pi_two_normals_type>,
      ::Tags::TempTensor<3, pi_type>, ::Tags::TempTensor<4, gamma0_type>,
      ::Tags::TempTensor<5, normal_spacetime_one_form_type>,
      ::Tags::TempTensor<6, gauge_constraint_type>,
      ::Tags::TempTensor<7, spacetime_metric_type>,
      ::Tags::TempTensor<8, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<9, christoffel_second_kind_type>,
      ::Tags::TempTensor<10, gauge_function_type>,
      ::Tags::TempTensor<11, pi_2_up_type>,
      ::Tags::TempTensor<12, phi_1_up_type>,
      ::Tags::TempTensor<13, phi_3_up_type>,
      ::Tags::TempTensor<14, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<15, pi_one_normal_spatial_type>,
      ::Tags::TempTensor<16, inverse_spatial_metric_type>,
      ::Tags::TempTensor<17, d_phi_type>, ::Tags::TempTensor<18, lapse_type>,
      ::Tags::TempTensor<19, gamma1gamma2_type>,
      ::Tags::TempTensor<20, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<21, shift_type>, ::Tags::TempTensor<22, d_pi_type>>>
      vars{num_grid_points};

  // RHS: spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function =
      get<::Tags::TempTensor<1, spacetime_deriv_gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&spacetime_deriv_gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi_two_normals
  pi_two_normals_type& pi_two_normals =
      get<::Tags::TempTensor<2, pi_two_normals_type>>(vars);
  fill_with_random_values(make_not_null(&pi_two_normals),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<3, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<4, gamma0_type>>(vars);
  fill_with_random_values(make_not_null(&gamma0), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form =
      get<::Tags::TempTensor<5, normal_spacetime_one_form_type>>(vars);
  fill_with_random_values(make_not_null(&normal_spacetime_one_form),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<6, gauge_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<7, spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint =
      get<::Tags::TempTensor<8, normal_dot_gauge_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&normal_dot_gauge_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<9, christoffel_second_kind_type>>(vars);
  fill_with_random_values(make_not_null(&christoffel_second_kind),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gauge_function
  gauge_function_type& gauge_function =
      get<::Tags::TempTensor<10, gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi_2_up
  pi_2_up_type& pi_2_up = get<::Tags::TempTensor<11, pi_2_up_type>>(vars);
  fill_with_random_values(make_not_null(&pi_2_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<12, phi_1_up_type>>(vars);
  fill_with_random_values(make_not_null(&phi_1_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_3_up
  phi_3_up_type& phi_3_up = get<::Tags::TempTensor<13, phi_3_up_type>>(vars);
  fill_with_random_values(make_not_null(&phi_3_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up =
      get<::Tags::TempTensor<14, christoffel_first_kind_3_up_type>>(vars);
  fill_with_random_values(make_not_null(&christoffel_first_kind_3_up),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial =
      get<::Tags::TempTensor<15, pi_one_normal_spatial_type>>(vars);
  fill_with_random_values(make_not_null(&pi_one_normal_spatial),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric =
      get<::Tags::TempTensor<16, inverse_spatial_metric_type>>(vars);
  fill_with_random_values(make_not_null(&inverse_spatial_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<17, d_phi_type>>(vars);
  fill_with_random_values(make_not_null(&d_phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<18, lapse_type>>(vars);
  fill_with_random_values(make_not_null(&lapse), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma1gamma2
  gamma1gamma2_type& gamma1gamma2 =
      get<::Tags::TempTensor<19, gamma1gamma2_type>>(vars);
  fill_with_random_values(make_not_null(&gamma1gamma2),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<20, shift_dot_three_index_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&shift_dot_three_index_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift
  shift_type& shift = get<::Tags::TempTensor<21, shift_type>>(vars);
  fill_with_random_values(make_not_null(&shift), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<22, d_pi_type>>(vars);
  fill_with_random_values(make_not_null(&d_pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_pi
  dt_pi_type& dt_pi = get<::Tags::TempTensor<0, dt_pi_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_pi), spacetime_deriv_gauge_function, pi_two_normals,
        pi, gamma0, normal_spacetime_one_form, gauge_constraint,
        spacetime_metric, normal_dot_gauge_constraint, christoffel_second_kind,
        gauge_function, pi_2_up, phi_1_up, phi_3_up,
        christoffel_first_kind_3_up, pi_one_normal_spatial,
        inverse_spatial_metric, d_phi, lapse, gamma1gamma2,
        shift_dot_three_index_constraint, shift, d_pi);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::ClobberMemory();
  }
}

// ========================================================

// Each DataVector case is run with each number of grid points
constexpr std::array<long int, 4> num_grid_point_values = {8, 125, 512, 1000};

// ======= BENCHMARK_TEMPLATE INSTANTIATION HELPERS =======

template <typename DataType, size_t Dim>
void setup_manual_lhs_arg_without_buffer() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "manual/lhs_arg/without_buffer/", Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
                       DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
                       DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(num_grid_point_values[0])
        ->Arg(num_grid_point_values[1])
        ->Arg(num_grid_point_values[2])
        ->Arg(num_grid_point_values[3]);
  }
}

template <typename DataType, size_t Dim>
void setup_manual_lhs_arg_with_buffer() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "manual/lhs_arg/with_buffer/", Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_with_buffer,
                       DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_with_buffer,
                       DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(num_grid_point_values[0])
        ->Arg(num_grid_point_values[1])
        ->Arg(num_grid_point_values[2])
        ->Arg(num_grid_point_values[3]);
  }
}

template <typename DataType, size_t Dim, size_t CaseNumber>
void setup_te_lhs_arg_without_buffer() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "TE : " + std::to_string(CaseNumber) + "/lhs_arg/without_buffer/",
          Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_without_buffer, DataType,
                       Dim, CaseNumber)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_without_buffer, DataType,
                       Dim, CaseNumber)
        ->Name(benchmark_name)
        ->Arg(num_grid_point_values[0])
        ->Arg(num_grid_point_values[1])
        ->Arg(num_grid_point_values[2])
        ->Arg(num_grid_point_values[3]);
  }
}

template <typename DataType, size_t Dim, size_t CaseNumber>
void setup_te_lhs_arg_with_buffer() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "TE : " + std::to_string(CaseNumber) + "/lhs_arg/with_buffer/", Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_with_buffer, DataType,
                       Dim, CaseNumber)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_with_buffer, DataType,
                       Dim, CaseNumber)
        ->Name(benchmark_name)
        ->Arg(num_grid_point_values[0])
        ->Arg(num_grid_point_values[1])
        ->Arg(num_grid_point_values[2])
        ->Arg(num_grid_point_values[3]);
  }
}

// ========================================================

// Instantiate all benchmark cases
void setup_benchmarks() {
  setup_manual_lhs_arg_without_buffer<double, 1>();
  setup_manual_lhs_arg_without_buffer<double, 2>();
  setup_manual_lhs_arg_without_buffer<double, 3>();
  setup_manual_lhs_arg_without_buffer<DataVector, 1>();
  setup_manual_lhs_arg_without_buffer<DataVector, 2>();
  setup_manual_lhs_arg_without_buffer<DataVector, 3>();

  setup_manual_lhs_arg_with_buffer<double, 1>();
  setup_manual_lhs_arg_with_buffer<double, 2>();
  setup_manual_lhs_arg_with_buffer<double, 3>();
  setup_manual_lhs_arg_with_buffer<DataVector, 1>();
  setup_manual_lhs_arg_with_buffer<DataVector, 2>();
  setup_manual_lhs_arg_with_buffer<DataVector, 3>();

  setup_te_lhs_arg_without_buffer<double, 1, 1>();
  setup_te_lhs_arg_without_buffer<double, 2, 1>();
  setup_te_lhs_arg_without_buffer<double, 3, 1>();
  setup_te_lhs_arg_without_buffer<DataVector, 1, 1>();
  setup_te_lhs_arg_without_buffer<DataVector, 2, 1>();
  setup_te_lhs_arg_without_buffer<DataVector, 3, 1>();

  setup_te_lhs_arg_with_buffer<double, 1, 1>();
  setup_te_lhs_arg_with_buffer<double, 2, 1>();
  setup_te_lhs_arg_with_buffer<double, 3, 1>();
  setup_te_lhs_arg_with_buffer<DataVector, 1, 1>();
  setup_te_lhs_arg_with_buffer<DataVector, 2, 1>();
  setup_te_lhs_arg_with_buffer<DataVector, 3, 1>();
}
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
int main(int argc, char** argv) {
  setup_benchmarks();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
#pragma GCC diagnostic pop
