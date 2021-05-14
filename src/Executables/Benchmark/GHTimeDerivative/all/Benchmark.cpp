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
#include "Executables/Benchmark/GHTimeDerivative/all/BenchmarkedImpls.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
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
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using spatial_metric_type = typename BenchmarkImpl::spatial_metric_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using phi_type = typename BenchmarkImpl::phi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using gamma1_type = typename BenchmarkImpl::gamma1_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  // types not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using phi_one_normal_spatial_type =
      typename BenchmarkImpl::phi_one_normal_spatial_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: d_spacetime_metric
  const d_spacetime_metric_type d_spacetime_metric =
      make_with_random_values<d_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: spacetime_metric
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  spacetime_metric_type spacetime_metric(used_for_size);
  gr::spacetime_metric(
      make_not_null(&spacetime_metric),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma0
  const gamma0_type gamma0 = make_with_random_values<gamma0_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1
  const gamma1_type gamma1 = make_with_random_values<gamma1_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma2
  const gamma2_type gamma2 = make_with_random_values<gamma2_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: spacetime_deriv_gauge_function
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<spacetime_deriv_gauge_function_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type dt_spacetime_metric(used_for_size);

  // LHS: dt_pi
  dt_pi_type dt_pi(used_for_size);

  // LHS: dt_phi
  dt_phi_type dt_phi(used_for_size);

  // LHS: temp_gamma1
  temp_gamma1_type temp_gamma1(used_for_size);

  // LHS: temp_gamma2
  temp_gamma2_type temp_gamma2(used_for_size);

  // LHS: gamma1gamma2
  gamma1gamma2_type gamma1gamma2(used_for_size);

  // LHS: pi_two_normals
  pi_two_normals_type pi_two_normals(used_for_size);

  // LHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint(used_for_size);

  // LHS: gamma1_plus_1
  gamma1_plus_1_type gamma1_plus_1(used_for_size);

  // LHS: pi_one_normal
  pi_one_normal_type pi_one_normal(used_for_size);

  // LHS: gauge_constraint
  gauge_constraint_type gauge_constraint(used_for_size);

  // LHS: phi_two_normals
  phi_two_normals_type phi_two_normals(used_for_size);

  // LHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint(
      used_for_size);

  // LHS: phi_one_normal
  phi_one_normal_type phi_one_normal(used_for_size);

  // LHS: pi_2_up
  pi_2_up_type pi_2_up(used_for_size);

  // LHS: three_index_constraint
  three_index_constraint_type three_index_constraint(used_for_size);

  // LHS: phi_1_up
  phi_1_up_type phi_1_up(used_for_size);

  // LHS: phi_3_up
  phi_3_up_type phi_3_up(used_for_size);

  // LHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up(used_for_size);

  // LHS: lapse
  lapse_type lapse(used_for_size);

  // LHS: shift
  shift_type shift(used_for_size);

  // LHS: spatial_metric
  spatial_metric_type spatial_metric(used_for_size);

  // LHS: inverse_spatial_metric
  inverse_spatial_metric_type inverse_spatial_metric(used_for_size);

  // LHS: det_spatial_metric
  det_spatial_metric_type det_spatial_metric(used_for_size);

  // LHS: inverse_spacetime_metric
  inverse_spacetime_metric_type inverse_spacetime_metric(used_for_size);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);

  // LHS: christoffel_second_kind
  christoffel_second_kind_type christoffel_second_kind(used_for_size);

  // LHS: trace_christoffel
  trace_christoffel_type trace_christoffel(used_for_size);

  // LHS: normal_spacetime_vector
  normal_spacetime_vector_type normal_spacetime_vector(used_for_size);

  // LHS: normal_spacetime_one_form
  normal_spacetime_one_form_type normal_spacetime_one_form(used_for_size);

  // LHS: da_spacetime_metric
  da_spacetime_metric_type da_spacetime_metric(used_for_size);

  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops

  // LHS: pi_one_normal_spatial
  pi_one_normal_spatial_type pi_one_normal_spatial(used_for_size);

  // LHS: phi_one_normal_spatial
  phi_one_normal_spatial_type phi_one_normal_spatial(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_spacetime_metric), make_not_null(&dt_pi),
        make_not_null(&dt_phi), make_not_null(&temp_gamma1),
        make_not_null(&temp_gamma2), make_not_null(&gamma1gamma2),
        make_not_null(&pi_two_normals),
        make_not_null(&normal_dot_gauge_constraint),
        make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
        make_not_null(&gauge_constraint), make_not_null(&phi_two_normals),
        make_not_null(&shift_dot_three_index_constraint),
        make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
        make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
        make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
        make_not_null(&lapse), make_not_null(&shift),
        make_not_null(&spatial_metric), make_not_null(&inverse_spatial_metric),
        make_not_null(&det_spatial_metric),
        make_not_null(&inverse_spacetime_metric),
        make_not_null(&christoffel_first_kind),
        make_not_null(&christoffel_second_kind),
        make_not_null(&trace_christoffel),
        make_not_null(&normal_spacetime_vector),
        make_not_null(&normal_spacetime_one_form),
        make_not_null(&da_spacetime_metric), d_spacetime_metric, d_pi, d_phi,
        spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
        spacetime_deriv_gauge_function, make_not_null(&pi_one_normal_spatial),
        make_not_null(&phi_one_normal_spatial));
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::DoNotOptimize(dt_phi);
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
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using spatial_metric_type = typename BenchmarkImpl::spatial_metric_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using phi_type = typename BenchmarkImpl::phi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using gamma1_type = typename BenchmarkImpl::gamma1_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  // types not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using phi_one_normal_spatial_type =
      typename BenchmarkImpl::phi_one_normal_spatial_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_spacetime_metric_type>,
      ::Tags::TempTensor<1, dt_pi_type>, ::Tags::TempTensor<2, dt_phi_type>,
      ::Tags::TempTensor<3, temp_gamma1_type>,
      ::Tags::TempTensor<4, temp_gamma2_type>,
      ::Tags::TempTensor<5, gamma1gamma2_type>,
      ::Tags::TempTensor<6, pi_two_normals_type>,
      ::Tags::TempTensor<7, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<8, gamma1_plus_1_type>,
      ::Tags::TempTensor<9, pi_one_normal_type>,
      ::Tags::TempTensor<10, gauge_constraint_type>,
      ::Tags::TempTensor<11, phi_two_normals_type>,
      ::Tags::TempTensor<12, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<13, phi_one_normal_type>,
      ::Tags::TempTensor<14, pi_2_up_type>,
      ::Tags::TempTensor<15, three_index_constraint_type>,
      ::Tags::TempTensor<16, phi_1_up_type>,
      ::Tags::TempTensor<17, phi_3_up_type>,
      ::Tags::TempTensor<18, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<19, lapse_type>, ::Tags::TempTensor<20, shift_type>,
      ::Tags::TempTensor<21, spatial_metric_type>,
      ::Tags::TempTensor<22, inverse_spatial_metric_type>,
      ::Tags::TempTensor<23, det_spatial_metric_type>,
      ::Tags::TempTensor<24, inverse_spacetime_metric_type>,
      ::Tags::TempTensor<25, christoffel_first_kind_type>,
      ::Tags::TempTensor<26, christoffel_second_kind_type>,
      ::Tags::TempTensor<27, trace_christoffel_type>,
      ::Tags::TempTensor<28, normal_spacetime_vector_type>,
      ::Tags::TempTensor<29, normal_spacetime_one_form_type>,
      ::Tags::TempTensor<30, da_spacetime_metric_type>,
      ::Tags::TempTensor<31, d_spacetime_metric_type>,
      ::Tags::TempTensor<32, d_pi_type>, ::Tags::TempTensor<33, d_phi_type>,
      ::Tags::TempTensor<34, spacetime_metric_type>,
      ::Tags::TempTensor<35, pi_type>, ::Tags::TempTensor<36, phi_type>,
      ::Tags::TempTensor<37, gamma0_type>, ::Tags::TempTensor<38, gamma1_type>,
      ::Tags::TempTensor<39, gamma2_type>,
      ::Tags::TempTensor<40, gauge_function_type>,
      ::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<42, pi_one_normal_spatial_type>,
      ::Tags::TempTensor<43, phi_one_normal_spatial_type>>>
      vars{num_grid_points};

  // RHS: d_spacetime_metric
  d_spacetime_metric_type& d_spacetime_metric =
      get<::Tags::TempTensor<31, d_spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&d_spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<32, d_pi_type>>(vars);
  fill_with_random_values(make_not_null(&d_pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<33, d_phi_type>>(vars);
  fill_with_random_values(make_not_null(&d_phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<34, spacetime_metric_type>>(vars);
  gr::spacetime_metric(
      make_not_null(&spacetime_metric),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<35, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));
  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<36, phi_type>>(vars);
  fill_with_random_values(make_not_null(&phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<37, gamma0_type>>(vars);
  fill_with_random_values(make_not_null(&gamma0), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma1
  gamma1_type& gamma1 = get<::Tags::TempTensor<38, gamma1_type>>(vars);
  fill_with_random_values(make_not_null(&gamma1), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<39, gamma2_type>>(vars);
  fill_with_random_values(make_not_null(&gamma2), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gauge_function
  gauge_function_type& gauge_function =
      get<::Tags::TempTensor<40, gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function =
      get<::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&spacetime_deriv_gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type& dt_spacetime_metric =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  // LHS: dt_pi
  dt_pi_type& dt_pi = get<::Tags::TempTensor<1, dt_pi_type>>(vars);

  // LHS: dt_phi
  dt_phi_type& dt_phi = get<::Tags::TempTensor<2, dt_phi_type>>(vars);

  // LHS: temp_gamma1
  temp_gamma1_type& temp_gamma1 =
      get<::Tags::TempTensor<3, temp_gamma1_type>>(vars);

  // LHS: temp_gamma2
  temp_gamma2_type& temp_gamma2 =
      get<::Tags::TempTensor<4, temp_gamma2_type>>(vars);

  // LHS: gamma1gamma2
  gamma1gamma2_type& gamma1gamma2 =
      get<::Tags::TempTensor<5, gamma1gamma2_type>>(vars);

  // LHS: pi_two_normals
  pi_two_normals_type& pi_two_normals =
      get<::Tags::TempTensor<6, pi_two_normals_type>>(vars);

  // LHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint =
      get<::Tags::TempTensor<7, normal_dot_gauge_constraint_type>>(vars);

  // LHS: gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1 =
      get<::Tags::TempTensor<8, gamma1_plus_1_type>>(vars);

  // LHS: pi_one_normal
  pi_one_normal_type& pi_one_normal =
      get<::Tags::TempTensor<9, pi_one_normal_type>>(vars);

  // LHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<10, gauge_constraint_type>>(vars);

  // LHS: phi_two_normals
  phi_two_normals_type& phi_two_normals =
      get<::Tags::TempTensor<11, phi_two_normals_type>>(vars);

  // LHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<12, shift_dot_three_index_constraint_type>>(vars);

  // LHS: phi_one_normal
  phi_one_normal_type& phi_one_normal =
      get<::Tags::TempTensor<13, phi_one_normal_type>>(vars);

  // LHS: pi_2_up
  pi_2_up_type& pi_2_up = get<::Tags::TempTensor<14, pi_2_up_type>>(vars);

  // LHS: three_index_constraint
  three_index_constraint_type& three_index_constraint =
      get<::Tags::TempTensor<15, three_index_constraint_type>>(vars);

  // LHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<16, phi_1_up_type>>(vars);

  // LHS: phi_3_up
  phi_3_up_type& phi_3_up = get<::Tags::TempTensor<17, phi_3_up_type>>(vars);

  // LHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up =
      get<::Tags::TempTensor<18, christoffel_first_kind_3_up_type>>(vars);

  // LHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<19, lapse_type>>(vars);

  // LHS: shift
  shift_type& shift = get<::Tags::TempTensor<20, shift_type>>(vars);

  // LHS: spatial_metric
  spatial_metric_type& spatial_metric =
      get<::Tags::TempTensor<21, spatial_metric_type>>(vars);

  // LHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric =
      get<::Tags::TempTensor<22, inverse_spatial_metric_type>>(vars);

  // LHS: det_spatial_metric
  det_spatial_metric_type& det_spatial_metric =
      get<::Tags::TempTensor<23, det_spatial_metric_type>>(vars);

  // LHS: inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<24, inverse_spacetime_metric_type>>(vars);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<25, christoffel_first_kind_type>>(vars);

  // LHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<26, christoffel_second_kind_type>>(vars);

  // LHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<27, trace_christoffel_type>>(vars);

  // LHS: normal_spacetime_vector
  normal_spacetime_vector_type& normal_spacetime_vector =
      get<::Tags::TempTensor<28, normal_spacetime_vector_type>>(vars);

  // LHS: normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form =
      get<::Tags::TempTensor<29, normal_spacetime_one_form_type>>(vars);

  // LHS: da_spacetime_metric
  da_spacetime_metric_type& da_spacetime_metric =
      get<::Tags::TempTensor<30, da_spacetime_metric_type>>(vars);

  // LHS: pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial =
      get<::Tags::TempTensor<42, pi_one_normal_spatial_type>>(vars);

  // LHS: phi_one_normal_spatial
  phi_one_normal_spatial_type& phi_one_normal_spatial =
      get<::Tags::TempTensor<43, phi_one_normal_spatial_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_spacetime_metric), make_not_null(&dt_pi),
        make_not_null(&dt_phi), make_not_null(&temp_gamma1),
        make_not_null(&temp_gamma2), make_not_null(&gamma1gamma2),
        make_not_null(&pi_two_normals),
        make_not_null(&normal_dot_gauge_constraint),
        make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
        make_not_null(&gauge_constraint), make_not_null(&phi_two_normals),
        make_not_null(&shift_dot_three_index_constraint),
        make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
        make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
        make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
        make_not_null(&lapse), make_not_null(&shift),
        make_not_null(&spatial_metric), make_not_null(&inverse_spatial_metric),
        make_not_null(&det_spatial_metric),
        make_not_null(&inverse_spacetime_metric),
        make_not_null(&christoffel_first_kind),
        make_not_null(&christoffel_second_kind),
        make_not_null(&trace_christoffel),
        make_not_null(&normal_spacetime_vector),
        make_not_null(&normal_spacetime_one_form),
        make_not_null(&da_spacetime_metric), d_spacetime_metric, d_pi, d_phi,
        spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
        spacetime_deriv_gauge_function, make_not_null(&pi_one_normal_spatial),
        make_not_null(&phi_one_normal_spatial));
    benchmark::DoNotOptimize(dt_spacetime_metric);
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
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using spatial_metric_type = typename BenchmarkImpl::spatial_metric_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using phi_type = typename BenchmarkImpl::phi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using gamma1_type = typename BenchmarkImpl::gamma1_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  // types not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using phi_one_normal_spatial_type =
      typename BenchmarkImpl::phi_one_normal_spatial_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: d_spacetime_metric
  const d_spacetime_metric_type d_spacetime_metric =
      make_with_random_values<d_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: spacetime_metric
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  spacetime_metric_type spacetime_metric(used_for_size);
  gr::spacetime_metric(
      make_not_null(&spacetime_metric),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma0
  const gamma0_type gamma0 = make_with_random_values<gamma0_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1
  const gamma1_type gamma1 = make_with_random_values<gamma1_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma2
  const gamma2_type gamma2 = make_with_random_values<gamma2_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: spacetime_deriv_gauge_function
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<spacetime_deriv_gauge_function_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type dt_spacetime_metric(used_for_size);

  // LHS: dt_pi
  dt_pi_type dt_pi(used_for_size);

  // LHS: dt_phi
  dt_phi_type dt_phi(used_for_size);

  // LHS: temp_gamma1
  temp_gamma1_type temp_gamma1(used_for_size);

  // LHS: temp_gamma2
  temp_gamma2_type temp_gamma2(used_for_size);

  // LHS: gamma1gamma2
  gamma1gamma2_type gamma1gamma2(used_for_size);

  // LHS: pi_two_normals
  pi_two_normals_type pi_two_normals(used_for_size);

  // LHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint(used_for_size);

  // LHS: gamma1_plus_1
  gamma1_plus_1_type gamma1_plus_1(used_for_size);

  // LHS: pi_one_normal
  pi_one_normal_type pi_one_normal(used_for_size);

  // LHS: gauge_constraint
  gauge_constraint_type gauge_constraint(used_for_size);

  // LHS: phi_two_normals
  phi_two_normals_type phi_two_normals(used_for_size);

  // LHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint(
      used_for_size);

  // LHS: phi_one_normal
  phi_one_normal_type phi_one_normal(used_for_size);

  // LHS: pi_2_up
  pi_2_up_type pi_2_up(used_for_size);

  // LHS: three_index_constraint
  three_index_constraint_type three_index_constraint(used_for_size);

  // LHS: phi_1_up
  phi_1_up_type phi_1_up(used_for_size);

  // LHS: phi_3_up
  phi_3_up_type phi_3_up(used_for_size);

  // LHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up(used_for_size);

  // LHS: lapse
  lapse_type lapse(used_for_size);

  // LHS: shift
  shift_type shift(used_for_size);

  // LHS: spatial_metric
  spatial_metric_type spatial_metric(used_for_size);

  // LHS: inverse_spatial_metric
  inverse_spatial_metric_type inverse_spatial_metric(used_for_size);

  // LHS: det_spatial_metric
  det_spatial_metric_type det_spatial_metric(used_for_size);

  // LHS: inverse_spacetime_metric
  inverse_spacetime_metric_type inverse_spacetime_metric(used_for_size);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);

  // LHS: christoffel_second_kind
  christoffel_second_kind_type christoffel_second_kind(used_for_size);

  // LHS: trace_christoffel
  trace_christoffel_type trace_christoffel(used_for_size);

  // LHS: normal_spacetime_vector
  normal_spacetime_vector_type normal_spacetime_vector(used_for_size);

  // LHS: normal_spacetime_one_form
  normal_spacetime_one_form_type normal_spacetime_one_form(used_for_size);

  // LHS: da_spacetime_metric
  da_spacetime_metric_type da_spacetime_metric(used_for_size);

  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops

  // LHS: pi_one_normal_spatial
  pi_one_normal_spatial_type pi_one_normal_spatial(used_for_size);

  // LHS: phi_one_normal_spatial
  phi_one_normal_spatial_type phi_one_normal_spatial(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_spacetime_metric), make_not_null(&dt_pi),
        make_not_null(&dt_phi), make_not_null(&temp_gamma1),
        make_not_null(&temp_gamma2), make_not_null(&gamma1gamma2),
        make_not_null(&pi_two_normals),
        make_not_null(&normal_dot_gauge_constraint),
        make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
        make_not_null(&gauge_constraint), make_not_null(&phi_two_normals),
        make_not_null(&shift_dot_three_index_constraint),
        make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
        make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
        make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
        make_not_null(&lapse), make_not_null(&shift),
        make_not_null(&spatial_metric), make_not_null(&inverse_spatial_metric),
        make_not_null(&det_spatial_metric),
        make_not_null(&inverse_spacetime_metric),
        make_not_null(&christoffel_first_kind),
        make_not_null(&christoffel_second_kind),
        make_not_null(&trace_christoffel),
        make_not_null(&normal_spacetime_vector),
        make_not_null(&normal_spacetime_one_form),
        make_not_null(&da_spacetime_metric), d_spacetime_metric, d_pi, d_phi,
        spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
        spacetime_deriv_gauge_function, make_not_null(&pi_one_normal_spatial),
        make_not_null(&phi_one_normal_spatial));
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::DoNotOptimize(dt_phi);
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
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using pi_two_normals_type = typename BenchmarkImpl::pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using pi_2_up_type = typename BenchmarkImpl::pi_2_up_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using phi_3_up_type = typename BenchmarkImpl::phi_3_up_type;
  using christoffel_first_kind_3_up_type =
      typename BenchmarkImpl::christoffel_first_kind_3_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using spatial_metric_type = typename BenchmarkImpl::spatial_metric_type;
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
  using normal_spacetime_one_form_type =
      typename BenchmarkImpl::normal_spacetime_one_form_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;
  using d_spacetime_metric_type =
      typename BenchmarkImpl::d_spacetime_metric_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;
  using spacetime_metric_type = typename BenchmarkImpl::spacetime_metric_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using phi_type = typename BenchmarkImpl::phi_type;
  using gamma0_type = typename BenchmarkImpl::gamma0_type;
  using gamma1_type = typename BenchmarkImpl::gamma1_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::spacetime_deriv_gauge_function_type;
  // types not in SpECTRE implementation, but needed by TE implementation since
  // TEs can't yet iterate over the spatial components of a spacetime index
  using pi_one_normal_spatial_type =
      typename BenchmarkImpl::pi_one_normal_spatial_type;
  using phi_one_normal_spatial_type =
      typename BenchmarkImpl::phi_one_normal_spatial_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_spacetime_metric_type>,
      ::Tags::TempTensor<1, dt_pi_type>, ::Tags::TempTensor<2, dt_phi_type>,
      ::Tags::TempTensor<3, temp_gamma1_type>,
      ::Tags::TempTensor<4, temp_gamma2_type>,
      ::Tags::TempTensor<5, gamma1gamma2_type>,
      ::Tags::TempTensor<6, pi_two_normals_type>,
      ::Tags::TempTensor<7, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<8, gamma1_plus_1_type>,
      ::Tags::TempTensor<9, pi_one_normal_type>,
      ::Tags::TempTensor<10, gauge_constraint_type>,
      ::Tags::TempTensor<11, phi_two_normals_type>,
      ::Tags::TempTensor<12, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<13, phi_one_normal_type>,
      ::Tags::TempTensor<14, pi_2_up_type>,
      ::Tags::TempTensor<15, three_index_constraint_type>,
      ::Tags::TempTensor<16, phi_1_up_type>,
      ::Tags::TempTensor<17, phi_3_up_type>,
      ::Tags::TempTensor<18, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<19, lapse_type>, ::Tags::TempTensor<20, shift_type>,
      ::Tags::TempTensor<21, spatial_metric_type>,
      ::Tags::TempTensor<22, inverse_spatial_metric_type>,
      ::Tags::TempTensor<23, det_spatial_metric_type>,
      ::Tags::TempTensor<24, inverse_spacetime_metric_type>,
      ::Tags::TempTensor<25, christoffel_first_kind_type>,
      ::Tags::TempTensor<26, christoffel_second_kind_type>,
      ::Tags::TempTensor<27, trace_christoffel_type>,
      ::Tags::TempTensor<28, normal_spacetime_vector_type>,
      ::Tags::TempTensor<29, normal_spacetime_one_form_type>,
      ::Tags::TempTensor<30, da_spacetime_metric_type>,
      ::Tags::TempTensor<31, d_spacetime_metric_type>,
      ::Tags::TempTensor<32, d_pi_type>, ::Tags::TempTensor<33, d_phi_type>,
      ::Tags::TempTensor<34, spacetime_metric_type>,
      ::Tags::TempTensor<35, pi_type>, ::Tags::TempTensor<36, phi_type>,
      ::Tags::TempTensor<37, gamma0_type>, ::Tags::TempTensor<38, gamma1_type>,
      ::Tags::TempTensor<39, gamma2_type>,
      ::Tags::TempTensor<40, gauge_function_type>,
      ::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<42, pi_one_normal_spatial_type>,
      ::Tags::TempTensor<43, phi_one_normal_spatial_type>>>
      vars{num_grid_points};

  // RHS: d_spacetime_metric
  d_spacetime_metric_type& d_spacetime_metric =
      get<::Tags::TempTensor<31, d_spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&d_spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<32, d_pi_type>>(vars);
  fill_with_random_values(make_not_null(&d_pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<33, d_phi_type>>(vars);
  fill_with_random_values(make_not_null(&d_phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<34, spacetime_metric_type>>(vars);
  gr::spacetime_metric(
      make_not_null(&spacetime_metric),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<35, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));
  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<36, phi_type>>(vars);
  fill_with_random_values(make_not_null(&phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<37, gamma0_type>>(vars);
  fill_with_random_values(make_not_null(&gamma0), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma1
  gamma1_type& gamma1 = get<::Tags::TempTensor<38, gamma1_type>>(vars);
  fill_with_random_values(make_not_null(&gamma1), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<39, gamma2_type>>(vars);
  fill_with_random_values(make_not_null(&gamma2), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gauge_function
  gauge_function_type& gauge_function =
      get<::Tags::TempTensor<40, gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function =
      get<::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&spacetime_deriv_gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type& dt_spacetime_metric =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  // LHS: dt_pi
  dt_pi_type& dt_pi = get<::Tags::TempTensor<1, dt_pi_type>>(vars);

  // LHS: dt_phi
  dt_phi_type& dt_phi = get<::Tags::TempTensor<2, dt_phi_type>>(vars);

  // LHS: temp_gamma1
  temp_gamma1_type& temp_gamma1 =
      get<::Tags::TempTensor<3, temp_gamma1_type>>(vars);

  // LHS: temp_gamma2
  temp_gamma2_type& temp_gamma2 =
      get<::Tags::TempTensor<4, temp_gamma2_type>>(vars);

  // LHS: gamma1gamma2
  gamma1gamma2_type& gamma1gamma2 =
      get<::Tags::TempTensor<5, gamma1gamma2_type>>(vars);

  // LHS: pi_two_normals
  pi_two_normals_type& pi_two_normals =
      get<::Tags::TempTensor<6, pi_two_normals_type>>(vars);

  // LHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint =
      get<::Tags::TempTensor<7, normal_dot_gauge_constraint_type>>(vars);

  // LHS: gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1 =
      get<::Tags::TempTensor<8, gamma1_plus_1_type>>(vars);

  // LHS: pi_one_normal
  pi_one_normal_type& pi_one_normal =
      get<::Tags::TempTensor<9, pi_one_normal_type>>(vars);

  // LHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<10, gauge_constraint_type>>(vars);

  // LHS: phi_two_normals
  phi_two_normals_type& phi_two_normals =
      get<::Tags::TempTensor<11, phi_two_normals_type>>(vars);

  // LHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<12, shift_dot_three_index_constraint_type>>(vars);

  // LHS: phi_one_normal
  phi_one_normal_type& phi_one_normal =
      get<::Tags::TempTensor<13, phi_one_normal_type>>(vars);

  // LHS: pi_2_up
  pi_2_up_type& pi_2_up = get<::Tags::TempTensor<14, pi_2_up_type>>(vars);

  // LHS: three_index_constraint
  three_index_constraint_type& three_index_constraint =
      get<::Tags::TempTensor<15, three_index_constraint_type>>(vars);

  // LHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<16, phi_1_up_type>>(vars);

  // LHS: phi_3_up
  phi_3_up_type& phi_3_up = get<::Tags::TempTensor<17, phi_3_up_type>>(vars);

  // LHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up =
      get<::Tags::TempTensor<18, christoffel_first_kind_3_up_type>>(vars);

  // LHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<19, lapse_type>>(vars);

  // LHS: shift
  shift_type& shift = get<::Tags::TempTensor<20, shift_type>>(vars);

  // LHS: spatial_metric
  spatial_metric_type& spatial_metric =
      get<::Tags::TempTensor<21, spatial_metric_type>>(vars);

  // LHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric =
      get<::Tags::TempTensor<22, inverse_spatial_metric_type>>(vars);

  // LHS: det_spatial_metric
  det_spatial_metric_type& det_spatial_metric =
      get<::Tags::TempTensor<23, det_spatial_metric_type>>(vars);

  // LHS: inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<24, inverse_spacetime_metric_type>>(vars);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<25, christoffel_first_kind_type>>(vars);

  // LHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<26, christoffel_second_kind_type>>(vars);

  // LHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<27, trace_christoffel_type>>(vars);

  // LHS: normal_spacetime_vector
  normal_spacetime_vector_type& normal_spacetime_vector =
      get<::Tags::TempTensor<28, normal_spacetime_vector_type>>(vars);

  // LHS: normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form =
      get<::Tags::TempTensor<29, normal_spacetime_one_form_type>>(vars);

  // LHS: da_spacetime_metric
  da_spacetime_metric_type& da_spacetime_metric =
      get<::Tags::TempTensor<30, da_spacetime_metric_type>>(vars);

  // LHS: pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial =
      get<::Tags::TempTensor<42, pi_one_normal_spatial_type>>(vars);

  // LHS: phi_one_normal_spatial
  phi_one_normal_spatial_type& phi_one_normal_spatial =
      get<::Tags::TempTensor<43, phi_one_normal_spatial_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_spacetime_metric), make_not_null(&dt_pi),
        make_not_null(&dt_phi), make_not_null(&temp_gamma1),
        make_not_null(&temp_gamma2), make_not_null(&gamma1gamma2),
        make_not_null(&pi_two_normals),
        make_not_null(&normal_dot_gauge_constraint),
        make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
        make_not_null(&gauge_constraint), make_not_null(&phi_two_normals),
        make_not_null(&shift_dot_three_index_constraint),
        make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
        make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
        make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
        make_not_null(&lapse), make_not_null(&shift),
        make_not_null(&spatial_metric), make_not_null(&inverse_spatial_metric),
        make_not_null(&det_spatial_metric),
        make_not_null(&inverse_spacetime_metric),
        make_not_null(&christoffel_first_kind),
        make_not_null(&christoffel_second_kind),
        make_not_null(&trace_christoffel),
        make_not_null(&normal_spacetime_vector),
        make_not_null(&normal_spacetime_one_form),
        make_not_null(&da_spacetime_metric), d_spacetime_metric, d_pi, d_phi,
        spacetime_metric, pi, phi, gamma0, gamma1, gamma2, gauge_function,
        spacetime_deriv_gauge_function, make_not_null(&pi_one_normal_spatial),
        make_not_null(&phi_one_normal_spatial));
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::DoNotOptimize(dt_phi);
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
