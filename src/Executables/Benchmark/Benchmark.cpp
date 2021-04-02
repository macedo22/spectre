// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// This file is an example of how to do microbenchmark with Google Benchmark
// https://github.com/google/benchmark
// For two examples in different anonymous namespaces

namespace {
// Benchmark of push_back() in std::vector, following Chandler Carruth's talk
// at CppCon in 2015,
// https://www.youtube.com/watch?v=nXaxk27zwlk

// void bench_create(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    benchmark::DoNotOptimize(&v);
//    static_cast<void>(v);
//  }
// }
// BENCHMARK(bench_create);

// void bench_reserve(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//  }
// }
// BENCHMARK(bench_reserve);

// void bench_push_back(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//    v.push_back(42);
//    benchmark::ClobberMemory();
//  }
// }
// BENCHMARK(bench_push_back);
}  // namespace

namespace {
// In this anonymous namespace is an example of microbenchmarking the
// all_gradient routine for the GH system

// template <size_t Dim>
// struct Kappa : db::SimpleTag {
//   using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
// };
// template <size_t Dim>
// struct Psi : db::SimpleTag {
//   using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
// };

// clang-tidy: don't pass be non-const reference
// void bench_all_gradient(benchmark::State& state) {  // NOLINT
//   constexpr const size_t pts_1d = 4;
//   constexpr const size_t Dim = 3;
//   const Mesh<Dim> mesh{pts_1d, Spectral::Basis::Legendre,
//                        Spectral::Quadrature::GaussLobatto};
//   domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
//   using Map3d =
//       domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
//                                              domain::CoordinateMaps::Affine,
//                                              domain::CoordinateMaps::Affine>;
//   domain::CoordinateMap<Frame::Logical, Frame::Grid, Map3d> map(
//       Map3d{map1d, map1d, map1d});

//   using VarTags = tmpl::list<Kappa<Dim>, Psi<Dim>>;
//   const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Grid> inv_jac
//   =
//       map.inv_jacobian(logical_coordinates(mesh));
//   const auto grid_coords = map(logical_coordinates(mesh));
//   Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

//   while (state.KeepRunning()) {
//     benchmark::DoNotOptimize(partial_derivatives<VarTags>(vars, mesh,
//     inv_jac));
//   }
// }
// BENCHMARK(bench_all_gradient);  // NOLINT
}  // namespace

namespace {
// set up shared stuff
constexpr size_t Dim = 3;
using BenchmarkImpl = BenchmarkImpl<Dim>;
// tensor types in tensor equation being benchmarked
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
using det_spatial_metric_type = typename BenchmarkImpl::det_spatial_metric_type;
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
using d_spacetime_metric_type = typename BenchmarkImpl::d_spacetime_metric_type;
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

constexpr size_t seed = 17;
std::mt19937 generator(seed);

// benchmark manual implementation, takes LHS as arg, equation terms not in
// buffer
void bench_manual_tensor_equation_lhs_tensor_as_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
  // Set up input constant tensors in equations
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = static_cast<size_t>(state.range(0));
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars),
                          make_not_null(&generator),
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const spacetime_metric_type& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const phi_type& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const pi_type& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const d_spacetime_metric_type& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_phi_type& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_pi_type& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const size_t num_grid_points = mesh.number_of_grid_points();

  // Initialize non-const tensors to be filled by manual implementation
  dt_spacetime_metric_type dt_spacetime_metric(num_grid_points);
  dt_pi_type dt_pi(num_grid_points);
  dt_phi_type dt_phi(num_grid_points);
  temp_gamma1_type temp_gamma1(num_grid_points);
  temp_gamma2_type temp_gamma2(num_grid_points);
  gamma1gamma2_type gamma1gamma2(num_grid_points);
  pi_two_normals_type pi_two_normals(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint(num_grid_points);
  gamma1_plus_1_type gamma1_plus_1(num_grid_points);
  pi_one_normal_type pi_one_normal(num_grid_points);
  gauge_constraint_type gauge_constraint(num_grid_points);
  phi_two_normals_type phi_two_normals(num_grid_points);
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint(
      num_grid_points);
  phi_one_normal_type phi_one_normal(num_grid_points);
  pi_2_up_type pi_2_up(num_grid_points);
  three_index_constraint_type three_index_constraint(num_grid_points);
  phi_1_up_type phi_1_up(num_grid_points);
  phi_3_up_type phi_3_up(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up(num_grid_points);
  lapse_type lapse(num_grid_points);
  shift_type shift(num_grid_points);
  spatial_metric_type spatial_metric(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric(num_grid_points);
  det_spatial_metric_type det_spatial_metric(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric(num_grid_points);
  christoffel_first_kind_type christoffel_first_kind(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind(num_grid_points);
  trace_christoffel_type trace_christoffel(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form(num_grid_points);
  da_spacetime_metric_type da_spacetime_metric(num_grid_points);
  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops
  pi_one_normal_spatial_type pi_one_normal_spatial(num_grid_points);
  phi_one_normal_spatial_type phi_one_normal_spatial(num_grid_points);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_as_arg(
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

// benchmark manual implementation, takes LHS as arg, equation terms in buffer
void bench_manual_tensor_equation_lhs_tensor_as_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
  // Set up input constant tensors in equations
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = static_cast<size_t>(state.range(0));
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars),
                          make_not_null(&generator),
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const spacetime_metric_type& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const phi_type& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const pi_type& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const d_spacetime_metric_type& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_phi_type& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_pi_type& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const size_t num_grid_points = mesh.number_of_grid_points();

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

  // dt_spacetime_metric
  dt_spacetime_metric_type& dt_spacetime_metric_temp =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  // dt_pi
  dt_pi_type& dt_pi_temp = get<::Tags::TempTensor<1, dt_pi_type>>(vars);

  // dt_phi
  dt_phi_type& dt_phi_temp = get<::Tags::TempTensor<2, dt_phi_type>>(vars);

  // temp_gamma1
  temp_gamma1_type& temp_gamma1_temp =
      get<::Tags::TempTensor<3, temp_gamma1_type>>(vars);

  // temp_gamma2
  temp_gamma2_type& temp_gamma2_temp =
      get<::Tags::TempTensor<4, temp_gamma2_type>>(vars);

  // gamma1gamma2
  gamma1gamma2_type& gamma1gamma2_temp =
      get<::Tags::TempTensor<5, gamma1gamma2_type>>(vars);

  // pi_two_normals
  pi_two_normals_type& pi_two_normals_temp =
      get<::Tags::TempTensor<6, pi_two_normals_type>>(vars);

  // normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_temp =
      get<::Tags::TempTensor<7, normal_dot_gauge_constraint_type>>(vars);

  // gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1_temp =
      get<::Tags::TempTensor<8, gamma1_plus_1_type>>(vars);

  // pi_one_normal
  pi_one_normal_type& pi_one_normal_temp =
      get<::Tags::TempTensor<9, pi_one_normal_type>>(vars);

  // gauge_constraint
  gauge_constraint_type& gauge_constraint_temp =
      get<::Tags::TempTensor<10, gauge_constraint_type>>(vars);

  // phi_two_normals
  phi_two_normals_type& phi_two_normals_temp =
      get<::Tags::TempTensor<11, phi_two_normals_type>>(vars);

  // shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint_temp =
      get<::Tags::TempTensor<12, shift_dot_three_index_constraint_type>>(vars);

  // phi_one_normal
  phi_one_normal_type& phi_one_normal_temp =
      get<::Tags::TempTensor<13, phi_one_normal_type>>(vars);

  // pi_2_up
  pi_2_up_type& pi_2_up_temp = get<::Tags::TempTensor<14, pi_2_up_type>>(vars);

  // three_index_constraint
  three_index_constraint_type& three_index_constraint_temp =
      get<::Tags::TempTensor<15, three_index_constraint_type>>(vars);

  // phi_1_up
  phi_1_up_type& phi_1_up_temp =
      get<::Tags::TempTensor<16, phi_1_up_type>>(vars);

  // phi_3_up
  phi_3_up_type& phi_3_up_temp =
      get<::Tags::TempTensor<17, phi_3_up_type>>(vars);

  // christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up_temp =
      get<::Tags::TempTensor<18, christoffel_first_kind_3_up_type>>(vars);

  // lapse
  lapse_type& lapse_temp = get<::Tags::TempTensor<19, lapse_type>>(vars);

  // shift
  shift_type& shift_temp = get<::Tags::TempTensor<20, shift_type>>(vars);

  // spatial_metric
  spatial_metric_type& spatial_metric_temp =
      get<::Tags::TempTensor<21, spatial_metric_type>>(vars);

  // inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric_temp =
      get<::Tags::TempTensor<22, inverse_spatial_metric_type>>(vars);

  // det_spatial_metric
  det_spatial_metric_type& det_spatial_metric_temp =
      get<::Tags::TempTensor<23, det_spatial_metric_type>>(vars);

  // inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric_temp =
      get<::Tags::TempTensor<24, inverse_spacetime_metric_type>>(vars);

  // christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind_temp =
      get<::Tags::TempTensor<25, christoffel_first_kind_type>>(vars);

  // christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind_temp =
      get<::Tags::TempTensor<26, christoffel_second_kind_type>>(vars);

  // trace_christoffel
  trace_christoffel_type& trace_christoffel_temp =
      get<::Tags::TempTensor<27, trace_christoffel_type>>(vars);

  // normal_spacetime_vector
  normal_spacetime_vector_type& normal_spacetime_vector_temp =
      get<::Tags::TempTensor<28, normal_spacetime_vector_type>>(vars);

  // normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form_temp =
      get<::Tags::TempTensor<29, normal_spacetime_one_form_type>>(vars);

  // da_spacetime_metric
  da_spacetime_metric_type& da_spacetime_metric_temp =
      get<::Tags::TempTensor<30, da_spacetime_metric_type>>(vars);

  // d_spacetime_metric
  d_spacetime_metric_type& d_spacetime_metric_temp =
      get<::Tags::TempTensor<31, d_spacetime_metric_type>>(vars);
  BenchmarkHelpers::copy_tensor(d_spacetime_metric,
                                make_not_null(&d_spacetime_metric_temp));

  // d_pi
  d_pi_type& d_pi_temp = get<::Tags::TempTensor<32, d_pi_type>>(vars);
  BenchmarkHelpers::copy_tensor(d_pi, make_not_null(&d_pi_temp));

  // d_phi
  d_phi_type& d_phi_temp = get<::Tags::TempTensor<33, d_phi_type>>(vars);
  BenchmarkHelpers::copy_tensor(d_phi, make_not_null(&d_phi_temp));

  // spacetime_metric
  spacetime_metric_type& spacetime_metric_temp =
      get<::Tags::TempTensor<34, spacetime_metric_type>>(vars);
  BenchmarkHelpers::copy_tensor(spacetime_metric,
                                make_not_null(&spacetime_metric_temp));

  // pi
  pi_type& pi_temp = get<::Tags::TempTensor<35, pi_type>>(vars);
  BenchmarkHelpers::copy_tensor(pi, make_not_null(&pi_temp));

  // phi
  phi_type& phi_temp = get<::Tags::TempTensor<36, phi_type>>(vars);
  BenchmarkHelpers::copy_tensor(phi, make_not_null(&phi_temp));

  // gamma0
  gamma0_type& gamma0_temp = get<::Tags::TempTensor<37, gamma0_type>>(vars);
  BenchmarkHelpers::copy_tensor(gamma0, make_not_null(&gamma0_temp));

  // gamma1
  gamma1_type& gamma1_temp = get<::Tags::TempTensor<38, gamma1_type>>(vars);
  BenchmarkHelpers::copy_tensor(gamma1, make_not_null(&gamma1_temp));

  // gamma2
  gamma2_type& gamma2_temp = get<::Tags::TempTensor<39, gamma2_type>>(vars);
  BenchmarkHelpers::copy_tensor(gamma2, make_not_null(&gamma2_temp));

  // gauge_function
  gauge_function_type& gauge_function_temp =
      get<::Tags::TempTensor<40, gauge_function_type>>(vars);
  BenchmarkHelpers::copy_tensor(gauge_function,
                                make_not_null(&gauge_function_temp));

  // spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function_temp =
      get<::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>>(vars);
  BenchmarkHelpers::copy_tensor(
      spacetime_deriv_gauge_function,
      make_not_null(&spacetime_deriv_gauge_function_temp));

  // pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial_temp =
      get<::Tags::TempTensor<42, pi_one_normal_spatial_type>>(vars);

  // phi_one_normal_spatial
  phi_one_normal_spatial_type& phi_one_normal_spatial_temp =
      get<::Tags::TempTensor<43, phi_one_normal_spatial_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_as_arg(
        make_not_null(&dt_spacetime_metric_temp), make_not_null(&dt_pi_temp),
        make_not_null(&dt_phi_temp), make_not_null(&temp_gamma1_temp),
        make_not_null(&temp_gamma2_temp), make_not_null(&gamma1gamma2_temp),
        make_not_null(&pi_two_normals_temp),
        make_not_null(&normal_dot_gauge_constraint_temp),
        make_not_null(&gamma1_plus_1_temp), make_not_null(&pi_one_normal_temp),
        make_not_null(&gauge_constraint_temp),
        make_not_null(&phi_two_normals_temp),
        make_not_null(&shift_dot_three_index_constraint_temp),
        make_not_null(&phi_one_normal_temp), make_not_null(&pi_2_up_temp),
        make_not_null(&three_index_constraint_temp),
        make_not_null(&phi_1_up_temp), make_not_null(&phi_3_up_temp),
        make_not_null(&christoffel_first_kind_3_up_temp),
        make_not_null(&lapse_temp), make_not_null(&shift_temp),
        make_not_null(&spatial_metric_temp),
        make_not_null(&inverse_spatial_metric_temp),
        make_not_null(&det_spatial_metric_temp),
        make_not_null(&inverse_spacetime_metric_temp),
        make_not_null(&christoffel_first_kind_temp),
        make_not_null(&christoffel_second_kind_temp),
        make_not_null(&trace_christoffel_temp),
        make_not_null(&normal_spacetime_vector_temp),
        make_not_null(&normal_spacetime_one_form_temp),
        make_not_null(&da_spacetime_metric_temp), d_spacetime_metric_temp,
        d_pi_temp, d_phi_temp, spacetime_metric_temp, pi_temp, phi_temp,
        gamma0_temp, gamma1_temp, gamma2_temp, gauge_function_temp,
        spacetime_deriv_gauge_function_temp,
        make_not_null(&pi_one_normal_spatial_temp),
        make_not_null(&phi_one_normal_spatial_temp));
    benchmark::DoNotOptimize(dt_spacetime_metric_temp);
    benchmark::DoNotOptimize(dt_pi_temp);
    benchmark::DoNotOptimize(dt_phi_temp);
    benchmark::ClobberMemory();
  }
}

// benchmark TE implementation, takes LHS as arg, equation terms not in buffer
void bench_tensorexpression_lhs_tensor_as_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
  // Set up input constant tensors in equations
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = static_cast<size_t>(state.range(0));
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars),
                          make_not_null(&generator),
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const spacetime_metric_type& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const phi_type& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const pi_type& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const d_spacetime_metric_type& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_phi_type& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_pi_type& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const size_t num_grid_points = mesh.number_of_grid_points();

  // Initialize non-const tensors to be filled by TensorExpression
  // implementation
  dt_spacetime_metric_type dt_spacetime_metric(num_grid_points);
  dt_pi_type dt_pi(num_grid_points);
  dt_phi_type dt_phi(num_grid_points);
  temp_gamma1_type temp_gamma1(num_grid_points);
  temp_gamma2_type temp_gamma2(num_grid_points);
  gamma1gamma2_type gamma1gamma2(num_grid_points);
  pi_two_normals_type pi_two_normals(num_grid_points);
  normal_dot_gauge_constraint_type normal_dot_gauge_constraint(num_grid_points);
  gamma1_plus_1_type gamma1_plus_1(num_grid_points);
  pi_one_normal_type pi_one_normal(num_grid_points);
  gauge_constraint_type gauge_constraint(num_grid_points);
  phi_two_normals_type phi_two_normals(num_grid_points);
  shift_dot_three_index_constraint_type shift_dot_three_index_constraint(
      num_grid_points);
  phi_one_normal_type phi_one_normal(num_grid_points);
  pi_2_up_type pi_2_up(num_grid_points);
  three_index_constraint_type three_index_constraint(num_grid_points);
  phi_1_up_type phi_1_up(num_grid_points);
  phi_3_up_type phi_3_up(num_grid_points);
  christoffel_first_kind_3_up_type christoffel_first_kind_3_up(num_grid_points);
  lapse_type lapse(num_grid_points);
  shift_type shift(num_grid_points);
  spatial_metric_type spatial_metric(num_grid_points);
  inverse_spatial_metric_type inverse_spatial_metric(num_grid_points);
  det_spatial_metric_type det_spatial_metric(num_grid_points);
  inverse_spacetime_metric_type inverse_spacetime_metric(num_grid_points);
  christoffel_first_kind_type christoffel_first_kind(num_grid_points);
  christoffel_second_kind_type christoffel_second_kind(num_grid_points);
  trace_christoffel_type trace_christoffel(num_grid_points);
  normal_spacetime_vector_type normal_spacetime_vector(num_grid_points);
  normal_spacetime_one_form_type normal_spacetime_one_form(num_grid_points);
  da_spacetime_metric_type da_spacetime_metric(num_grid_points);
  // TEs can't iterate over only spatial indices of a spacetime index yet, so
  // where this is needed for the dt_pi and dt_phi calculations, these tensors
  // will be used, which hold only the spatial components to enable writing the
  // equations as closely as possible to how they appear in the manual loops
  pi_one_normal_spatial_type pi_one_normal_spatial(num_grid_points);
  phi_one_normal_spatial_type phi_one_normal_spatial(num_grid_points);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
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

// benchmark TE implementation, takes LHS as arg, equation terms in buffer
void bench_tensorexpression_lhs_tensor_as_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
  // Set up input constant tensors in equations
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = static_cast<size_t>(state.range(0));
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars),
                          make_not_null(&generator),
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size),
      TestHelpers::gr::random_shift<Dim>(make_not_null(&generator),
                                         used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(make_not_null(&generator),
                                                  used_for_size));

  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const spacetime_metric_type& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const phi_type& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const pi_type& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const d_spacetime_metric_type& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_phi_type& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const d_pi_type& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const size_t num_grid_points = mesh.number_of_grid_points();

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

  // dt_spacetime_metric
  dt_spacetime_metric_type& dt_spacetime_metric_temp =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  // dt_pi
  dt_pi_type& dt_pi_temp = get<::Tags::TempTensor<1, dt_pi_type>>(vars);

  // dt_phi
  dt_phi_type& dt_phi_temp = get<::Tags::TempTensor<2, dt_phi_type>>(vars);

  // temp_gamma1
  temp_gamma1_type& temp_gamma1_temp =
      get<::Tags::TempTensor<3, temp_gamma1_type>>(vars);

  // temp_gamma2
  temp_gamma2_type& temp_gamma2_temp =
      get<::Tags::TempTensor<4, temp_gamma2_type>>(vars);

  // gamma1gamma2
  gamma1gamma2_type& gamma1gamma2_temp =
      get<::Tags::TempTensor<5, gamma1gamma2_type>>(vars);

  // pi_two_normals
  pi_two_normals_type& pi_two_normals_temp =
      get<::Tags::TempTensor<6, pi_two_normals_type>>(vars);

  // normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint_temp =
      get<::Tags::TempTensor<7, normal_dot_gauge_constraint_type>>(vars);

  // gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1_temp =
      get<::Tags::TempTensor<8, gamma1_plus_1_type>>(vars);

  // pi_one_normal
  pi_one_normal_type& pi_one_normal_temp =
      get<::Tags::TempTensor<9, pi_one_normal_type>>(vars);

  // gauge_constraint
  gauge_constraint_type& gauge_constraint_temp =
      get<::Tags::TempTensor<10, gauge_constraint_type>>(vars);

  // phi_two_normals
  phi_two_normals_type& phi_two_normals_temp =
      get<::Tags::TempTensor<11, phi_two_normals_type>>(vars);

  // shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint_temp =
      get<::Tags::TempTensor<12, shift_dot_three_index_constraint_type>>(vars);

  // phi_one_normal
  phi_one_normal_type& phi_one_normal_temp =
      get<::Tags::TempTensor<13, phi_one_normal_type>>(vars);

  // pi_2_up
  pi_2_up_type& pi_2_up_temp = get<::Tags::TempTensor<14, pi_2_up_type>>(vars);

  // three_index_constraint
  three_index_constraint_type& three_index_constraint_temp =
      get<::Tags::TempTensor<15, three_index_constraint_type>>(vars);

  // phi_1_up
  phi_1_up_type& phi_1_up_temp =
      get<::Tags::TempTensor<16, phi_1_up_type>>(vars);

  // phi_3_up
  phi_3_up_type& phi_3_up_temp =
      get<::Tags::TempTensor<17, phi_3_up_type>>(vars);

  // christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up_temp =
      get<::Tags::TempTensor<18, christoffel_first_kind_3_up_type>>(vars);

  // lapse
  lapse_type& lapse_temp = get<::Tags::TempTensor<19, lapse_type>>(vars);

  // shift
  shift_type& shift_temp = get<::Tags::TempTensor<20, shift_type>>(vars);

  // spatial_metric
  spatial_metric_type& spatial_metric_temp =
      get<::Tags::TempTensor<21, spatial_metric_type>>(vars);

  // inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric_temp =
      get<::Tags::TempTensor<22, inverse_spatial_metric_type>>(vars);

  // det_spatial_metric
  det_spatial_metric_type& det_spatial_metric_temp =
      get<::Tags::TempTensor<23, det_spatial_metric_type>>(vars);

  // inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric_temp =
      get<::Tags::TempTensor<24, inverse_spacetime_metric_type>>(vars);

  // christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind_temp =
      get<::Tags::TempTensor<25, christoffel_first_kind_type>>(vars);

  // christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind_temp =
      get<::Tags::TempTensor<26, christoffel_second_kind_type>>(vars);

  // trace_christoffel
  trace_christoffel_type& trace_christoffel_temp =
      get<::Tags::TempTensor<27, trace_christoffel_type>>(vars);

  // normal_spacetime_vector
  normal_spacetime_vector_type& normal_spacetime_vector_temp =
      get<::Tags::TempTensor<28, normal_spacetime_vector_type>>(vars);

  // normal_spacetime_one_form
  normal_spacetime_one_form_type& normal_spacetime_one_form_temp =
      get<::Tags::TempTensor<29, normal_spacetime_one_form_type>>(vars);

  // da_spacetime_metric
  da_spacetime_metric_type& da_spacetime_metric_temp =
      get<::Tags::TempTensor<30, da_spacetime_metric_type>>(vars);

  // d_spacetime_metric
  d_spacetime_metric_type& d_spacetime_metric_temp =
      get<::Tags::TempTensor<31, d_spacetime_metric_type>>(vars);
  BenchmarkHelpers::copy_tensor(d_spacetime_metric,
                                make_not_null(&d_spacetime_metric_temp));

  // d_pi
  d_pi_type& d_pi_temp = get<::Tags::TempTensor<32, d_pi_type>>(vars);
  BenchmarkHelpers::copy_tensor(d_pi, make_not_null(&d_pi_temp));

  // d_phi
  d_phi_type& d_phi_temp = get<::Tags::TempTensor<33, d_phi_type>>(vars);
  BenchmarkHelpers::copy_tensor(d_phi, make_not_null(&d_phi_temp));

  // spacetime_metric
  spacetime_metric_type& spacetime_metric_temp =
      get<::Tags::TempTensor<34, spacetime_metric_type>>(vars);
  BenchmarkHelpers::copy_tensor(spacetime_metric,
                                make_not_null(&spacetime_metric_temp));

  // pi
  pi_type& pi_temp = get<::Tags::TempTensor<35, pi_type>>(vars);
  BenchmarkHelpers::copy_tensor(pi, make_not_null(&pi_temp));

  // phi
  phi_type& phi_temp = get<::Tags::TempTensor<36, phi_type>>(vars);
  BenchmarkHelpers::copy_tensor(phi, make_not_null(&phi_temp));

  // gamma0
  gamma0_type& gamma0_temp = get<::Tags::TempTensor<37, gamma0_type>>(vars);
  BenchmarkHelpers::copy_tensor(gamma0, make_not_null(&gamma0_temp));

  // gamma1
  gamma1_type& gamma1_temp = get<::Tags::TempTensor<38, gamma1_type>>(vars);
  BenchmarkHelpers::copy_tensor(gamma1, make_not_null(&gamma1_temp));

  // gamma2
  gamma2_type& gamma2_temp = get<::Tags::TempTensor<39, gamma2_type>>(vars);
  BenchmarkHelpers::copy_tensor(gamma2, make_not_null(&gamma2_temp));

  // gauge_function
  gauge_function_type& gauge_function_temp =
      get<::Tags::TempTensor<40, gauge_function_type>>(vars);
  BenchmarkHelpers::copy_tensor(gauge_function,
                                make_not_null(&gauge_function_temp));

  // spacetime_deriv_gauge_function
  spacetime_deriv_gauge_function_type& spacetime_deriv_gauge_function_temp =
      get<::Tags::TempTensor<41, spacetime_deriv_gauge_function_type>>(vars);
  BenchmarkHelpers::copy_tensor(
      spacetime_deriv_gauge_function,
      make_not_null(&spacetime_deriv_gauge_function_temp));

  // pi_one_normal_spatial
  pi_one_normal_spatial_type& pi_one_normal_spatial_temp =
      get<::Tags::TempTensor<42, pi_one_normal_spatial_type>>(vars);

  // phi_one_normal_spatial
  phi_one_normal_spatial_type& phi_one_normal_spatial_temp =
      get<::Tags::TempTensor<43, phi_one_normal_spatial_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&dt_spacetime_metric_temp), make_not_null(&dt_pi_temp),
        make_not_null(&dt_phi_temp), make_not_null(&temp_gamma1_temp),
        make_not_null(&temp_gamma2_temp), make_not_null(&gamma1gamma2_temp),
        make_not_null(&pi_two_normals_temp),
        make_not_null(&normal_dot_gauge_constraint_temp),
        make_not_null(&gamma1_plus_1_temp), make_not_null(&pi_one_normal_temp),
        make_not_null(&gauge_constraint_temp),
        make_not_null(&phi_two_normals_temp),
        make_not_null(&shift_dot_three_index_constraint_temp),
        make_not_null(&phi_one_normal_temp), make_not_null(&pi_2_up_temp),
        make_not_null(&three_index_constraint_temp),
        make_not_null(&phi_1_up_temp), make_not_null(&phi_3_up_temp),
        make_not_null(&christoffel_first_kind_3_up_temp),
        make_not_null(&lapse_temp), make_not_null(&shift_temp),
        make_not_null(&spatial_metric_temp),
        make_not_null(&inverse_spatial_metric_temp),
        make_not_null(&det_spatial_metric_temp),
        make_not_null(&inverse_spacetime_metric_temp),
        make_not_null(&christoffel_first_kind_temp),
        make_not_null(&christoffel_second_kind_temp),
        make_not_null(&trace_christoffel_temp),
        make_not_null(&normal_spacetime_vector_temp),
        make_not_null(&normal_spacetime_one_form_temp),
        make_not_null(&da_spacetime_metric_temp), d_spacetime_metric_temp,
        d_pi_temp, d_phi_temp, spacetime_metric_temp, pi_temp, phi_temp,
        gamma0_temp, gamma1_temp, gamma2_temp, gauge_function_temp,
        spacetime_deriv_gauge_function_temp,
        make_not_null(&pi_one_normal_spatial_temp),
        make_not_null(&phi_one_normal_spatial_temp));
    benchmark::DoNotOptimize(dt_spacetime_metric_temp);
    benchmark::DoNotOptimize(dt_pi_temp);
    benchmark::DoNotOptimize(dt_phi_temp);
    benchmark::ClobberMemory();
  }
}

// Benchmark with each of these number of grid points for DataVector for a
// single dimension
const std::array<long int, 4> num_grid_points_1d_values = {2, 5, 8, 10};

// Benchmark manual implementations and TensorExpression implementations that:
// (i) take LHS tensor as an argument and do not use a buffer
// (ii) take LHS tensor as an argument and do use a buffer
BENCHMARK(bench_manual_tensor_equation_lhs_tensor_as_arg_without_buffer)
    ->Arg(num_grid_points_1d_values[0])
    ->Arg(num_grid_points_1d_values[1])
    ->Arg(num_grid_points_1d_values[2])
    ->Arg(num_grid_points_1d_values[3]);  // NOLINT
BENCHMARK(bench_manual_tensor_equation_lhs_tensor_as_arg_with_buffer)
    ->Arg(num_grid_points_1d_values[0])
    ->Arg(num_grid_points_1d_values[1])
    ->Arg(num_grid_points_1d_values[2])
    ->Arg(num_grid_points_1d_values[3]);  // NOLINT
BENCHMARK(bench_tensorexpression_lhs_tensor_as_arg_without_buffer)
    ->Arg(num_grid_points_1d_values[0])
    ->Arg(num_grid_points_1d_values[1])
    ->Arg(num_grid_points_1d_values[2])
    ->Arg(num_grid_points_1d_values[3]);  // NOLINT
BENCHMARK(bench_tensorexpression_lhs_tensor_as_arg_with_buffer)
    ->Arg(num_grid_points_1d_values[0])
    ->Arg(num_grid_points_1d_values[1])
    ->Arg(num_grid_points_1d_values[2])
    ->Arg(num_grid_points_1d_values[3]);  // NOLINT
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
