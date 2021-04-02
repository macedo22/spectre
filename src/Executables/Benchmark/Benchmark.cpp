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
using DataType = DataVector;
constexpr size_t Dim = 3;
using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
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

// benchmark manual implementation, equation terms not in buffer
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
  const DataType used_for_size(mesh.number_of_grid_points());

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

  InverseJacobian<DataType, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataType(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataType(mesh.number_of_grid_points(), 0.0);
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

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataType, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataType, Dim>>(
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
  const DataType used_for_size(mesh.number_of_grid_points());

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

  InverseJacobian<DataType, Dim, Frame::Logical, Frame::Inertial> inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataType(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataType(mesh.number_of_grid_points(), 0.0);
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

  const gamma0_type gamma0 = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma1_type gamma1 = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gamma2_type gamma2 = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const gauge_function_type gauge_function =
      make_with_random_values<tnsr::a<DataType, Dim>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const spacetime_deriv_gauge_function_type spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataType, Dim>>(
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
BENCHMARK(bench_tensorexpression_lhs_tensor_as_arg_without_buffer)
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
