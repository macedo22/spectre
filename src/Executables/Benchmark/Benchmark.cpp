// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <charm++.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
// #include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
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

using DataType = DataVector;
constexpr size_t Dim = 3;
using DerivativeFrame = Frame::Grid;
constexpr Spectral::Basis basis = Spectral::Basis::Legendre;
constexpr Spectral::Quadrature quadrature = Spectral::Quadrature::GaussLobatto;
constexpr double time = 3.4;

template <size_t Dim>
struct Kappa : db::SimpleTag {
  using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
};
template <size_t Dim>
struct Psi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
};

// clang-tidy: don't pass be non-const reference
void bench_partial_derivatives_3D_size_10(benchmark::State& state) {  // NOLINT
  constexpr const size_t pts_1d = 10;
  // constexpr const size_t Dim = 3;
  const Mesh<Dim> mesh{pts_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
  using Map3d =
      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;
  domain::CoordinateMap<Frame::ElementLogical, DerivativeFrame, Map3d> map(
      Map3d{map1d, map1d, map1d});

  using VarTags = tmpl::list<Kappa<Dim>, Psi<Dim>>;
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, DerivativeFrame>
      inv_jac = map.inv_jacobian(logical_coordinates(mesh));
  const auto grid_coords = map(logical_coordinates(mesh));
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);
  Variables<db::wrap_tags_in<Tags::deriv, VarTags, tmpl::size_t<Dim>,
                             DerivativeFrame>>
      result(mesh.number_of_grid_points());

  while (state.KeepRunning()) {
    // benchmark::DoNotOptimize(
    //     partial_derivatives(make_not_null(&result), vars, mesh, inv_jac));
    partial_derivatives(make_not_null(&result), vars, mesh, inv_jac);
    benchmark::ClobberMemory();
  }
}

void bench_partial_derivatives_unroll_3D_size_10(
    benchmark::State& state) {  // NOLINT
  constexpr const size_t pts_1d = 10;
  // constexpr const size_t Dim = 3;
  const Mesh<Dim> mesh{pts_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
  using Map3d =
      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;
  domain::CoordinateMap<Frame::ElementLogical, DerivativeFrame, Map3d> map(
      Map3d{map1d, map1d, map1d});

  using VarTags = tmpl::list<Kappa<Dim>, Psi<Dim>>;
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, DerivativeFrame>
      inv_jac = map.inv_jacobian(logical_coordinates(mesh));
  const auto grid_coords = map(logical_coordinates(mesh));
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);
  Variables<db::wrap_tags_in<Tags::deriv, VarTags, tmpl::size_t<Dim>,
                             DerivativeFrame>>
      result(mesh.number_of_grid_points());

  while (state.KeepRunning()) {
    // benchmark::DoNotOptimize(partial_derivatives_unroll(make_not_null(&result),
    //                                                     vars, mesh,
    //                                                     inv_jac));
    partial_derivatives_unroll(make_not_null(&result), vars, mesh, inv_jac);
    benchmark::ClobberMemory();
  }
}

void bench_datavector_mult_2_int_size_10(benchmark::State& state) {  // NOLINT
  // constexpr const size_t pts_1d = 10;
  constexpr const size_t num_points = 1000;
  DataVector dv_1{num_points, 0.0};
  DataVector dv_2{num_points, 0.0};
  // DataVector dv_3{num_points, 0.0};
  DataVector result{num_points, 0.0};

  double value = 1.0;
  for (size_t i = 0; i < num_points; i++) {
    dv_1[i] = value;
    dv_2[i] = 0.8 * value;
    ;
    // dv_3[i] = dv_1[i] + dv_2[i];
    result[i] = 1.1 * value - 0.2;
  }

  while (state.KeepRunning()) {
    // benchmark::DoNotOptimize(partial_derivatives_unroll(make_not_null(&result),
    //                                                     vars, mesh,
    //                                                     inv_jac));

    result += 2 * (dv_1 + dv_2);
    benchmark::ClobberMemory();
  }
}

void bench_datavector_mult_2_double_size_10(
    benchmark::State& state) {  // NOLINT
  // constexpr const size_t pts_1d = 10;
  constexpr const size_t num_points = 1000;
  DataVector dv_1{num_points, 0.0};
  DataVector dv_2{num_points, 0.0};
  // DataVector dv_3{num_points, 0.0};
  DataVector result{num_points, 0.0};

  double value = 1.0;
  for (size_t i = 0; i < num_points; i++) {
    dv_1[i] = value;
    dv_2[i] = 0.8 * value;
    ;
    // dv_3[i] = dv_1[i] + dv_2[i];
    result[i] = 1.1 * value - 0.2;
  }

  while (state.KeepRunning()) {
    // benchmark::DoNotOptimize(partial_derivatives_unroll(make_not_null(&result),
    //                                                     vars, mesh,
    //                                                     inv_jac));

    result += 2.0 * (dv_1 + dv_2);
    benchmark::ClobberMemory();
  }
}

// - manual implementation
// - LHS tensor is a function argument to implementation
// - equation terms are stored in buffer
void bench_manual_tensor_equation_lhs_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using dt_pi_type = typename BenchmarkImpl::dt_pi_type;
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using temp_gamma1_type = typename BenchmarkImpl::temp_gamma1_type;
  using temp_gamma2_type = typename BenchmarkImpl::temp_gamma2_type;
  using temp_gauge_function_type =
      typename BenchmarkImpl::temp_gauge_function_type;
  using temp_spacetime_deriv_gauge_function_type =
      typename BenchmarkImpl::temp_spacetime_deriv_gauge_function_type;
  using gamma1gamma2_type = typename BenchmarkImpl::gamma1gamma2_type;
  using half_pi_two_normals_type =
      typename BenchmarkImpl::half_pi_two_normals_type;
  using normal_dot_gauge_constraint_type =
      typename BenchmarkImpl::normal_dot_gauge_constraint_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using pi_one_normal_type = typename BenchmarkImpl::pi_one_normal_type;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using half_phi_two_normals_type =
      typename BenchmarkImpl::half_phi_two_normals_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using mesh_velocity_dot_three_index_constraint_type =
      typename BenchmarkImpl::mesh_velocity_dot_three_index_constraint_type;
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
  using inverse_spatial_metric_type =
      typename BenchmarkImpl::inverse_spatial_metric_type;
  using det_spatial_metric_type =
      typename BenchmarkImpl::det_spatial_metric_type;
  using sqrt_det_spatial_metric_type =
      typename BenchmarkImpl::sqrt_det_spatial_metric_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using christoffel_second_kind_type =
      typename BenchmarkImpl::christoffel_second_kind_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using normal_spacetime_vector_type =
      typename BenchmarkImpl::normal_spacetime_vector_type;
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
  using gauge_condition_type = typename BenchmarkImpl::gauge_condition_type;
  using mesh_type = typename BenchmarkImpl::mesh_type;
  using inertial_coords_type = typename BenchmarkImpl::inertial_coords_type;
  using inverse_jacobian_type = typename BenchmarkImpl::inverse_jacobian_type;
  using mesh_velocity_type = typename BenchmarkImpl::mesh_velocity_type;

  const size_t num_1d_grid_points = 10;
  const mesh_type mesh{num_1d_grid_points, basis, quadrature};
  const size_t num_grid_points = pow(num_1d_grid_points, 3);
  const DataType used_for_size = DataVector(num_grid_points, 0.0);
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const gauge_condition_type gauge_condition{};

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_spacetime_metric_type>,
      ::Tags::TempTensor<1, dt_pi_type>, ::Tags::TempTensor<2, dt_phi_type>,
      ::Tags::TempTensor<3, temp_gamma1_type>,
      ::Tags::TempTensor<4, temp_gamma2_type>,
      ::Tags::TempTensor<5, temp_gauge_function_type>,
      ::Tags::TempTensor<6, temp_spacetime_deriv_gauge_function_type>,
      ::Tags::TempTensor<7, gamma1gamma2_type>,
      ::Tags::TempTensor<8, half_pi_two_normals_type>,
      ::Tags::TempTensor<9, normal_dot_gauge_constraint_type>,
      ::Tags::TempTensor<10, gamma1_plus_1_type>,
      ::Tags::TempTensor<11, pi_one_normal_type>,
      ::Tags::TempTensor<12, gauge_constraint_type>,
      ::Tags::TempTensor<13, half_phi_two_normals_type>,
      ::Tags::TempTensor<14, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<15, mesh_velocity_dot_three_index_constraint_type>,
      ::Tags::TempTensor<16, phi_one_normal_type>,
      ::Tags::TempTensor<17, pi_2_up_type>,
      ::Tags::TempTensor<18, three_index_constraint_type>,
      ::Tags::TempTensor<19, phi_1_up_type>,
      ::Tags::TempTensor<20, phi_3_up_type>,
      ::Tags::TempTensor<21, christoffel_first_kind_3_up_type>,
      ::Tags::TempTensor<22, lapse_type>, ::Tags::TempTensor<23, shift_type>,
      ::Tags::TempTensor<24, inverse_spatial_metric_type>,
      ::Tags::TempTensor<25, det_spatial_metric_type>,
      ::Tags::TempTensor<26, sqrt_det_spatial_metric_type>,
      ::Tags::TempTensor<27, inverse_spacetime_metric_type>,
      ::Tags::TempTensor<28, christoffel_first_kind_type>,
      ::Tags::TempTensor<29, christoffel_second_kind_type>,
      ::Tags::TempTensor<30, trace_christoffel_type>,
      ::Tags::TempTensor<31, normal_spacetime_vector_type>,
      ::Tags::TempTensor<32, d_spacetime_metric_type>,
      ::Tags::TempTensor<33, d_pi_type>, ::Tags::TempTensor<34, d_phi_type>,
      ::Tags::TempTensor<35, spacetime_metric_type>,
      ::Tags::TempTensor<36, pi_type>, ::Tags::TempTensor<37, phi_type>,
      ::Tags::TempTensor<38, gamma0_type>, ::Tags::TempTensor<39, gamma1_type>,
      ::Tags::TempTensor<40, gamma2_type>,
      ::Tags::TempTensor<41, inertial_coords_type>,
      ::Tags::TempTensor<42, inverse_jacobian_type>,
      ::Tags::TempTensor<43, mesh_velocity_type>>>
      vars{num_grid_points};

  // RHS: d_spacetime_metric
  d_spacetime_metric_type& d_spacetime_metric =
      get<::Tags::TempTensor<32, d_spacetime_metric_type>>(vars);
  fill_with_values(make_not_null(&d_spacetime_metric));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<33, d_pi_type>>(vars);
  fill_with_values(make_not_null(&d_pi));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<34, d_phi_type>>(vars);
  fill_with_values(make_not_null(&d_phi));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<35, spacetime_metric_type>>(vars);
  fill_with_values(make_not_null(&spacetime_metric));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<36, pi_type>>(vars);
  fill_with_values(make_not_null(&pi));
  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<37, phi_type>>(vars);
  fill_with_values(make_not_null(&phi));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<38, gamma0_type>>(vars);
  fill_with_values(make_not_null(&gamma0));

  // RHS: gamma1
  gamma1_type& gamma1 = get<::Tags::TempTensor<39, gamma1_type>>(vars);
  fill_with_values(make_not_null(&gamma1));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<40, gamma2_type>>(vars);
  fill_with_values(make_not_null(&gamma2));

  // RHS: inertial_coords
  inertial_coords_type& inertial_coords =
      get<::Tags::TempTensor<41, inertial_coords_type>>(vars);
  fill_with_values(make_not_null(&inertial_coords));

  // RHS: inverse_jacobian
  inverse_jacobian_type& inverse_jacobian =
      get<::Tags::TempTensor<42, inverse_jacobian_type>>(vars);
  fill_with_values(make_not_null(&inverse_jacobian));

  // RHS: mesh_velocity
  mesh_velocity_type& mesh_velocity =
      get<::Tags::TempTensor<43, mesh_velocity_type>>(vars);
  fill_with_values(make_not_null(&mesh_velocity));

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

  // LHS: temp_gauge_function
  temp_gauge_function_type& temp_gauge_function =
      get<::Tags::TempTensor<5, temp_gauge_function_type>>(vars);

  // LHS: temp_spacetime_deriv_gauge_function
  temp_spacetime_deriv_gauge_function_type&
      temp_spacetime_deriv_gauge_function =
          get<::Tags::TempTensor<6, temp_spacetime_deriv_gauge_function_type>>(
              vars);

  // LHS: gamma1gamma2
  gamma1gamma2_type& gamma1gamma2 =
      get<::Tags::TempTensor<7, gamma1gamma2_type>>(vars);

  // LHS: half_pi_two_normals
  half_pi_two_normals_type& half_pi_two_normals =
      get<::Tags::TempTensor<8, half_pi_two_normals_type>>(vars);

  // LHS: normal_dot_gauge_constraint
  normal_dot_gauge_constraint_type& normal_dot_gauge_constraint =
      get<::Tags::TempTensor<9, normal_dot_gauge_constraint_type>>(vars);

  // LHS: gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1 =
      get<::Tags::TempTensor<10, gamma1_plus_1_type>>(vars);

  // LHS: pi_one_normal
  pi_one_normal_type& pi_one_normal =
      get<::Tags::TempTensor<11, pi_one_normal_type>>(vars);

  // LHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<12, gauge_constraint_type>>(vars);

  // LHS: half_phi_two_normals
  half_phi_two_normals_type& half_phi_two_normals =
      get<::Tags::TempTensor<13, half_phi_two_normals_type>>(vars);

  // LHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<14, shift_dot_three_index_constraint_type>>(vars);

  // LHS: mesh_velocity_dot_three_index_constraint
  mesh_velocity_dot_three_index_constraint_type&
      mesh_velocity_dot_three_index_constraint = get<::Tags::TempTensor<
          15, mesh_velocity_dot_three_index_constraint_type>>(vars);

  // LHS: phi_one_normal
  phi_one_normal_type& phi_one_normal =
      get<::Tags::TempTensor<16, phi_one_normal_type>>(vars);

  // LHS: pi_2_up
  pi_2_up_type& pi_2_up = get<::Tags::TempTensor<17, pi_2_up_type>>(vars);

  // LHS: three_index_constraint
  three_index_constraint_type& three_index_constraint =
      get<::Tags::TempTensor<18, three_index_constraint_type>>(vars);

  // LHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<19, phi_1_up_type>>(vars);

  // LHS: phi_3_up
  phi_3_up_type& phi_3_up = get<::Tags::TempTensor<20, phi_3_up_type>>(vars);

  // LHS: christoffel_first_kind_3_up
  christoffel_first_kind_3_up_type& christoffel_first_kind_3_up =
      get<::Tags::TempTensor<21, christoffel_first_kind_3_up_type>>(vars);

  // LHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<22, lapse_type>>(vars);

  // LHS: shift
  shift_type& shift = get<::Tags::TempTensor<23, shift_type>>(vars);

  // LHS: inverse_spatial_metric
  inverse_spatial_metric_type& inverse_spatial_metric =
      get<::Tags::TempTensor<24, inverse_spatial_metric_type>>(vars);

  // LHS: det_spatial_metric
  det_spatial_metric_type& det_spatial_metric =
      get<::Tags::TempTensor<25, det_spatial_metric_type>>(vars);

  // LHS: sqrt_det_spatial_metric
  sqrt_det_spatial_metric_type& sqrt_det_spatial_metric =
      get<::Tags::TempTensor<26, sqrt_det_spatial_metric_type>>(vars);

  // LHS: inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<27, inverse_spacetime_metric_type>>(vars);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<28, christoffel_first_kind_type>>(vars);

  // LHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<29, christoffel_second_kind_type>>(vars);

  // LHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<30, trace_christoffel_type>>(vars);

  // LHS: normal_spacetime_vector
  normal_spacetime_vector_type& normal_spacetime_vector =
      get<::Tags::TempTensor<31, normal_spacetime_vector_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_spacetime_metric), make_not_null(&dt_pi),
        make_not_null(&dt_phi), make_not_null(&temp_gamma1),
        make_not_null(&temp_gamma2), make_not_null(&temp_gauge_function),
        make_not_null(&temp_spacetime_deriv_gauge_function),
        make_not_null(&gamma1gamma2), make_not_null(&half_pi_two_normals),
        make_not_null(&normal_dot_gauge_constraint),
        make_not_null(&gamma1_plus_1), make_not_null(&pi_one_normal),
        make_not_null(&gauge_constraint), make_not_null(&half_phi_two_normals),
        make_not_null(&shift_dot_three_index_constraint),
        make_not_null(&mesh_velocity_dot_three_index_constraint),
        make_not_null(&phi_one_normal), make_not_null(&pi_2_up),
        make_not_null(&three_index_constraint), make_not_null(&phi_1_up),
        make_not_null(&phi_3_up), make_not_null(&christoffel_first_kind_3_up),
        make_not_null(&lapse), make_not_null(&shift),
        make_not_null(&inverse_spatial_metric),
        make_not_null(&det_spatial_metric),
        make_not_null(&sqrt_det_spatial_metric),
        make_not_null(&inverse_spacetime_metric),
        make_not_null(&christoffel_first_kind),
        make_not_null(&christoffel_second_kind),
        make_not_null(&trace_christoffel),
        make_not_null(&normal_spacetime_vector), d_spacetime_metric, d_pi,
        d_phi, spacetime_metric, pi, phi, gamma0, gamma1, gamma2,
        gauge_condition, mesh, time, inertial_coords, inverse_jacobian,
        mesh_velocity);
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::DoNotOptimize(dt_phi);
    benchmark::ClobberMemory();
  }
}

// BENCHMARK(bench_partial_derivatives_3D_size_10);         // NOLINT
// BENCHMARK(bench_partial_derivatives_unroll_3D_size_10);  // NOLINT
// BENCHMARK(bench_datavector_mult_2_int_size_10);     // NOLINT
// BENCHMARK(bench_datavector_mult_2_double_size_10);  // NOLINT
BENCHMARK(bench_manual_tensor_equation_lhs_arg_with_buffer);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
