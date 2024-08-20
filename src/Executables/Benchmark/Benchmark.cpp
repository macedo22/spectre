// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <array>
#include <benchmark.h>
#include <cassert>
#pragma GCC diagnostic pop
#include <charm++.h>
#include <cmath>
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
// #include "Executables/Benchmark/BenchmarkedImpls.hpp"
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

namespace {
#define assertm(exp, msg) assert(((void)msg, exp))

constexpr size_t Dim = 3;
constexpr Spectral::Basis basis = BenchmarkImpl::basis;
constexpr Spectral::Quadrature quadrature = BenchmarkImpl::quadrature;

using DerivativeFrame = BenchmarkImpl::DerivativeFrame;
// quantities for computing partial derivatives
using Kappa = BenchmarkImpl::Kappa<Dim>;
using Psi = BenchmarkImpl::Psi<Dim>;

// GH vars types
using dt_spacetime_metric_type = BenchmarkImpl::dt_spacetime_metric_type<Dim>;
using dt_pi_type = BenchmarkImpl::dt_pi_type<Dim>;
using dt_phi_type = BenchmarkImpl::dt_phi_type<Dim>;
using temp_gamma1_type = BenchmarkImpl::temp_gamma1_type<Dim>;
using temp_gamma2_type = BenchmarkImpl::temp_gamma2_type<Dim>;
using temp_gauge_function_type = BenchmarkImpl::temp_gauge_function_type<Dim>;
using temp_spacetime_deriv_gauge_function_type =
    BenchmarkImpl::temp_spacetime_deriv_gauge_function_type<Dim>;
using gamma1gamma2_type = BenchmarkImpl::gamma1gamma2_type<Dim>;
using half_pi_two_normals_type = BenchmarkImpl::half_pi_two_normals_type<Dim>;
using normal_dot_gauge_constraint_type =
    BenchmarkImpl::normal_dot_gauge_constraint_type<Dim>;
using gamma1_plus_1_type = BenchmarkImpl::gamma1_plus_1_type<Dim>;
using pi_one_normal_type = BenchmarkImpl::pi_one_normal_type<Dim>;
using gauge_constraint_type = BenchmarkImpl::gauge_constraint_type<Dim>;
using half_phi_two_normals_type = BenchmarkImpl::half_phi_two_normals_type<Dim>;
using shift_dot_three_index_constraint_type =
    BenchmarkImpl::shift_dot_three_index_constraint_type<Dim>;
using mesh_velocity_dot_three_index_constraint_type =
    BenchmarkImpl::mesh_velocity_dot_three_index_constraint_type<Dim>;
using phi_one_normal_type = BenchmarkImpl::phi_one_normal_type<Dim>;
using pi_2_up_type = BenchmarkImpl::pi_2_up_type<Dim>;
using three_index_constraint_type =
    BenchmarkImpl::three_index_constraint_type<Dim>;
using phi_1_up_type = BenchmarkImpl::phi_1_up_type<Dim>;
using phi_3_up_type = BenchmarkImpl::phi_3_up_type<Dim>;
using christoffel_first_kind_3_up_type =
    BenchmarkImpl::christoffel_first_kind_3_up_type<Dim>;
using lapse_type = BenchmarkImpl::lapse_type<Dim>;
using shift_type = BenchmarkImpl::shift_type<Dim>;
using inverse_spatial_metric_type =
    BenchmarkImpl::inverse_spatial_metric_type<Dim>;
using det_spatial_metric_type = BenchmarkImpl::det_spatial_metric_type<Dim>;
using sqrt_det_spatial_metric_type =
    BenchmarkImpl::sqrt_det_spatial_metric_type<Dim>;
using inverse_spacetime_metric_type =
    BenchmarkImpl::inverse_spacetime_metric_type<Dim>;
using christoffel_first_kind_type =
    BenchmarkImpl::christoffel_first_kind_type<Dim>;
using christoffel_second_kind_type =
    BenchmarkImpl::christoffel_second_kind_type<Dim>;
using trace_christoffel_type = BenchmarkImpl::trace_christoffel_type<Dim>;
using normal_spacetime_vector_type =
    BenchmarkImpl::normal_spacetime_vector_type<Dim>;
using d_spacetime_metric_type = BenchmarkImpl::d_spacetime_metric_type<Dim>;
using d_pi_type = BenchmarkImpl::d_pi_type<Dim>;
using d_phi_type = BenchmarkImpl::d_phi_type<Dim>;
using spacetime_metric_type = BenchmarkImpl::spacetime_metric_type<Dim>;
using pi_type = BenchmarkImpl::pi_type<Dim>;
using phi_type = BenchmarkImpl::phi_type<Dim>;
using gamma0_type = BenchmarkImpl::gamma0_type<Dim>;
using gamma1_type = BenchmarkImpl::gamma1_type<Dim>;
using gamma2_type = BenchmarkImpl::gamma2_type<Dim>;
using gauge_condition_type = BenchmarkImpl::gauge_condition_type<Dim>;
using inertial_coords_type = BenchmarkImpl::inertial_coords_type<Dim>;
using inverse_jacobian_type = BenchmarkImpl::inverse_jacobian_type<Dim>;
using mesh_velocity_type = BenchmarkImpl::mesh_velocity_type<Dim>;

// clang-tidy: don't pass be non-const reference
void bench_partial_derivatives(benchmark::State& state) {  // NOLINT
  const std::array<size_t, Dim> extents = {
      {static_cast<size_t>(state.range(1)), static_cast<size_t>(state.range(2)),
       static_cast<size_t>(state.range(3))}};
  const size_t num_3d_points = static_cast<size_t>(state.range(0));
  assertm(num_3d_points == extents[0] * extents[1] * extents[2],
          "Num 3D points does not match the product of the three 1D points");
  const Mesh<Dim> mesh{extents, basis, quadrature};
  domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
  using Map3d =
      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;
  domain::CoordinateMap<Frame::ElementLogical, DerivativeFrame, Map3d> map(
      Map3d{map1d, map1d, map1d});

  using VarTags = gh_evolution_vars_tags<Dim>;
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, DerivativeFrame>
      inv_jac = map.inv_jacobian(logical_coordinates(mesh));
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(partial_derivatives<VarTags>(vars, mesh, inv_jac));
    benchmark::ClobberMemory();
  }
}

// clang-tidy: don't pass be non-const reference
void bench_logical_partial_derivatives(benchmark::State& state) {  // NOLINT
  const std::array<size_t, Dim> extents = {
      {static_cast<size_t>(state.range(1)), static_cast<size_t>(state.range(2)),
       static_cast<size_t>(state.range(3))}};
  const size_t num_3d_points = static_cast<size_t>(state.range(0));
  assertm(num_3d_points == extents[0] * extents[1] * extents[2],
          "Num 3D points does not match the product of the three 1D points");
  const Mesh<Dim> mesh{extents, basis, quadrature};

  using VarTags = gh_evolution_vars_tags<Dim>;
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(logical_partial_derivatives<VarTags>(vars, mesh));
    benchmark::ClobberMemory();
  }
}

void bench_og_time_derivative(benchmark::State& state) {  // NOLINT
  const std::array<size_t, Dim> extents = {
      {static_cast<size_t>(state.range(1)), static_cast<size_t>(state.range(2)),
       static_cast<size_t>(state.range(3))}};
  const size_t num_3d_points = static_cast<size_t>(state.range(0));
  assertm(num_3d_points == extents[0] * extents[1] * extents[2],
          "Num 3D points does not match the product of the three 1D points");
  const Mesh<Dim> mesh{extents, basis, qudrature};
  const size_t num_grid_points = mesh.number_of_grid_points();
  const DataVector used_for_size = DataVector(num_grid_points, 0.0);
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
    gh::OgTimeDerivative<Dim>::apply(
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

void bench_time_derivative(benchmark::State& state) {  // NOLINT
  const std::array<size_t, Dim> extents = {
      {static_cast<size_t>(state.range(1)), static_cast<size_t>(state.range(2)),
       static_cast<size_t>(state.range(3))}};
  const size_t num_3d_points = static_cast<size_t>(state.range(0));
  assertm(num_3d_points == extents[0] * extents[1] * extents[2],
          "Num 3D points does not match the product of the three 1D points");
  (void)num_3d_points;
  const Mesh<Dim> mesh{extents, basis, quadrature};
  const size_t num_grid_points = mesh.number_of_grid_points();
  const DataVector used_for_size = DataVector(num_grid_points, 0.0);
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
      ::Tags::TempTensor<32, logical_shift_type>,
      ::Tags::TempTensor<33, inverse_spatial_metric_logical_1_type>,
      ::Tags::TempTensor<34, logical_mesh_velocity_type>,
      ::Tags::TempTensor<35, shift_dot_d_spacetime_metric_type>,
      ::Tags::TempTensor<36, shift_dot_phi_type>,
      ::Tags::TempTensor<37, mesh_velocity_dot_phi_type>,
      ::Tags::TempTensor<38, mesh_velocity_dot_d_spacetime_metric_type>,
      ::Tags::TempTensor<39, upper_gauge_function_type>,
      ::Tags::TempTensor<
          40, gamma2_logical_d_spacetime_metric_minus_logical_d_pi_type>,
      ::Tags::TempTensor<41, spacetime_metric_type>,
      ::Tags::TempTensor<42, pi_type>, ::Tags::TempTensor<43, phi_type>,
      ::Tags::TempTensor<44, gamma0_type>, ::Tags::TempTensor<45, gamma1_type>,
      ::Tags::TempTensor<46, gamma2_type>,
      ::Tags::TempTensor<47, inertial_coords_type>,
      ::Tags::TempTensor<48, inverse_jacobian_type>,
      ::Tags::TempTensor<49, mesh_velocity_type>>>
      vars{num_grid_points};

  const auto logical_partial_derivs =
      logical_partial_derivatives<gh_evolution_vars_tags<Dim>>(vars, mesh);

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<41, spacetime_metric_type>>(vars);
  fill_with_values(make_not_null(&spacetime_metric));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<42, pi_type>>(vars);
  fill_with_values(make_not_null(&pi));
  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<43, phi_type>>(vars);
  fill_with_values(make_not_null(&phi));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<44, gamma0_type>>(vars);
  fill_with_values(make_not_null(&gamma0));

  // RHS: gamma1
  gamma1_type& gamma1 = get<::Tags::TempTensor<45, gamma1_type>>(vars);
  fill_with_values(make_not_null(&gamma1));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<46, gamma2_type>>(vars);
  fill_with_values(make_not_null(&gamma2));

  // RHS: inertial_coords
  inertial_coords_type& inertial_coords =
      get<::Tags::TempTensor<47, inertial_coords_type>>(vars);
  fill_with_values(make_not_null(&inertial_coords));

  // RHS: inverse_jacobian
  inverse_jacobian_type& inverse_jacobian =
      get<::Tags::TempTensor<48, inverse_jacobian_type>>(vars);
  fill_with_values(make_not_null(&inverse_jacobian));

  // RHS: mesh_velocity
  mesh_velocity_type& mesh_velocity =
      get<::Tags::TempTensor<49, mesh_velocity_type>>(vars);
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

  // LHS: logical_shift
  logical_shift_type& logical_shift =
      get<::Tags::TempTensor<32, logical_shift_type>>(vars);

  // LHS: inverse_spatial_metric_logical_1
  inverse_spatial_metric_logical_1_type& inverse_spatial_metric_logical_1 =
      get<::Tags::TempTensor<33, inverse_spatial_metric_logical_1_type>>(vars);

  // LHS: logical_mesh_velocity
  logical_mesh_velocity_type& logical_mesh_velocity =
      get<::Tags::TempTensor<34, logical_mesh_velocity_type>>(vars);

  // LHS: shift_dot_d_spacetime_metric
  shift_dot_d_spacetime_metric_type& shift_dot_d_spacetime_metric =
      get<::Tags::TempTensor<35, shift_dot_d_spacetime_metric_type>>(vars);

  // LHS: shift_dot_phi
  shift_dot_phi_type& shift_dot_phi =
      get<::Tags::TempTensor<36, shift_dot_phi_type>>(vars);

  // LHS: mesh_velocity_dot_phi
  mesh_velocity_dot_phi_type& mesh_velocity_dot_phi =
      get<::Tags::TempTensor<37, mesh_velocity_dot_phi_type>>(vars);

  // LHS: mesh_velocity_dot_d_spacetime_metric
  mesh_velocity_dot_d_spacetime_metric_type&
      mesh_velocity_dot_d_spacetime_metric = get<
          ::Tags::TempTensor<38, mesh_velocity_dot_d_spacetime_metric_type>>(
          vars);

  // LHS: upper_gauge_function
  upper_gauge_function_type& upper_gauge_function =
      get<::Tags::TempTensor<39, upper_gauge_function_type>>(vars);

  // LHS: gamma2_logical_d_spacetime_metric_minus_logical_d_pi
  gamma2_logical_d_spacetime_metric_minus_logical_d_pi_type&
      gamma2_logical_d_spacetime_metric_minus_logical_d_pi =
          get<::Tags::TempTensor<
              40, gamma2_logical_d_spacetime_metric_minus_logical_d_pi_type>>(
              vars);

  for (auto _ : state) {
    gh::TimeDerivative<Dim>::apply(
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
        make_not_null(&normal_spacetime_vector), make_not_null(&logical_shift),
        make_not_null(&inverse_spatial_metric_logical_1),
        make_not_null(&logical_mesh_velocity),
        make_not_null(&shift_dot_d_spacetime_metric),
        make_not_null(&shift_dot_phi), make_not_null(&mesh_velocity_dot_phi),
        make_not_null(&mesh_velocity_dot_d_spacetime_metric),
        make_not_null(&upper_gauge_function),
        make_not_null(&gamma2_logical_d_spacetime_metric_minus_logical_d_pi),
        logical_partial_derivs, spacetime_metric, pi, phi, gamma0, gamma1,
        gamma2, gauge_condition, mesh, time, inertial_coords, inverse_jacobian,
        mesh_velocity);
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::DoNotOptimize(dt_pi);
    benchmark::DoNotOptimize(dt_phi);
    benchmark::ClobberMemory();
  }
}

// Each DataVector case is run with each number of grid points
constexpr size_t num_cases = 11;
constexpr std::array<std::array<size_t, 3>, num_cases> extents{{
    {{2, 2, 2}},     // 8
    {{4, 2, 2}},     // 16
    {{4, 4, 2}},     // 32
    {{4, 4, 4}},     // 64
    {{8, 4, 4}},     // 128
    {{8, 8, 4}},     // 256
    {{8, 8, 8}},     // 512
    {{16, 8, 8}},    // 1024
    {{16, 16, 8}},   // 2048
    {{16, 16, 16}},  // 4096
    {{20, 20, 20}}   // 8000
}};

constexpr std::array<long int, num_cases> num_3d_grid_point_values{{
    extents[0][0] * extents[0][1] * extents[0][2],    // 8
    extents[1][0] * extents[1][1] * extents[1][2],    // 16
    extents[2][0] * extents[2][1] * extents[2][2],    // 32
    extents[3][0] * extents[3][1] * extents[3][2],    // 64
    extents[4][0] * extents[4][1] * extents[4][2],    // 128
    extents[5][0] * extents[5][1] * extents[5][2],    // 256
    extents[6][0] * extents[6][1] * extents[6][2],    // 512
    extents[7][0] * extents[7][1] * extents[7][2],    // 1024
    extents[8][0] * extents[8][1] * extents[8][2],    // 2048
    extents[9][0] * extents[9][1] * extents[9][2],    // 4096
    extents[10][0] * extents[10][1] * extents[10][2]  // 8000
}};

void run_benchmarks() {
  const std::string partial_derivatives_benchmark_name =
      "partial_derivatives/3D/num_3d_points";
  BENCHMARK(bench_partial_derivatives)
      ->Name(partial_derivatives_benchmark_name)
      ->Args({num_3d_grid_point_values[0], extents[0][0], extents[0][1],
              extents[0][2]})
      ->Args({num_3d_grid_point_values[1], extents[1][0], extents[1][1],
              extents[1][2]})
      ->Args({num_3d_grid_point_values[2], extents[2][0], extents[2][1],
              extents[2][2]})
      ->Args({num_3d_grid_point_values[3], extents[3][0], extents[3][1],
              extents[3][2]})
      ->Args({num_3d_grid_point_values[4], extents[4][0], extents[4][1],
              extents[4][2]})
      ->Args({num_3d_grid_point_values[5], extents[5][0], extents[5][1],
              extents[5][2]})
      ->Args({num_3d_grid_point_values[6], extents[6][0], extents[6][1],
              extents[6][2]})
      ->Args({num_3d_grid_point_values[7], extents[7][0], extents[7][1],
              extents[7][2]})
      ->Args({num_3d_grid_point_values[8], extents[8][0], extents[8][1],
              extents[8][2]})
      ->Args({num_3d_grid_point_values[9], extents[9][0], extents[9][1],
              extents[9][2]})
      ->Args({num_3d_grid_point_values[10], extents[10][0], extents[10][1],
              extents[10][2]});

  const std::string logical_partial_derivatives_benchmark_name =
      "logical_partial_derivatives/3D/num_3d_points";
  BENCHMARK(bench_logical_partial_derivatives)
      ->Name(logical_partial_derivatives_benchmark_name)
      ->Args({num_3d_grid_point_values[0], extents[0][0], extents[0][1],
              extents[0][2]})
      ->Args({num_3d_grid_point_values[1], extents[1][0], extents[1][1],
              extents[1][2]})
      ->Args({num_3d_grid_point_values[2], extents[2][0], extents[2][1],
              extents[2][2]})
      ->Args({num_3d_grid_point_values[3], extents[3][0], extents[3][1],
              extents[3][2]})
      ->Args({num_3d_grid_point_values[4], extents[4][0], extents[4][1],
              extents[4][2]})
      ->Args({num_3d_grid_point_values[5], extents[5][0], extents[5][1],
              extents[5][2]})
      ->Args({num_3d_grid_point_values[6], extents[6][0], extents[6][1],
              extents[6][2]})
      ->Args({num_3d_grid_point_values[7], extents[7][0], extents[7][1],
              extents[7][2]})
      ->Args({num_3d_grid_point_values[8], extents[8][0], extents[8][1],
              extents[8][2]})
      ->Args({num_3d_grid_point_values[9], extents[9][0], extents[9][1],
              extents[9][2]})
      ->Args({num_3d_grid_point_values[10], extents[10][0], extents[10][1],
              extents[10][2]});

  const std::string og_time_derivative_benchmark_name =
      "og_time_derivative/3D/num_3d_points";
  BENCHMARK(bench_og_time_derivative)
      ->Name(og_time_derivative_benchmark_name)
      ->Args({num_3d_grid_point_values[0], extents[0][0], extents[0][1],
              extents[0][2]})
      ->Args({num_3d_grid_point_values[1], extents[1][0], extents[1][1],
              extents[1][2]})
      ->Args({num_3d_grid_point_values[2], extents[2][0], extents[2][1],
              extents[2][2]})
      ->Args({num_3d_grid_point_values[3], extents[3][0], extents[3][1],
              extents[3][2]})
      ->Args({num_3d_grid_point_values[4], extents[4][0], extents[4][1],
              extents[4][2]})
      ->Args({num_3d_grid_point_values[5], extents[5][0], extents[5][1],
              extents[5][2]})
      ->Args({num_3d_grid_point_values[6], extents[6][0], extents[6][1],
              extents[6][2]})
      ->Args({num_3d_grid_point_values[7], extents[7][0], extents[7][1],
              extents[7][2]})
      ->Args({num_3d_grid_point_values[8], extents[8][0], extents[8][1],
              extents[8][2]})
      ->Args({num_3d_grid_point_values[9], extents[9][0], extents[9][1],
              extents[9][2]})
      ->Args({num_3d_grid_point_values[10], extents[10][0], extents[10][1],
              extents[10][2]});

  const std::string time_derivative_benchmark_name =
      "time_derivative/3D/num_3d_points";
  BENCHMARK(bench_time_derivative)
      ->Name(time_derivative_benchmark_name)
      ->Args({num_3d_grid_point_values[0], extents[0][0], extents[0][1],
              extents[0][2]})
      ->Args({num_3d_grid_point_values[1], extents[1][0], extents[1][1],
              extents[1][2]})
      ->Args({num_3d_grid_point_values[2], extents[2][0], extents[2][1],
              extents[2][2]})
      ->Args({num_3d_grid_point_values[3], extents[3][0], extents[3][1],
              extents[3][2]})
      ->Args({num_3d_grid_point_values[4], extents[4][0], extents[4][1],
              extents[4][2]})
      ->Args({num_3d_grid_point_values[5], extents[5][0], extents[5][1],
              extents[5][2]})
      ->Args({num_3d_grid_point_values[6], extents[6][0], extents[6][1],
              extents[6][2]})
      ->Args({num_3d_grid_point_values[7], extents[7][0], extents[7][1],
              extents[7][2]})
      ->Args({num_3d_grid_point_values[8], extents[8][0], extents[8][1],
              extents[8][2]})
      ->Args({num_3d_grid_point_values[9], extents[9][0], extents[9][1],
              extents[9][2]})
      ->Args({num_3d_grid_point_values[10], extents[10][0], extents[10][1],
              extents[10][2]});
}
// BENCHMARK(bench_all_gradient);  // NOLINT
}  // namespace

// // Ignore the warning about an extra ';' because some versions of benchmark
// // require it
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wpedantic"
// BENCHMARK_MAIN();
// #pragma GCC diagnostic pop
// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
int main(int argc, char** argv) {
  run_benchmarks();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
#pragma GCC diagnostic pop
