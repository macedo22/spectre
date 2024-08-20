// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <array>
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <charm++.h>
#include <initializer_list>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Executables/Benchmark/BenchmarkImpl.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
using Function = BenchmarkHelpers::Function;

// Function, Dim, and grid points to benchmark for this specific cpp
constexpr Function FunctionToRun = Function::OgTimeDerivative;
constexpr size_t Dim = 3;
constexpr std::array<std::array<size_t, Dim>, 1> extents_to_run{{{{2, 2, 2}}}};
constexpr size_t total_num_cases = extents_to_run.size();
constexpr size_t num_cases_to_run = total_num_cases;

// General settings
constexpr Spectral::Basis basis = BenchmarkImpl::basis;
constexpr Spectral::Quadrature quadrature = BenchmarkImpl::quadrature;
constexpr double time = BenchmarkImpl::time;

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

// new temporaries used in new version of time derivative
using logical_shift_type = BenchmarkImpl::logical_shift_type<Dim>;
using inverse_spatial_metric_logical_1_type =
    BenchmarkImpl::inverse_spatial_metric_logical_1_type<Dim>;
using logical_mesh_velocity_type =
    BenchmarkImpl::logical_mesh_velocity_type<Dim>;
using shift_dot_d_spacetime_metric_type =
    BenchmarkImpl::shift_dot_d_spacetime_metric_type<Dim>;
using shift_dot_phi_type = BenchmarkImpl::shift_dot_phi_type<Dim>;
using mesh_velocity_dot_phi_type =
    BenchmarkImpl::mesh_velocity_dot_phi_type<Dim>;
using mesh_velocity_dot_d_spacetime_metric_type =
    BenchmarkImpl::mesh_velocity_dot_d_spacetime_metric_type<Dim>;
using upper_gauge_function_type = BenchmarkImpl::upper_gauge_function_type<Dim>;
using gamma2_logical_d_spacetime_metric_minus_logical_d_pi_type =
    BenchmarkImpl::gamma2_logical_d_spacetime_metric_minus_logical_d_pi_type<
        Dim>;

void bench(benchmark::State& state) {  // NOLINT
  const std::array<size_t, Dim> extents =
      BenchmarkHelpers::get_extents_from_state<Dim>(state);
  const size_t total_num_grid_points = static_cast<size_t>(state.range(0));
  (void)total_num_grid_points;
  assertm(
      total_grid_points_is_product_of_extents(extents, total_num_grid_points),
      "Total number of grid points is not equal to the product of the extents");

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
  BenchmarkHelpers::assign_unique_values_to_tensor(
      make_not_null(&d_spacetime_metric));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<33, d_pi_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&d_pi));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<34, d_phi_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&d_phi));

  // RHS: spacetime_metric
  spacetime_metric_type& spacetime_metric =
      get<::Tags::TempTensor<35, spacetime_metric_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(
      make_not_null(&spacetime_metric));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<36, pi_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&pi));
  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<37, phi_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&phi));

  // RHS: gamma0
  gamma0_type& gamma0 = get<::Tags::TempTensor<38, gamma0_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&gamma0));

  // RHS: gamma1
  gamma1_type& gamma1 = get<::Tags::TempTensor<39, gamma1_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&gamma1));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<40, gamma2_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(make_not_null(&gamma2));

  // RHS: inertial_coords
  inertial_coords_type& inertial_coords =
      get<::Tags::TempTensor<41, inertial_coords_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(
      make_not_null(&inertial_coords));

  // RHS: inverse_jacobian
  inverse_jacobian_type& inverse_jacobian =
      get<::Tags::TempTensor<42, inverse_jacobian_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(
      make_not_null(&inverse_jacobian));

  // RHS: mesh_velocity
  mesh_velocity_type& mesh_velocity =
      get<::Tags::TempTensor<43, mesh_velocity_type>>(vars);
  BenchmarkHelpers::assign_unique_values_to_tensor(
      make_not_null(&mesh_velocity));

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

template <Function Func, size_t NumberOfCases, size_t Dimension>
struct run_benchmark_case;

template <Function Func, size_t NumberOfCases>
struct run_benchmark_case<Func, NumberOfCases, 1> {
  template <size_t I>
  static void apply(
      const std::array<std::array<size_t, 1>, NumberOfCases>& extents) {
    static_assert(
        I < NumberOfCases,
        "Attempting to generate a case number not requested to be run");
    static_assert(
        I < total_num_cases,
        "Attempting to generate a case number whose index is out of bounds");

    const std::string name_prefix =
        BenchmarkHelpers::get_benchmark_name_prefix<DataVector, 1, Func>();
    const size_t total_num_grid_points =
        BenchmarkHelpers::get_total_num_grid_points(extents[I]);

    BENCHMARK(bench)
        ->Name(name_prefix)
        ->Args({static_cast<long int>(total_num_grid_points),
                static_cast<long int>(extents[I][0])});
  }
};
template <Function Func, size_t NumberOfCases>
struct run_benchmark_case<Func, NumberOfCases, 2> {
  template <size_t I>
  static void apply(
      const std::array<std::array<size_t, 2>, NumberOfCases>& extents) {
    static_assert(
        I < NumberOfCases,
        "Attempting to generate a case number not requested to be run");
    static_assert(
        I < total_num_cases,
        "Attempting to generate a case number whose index is out of bounds");

    const std::string name_prefix =
        BenchmarkHelpers::get_benchmark_name_prefix<DataVector, 2, Func>();
    const size_t total_num_grid_points =
        BenchmarkHelpers::get_total_num_grid_points(extents[I]);

    BENCHMARK(bench)
        ->Name(name_prefix)
        ->Args({static_cast<long int>(total_num_grid_points),
                static_cast<long int>(extents[I][0]),
                static_cast<long int>(extents[I][1])});
  }
};
template <Function Func, size_t NumberOfCases>
struct run_benchmark_case<Func, NumberOfCases, 3> {
  template <size_t I>
  static void apply(
      const std::array<std::array<size_t, 3>, NumberOfCases>& extents) {
    static_assert(
        I < NumberOfCases,
        "Attempting to generate a case number not requested to be run");
    static_assert(
        I < total_num_cases,
        "Attempting to generate a case number whose index is out of bounds");

    const std::string name_prefix =
        BenchmarkHelpers::get_benchmark_name_prefix<DataVector, 3, Func>();
    const size_t total_num_grid_points =
        BenchmarkHelpers::get_total_num_grid_points(extents[I]);

    BENCHMARK(bench)
        ->Name(name_prefix)
        ->Args({static_cast<long int>(total_num_grid_points),
                static_cast<long int>(extents[I][0]),
                static_cast<long int>(extents[I][1]),
                static_cast<long int>(extents[I][2])});
  }
};

template <Function Func, size_t NumberOfCases = num_cases_to_run,
          size_t Dimension = Dim>
struct run_benchmark_cases_helper {
  template <size_t... Is>
  static void apply(
      const std::array<std::array<size_t, Dimension>, NumberOfCases>& extents,
      const std::index_sequence<Is...>& /*meta*/) {
    (void)std::initializer_list<int>{
        (run_benchmark_case<Func, NumberOfCases, Dim>::template apply<Is>(
             extents),
         0)...};
  }
};

template <Function Func, size_t NumberOfCases = num_cases_to_run,
          size_t Dimension = Dim>
void run_benchmark_cases(
    const std::array<std::array<size_t, Dimension>, NumberOfCases>& extents) {
  run_benchmark_cases_helper<Func, NumberOfCases, Dimension>::apply(
      extents, std::make_index_sequence<num_cases_to_run>{});
}

void run_benchmark() { run_benchmark_cases<FunctionToRun>(extents_to_run); }
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
  run_benchmark();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
#pragma GCC diagnostic pop
