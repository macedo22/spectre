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
#include "Executables/Benchmark/GHTimeDerivative/dt_phi/BenchmarkedImpls.hpp"
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
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi_two_normals
  const phi_two_normals_type phi_two_normals =
      make_with_random_values<phi_two_normals_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: three_index_constraint
  const three_index_constraint_type three_index_constraint =
      make_with_random_values<three_index_constraint_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: gamma2
  const gamma2_type gamma2 = make_with_random_values<gamma2_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi_one_normal
  const phi_one_normal_type phi_one_normal =
      make_with_random_values<phi_one_normal_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: phi_1_up
  const phi_1_up_type phi_1_up = make_with_random_values<phi_1_up_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // LHS: dt_phi
  dt_phi_type dt_phi(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&dt_phi), pi, d_pi,
                                       phi_two_normals, three_index_constraint,
                                       gamma2, phi_one_normal, phi_1_up, lapse,
                                       shift, d_phi);
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
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_phi_type>, ::Tags::TempTensor<1, pi_type>,
      ::Tags::TempTensor<2, d_pi_type>,
      ::Tags::TempTensor<3, phi_two_normals_type>,
      ::Tags::TempTensor<4, three_index_constraint_type>,
      ::Tags::TempTensor<5, gamma2_type>,
      ::Tags::TempTensor<6, phi_one_normal_type>,
      ::Tags::TempTensor<7, phi_1_up_type>, ::Tags::TempTensor<8, lapse_type>,
      ::Tags::TempTensor<9, shift_type>, ::Tags::TempTensor<10, d_phi_type>>>
      vars{num_grid_points};

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<1, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<2, d_pi_type>>(vars);
  fill_with_random_values(make_not_null(&d_pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_two_normals
  phi_two_normals_type& phi_two_normals =
      get<::Tags::TempTensor<3, phi_two_normals_type>>(vars);
  fill_with_random_values(make_not_null(&phi_two_normals),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: three_index_constraint
  three_index_constraint_type& three_index_constraint =
      get<::Tags::TempTensor<4, three_index_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&three_index_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<5, gamma2_type>>(vars);
  fill_with_random_values(make_not_null(&gamma2), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_one_normal
  phi_one_normal_type& phi_one_normal =
      get<::Tags::TempTensor<6, phi_one_normal_type>>(vars);
  fill_with_random_values(make_not_null(&phi_one_normal),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<7, phi_1_up_type>>(vars);
  fill_with_random_values(make_not_null(&phi_1_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<8, lapse_type>>(vars);
  fill_with_random_values(make_not_null(&lapse), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift
  shift_type& shift = get<::Tags::TempTensor<9, shift_type>>(vars);
  fill_with_random_values(make_not_null(&shift), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<10, d_phi_type>>(vars);
  fill_with_random_values(make_not_null(&d_phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_phi
  dt_phi_type& dt_phi = get<::Tags::TempTensor<0, dt_phi_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&dt_phi), pi, d_pi,
                                       phi_two_normals, three_index_constraint,
                                       gamma2, phi_one_normal, phi_1_up, lapse,
                                       shift, d_phi);
    benchmark::DoNotOptimize(dt_phi);
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
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_pi
  const d_pi_type d_pi = make_with_random_values<d_pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi_two_normals
  const phi_two_normals_type phi_two_normals =
      make_with_random_values<phi_two_normals_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: three_index_constraint
  const three_index_constraint_type three_index_constraint =
      make_with_random_values<three_index_constraint_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: gamma2
  const gamma2_type gamma2 = make_with_random_values<gamma2_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: phi_one_normal
  const phi_one_normal_type phi_one_normal =
      make_with_random_values<phi_one_normal_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: phi_1_up
  const phi_1_up_type phi_1_up = make_with_random_values<phi_1_up_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: shift
  const shift_type shift = make_with_random_values<shift_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: d_phi
  const d_phi_type d_phi = make_with_random_values<d_phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // LHS: dt_phi
  dt_phi_type dt_phi(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_phi), pi, d_pi, phi_two_normals,
        three_index_constraint, gamma2, phi_one_normal, phi_1_up, lapse, shift,
        d_phi);
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
  using dt_phi_type = typename BenchmarkImpl::dt_phi_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using d_pi_type = typename BenchmarkImpl::d_pi_type;
  using phi_two_normals_type = typename BenchmarkImpl::phi_two_normals_type;
  using three_index_constraint_type =
      typename BenchmarkImpl::three_index_constraint_type;
  using gamma2_type = typename BenchmarkImpl::gamma2_type;
  using phi_one_normal_type = typename BenchmarkImpl::phi_one_normal_type;
  using phi_1_up_type = typename BenchmarkImpl::phi_1_up_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using d_phi_type = typename BenchmarkImpl::d_phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_phi_type>, ::Tags::TempTensor<1, pi_type>,
      ::Tags::TempTensor<2, d_pi_type>,
      ::Tags::TempTensor<3, phi_two_normals_type>,
      ::Tags::TempTensor<4, three_index_constraint_type>,
      ::Tags::TempTensor<5, gamma2_type>,
      ::Tags::TempTensor<6, phi_one_normal_type>,
      ::Tags::TempTensor<7, phi_1_up_type>, ::Tags::TempTensor<8, lapse_type>,
      ::Tags::TempTensor<9, shift_type>, ::Tags::TempTensor<10, d_phi_type>>>
      vars{num_grid_points};

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<1, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_pi
  d_pi_type& d_pi = get<::Tags::TempTensor<2, d_pi_type>>(vars);
  fill_with_random_values(make_not_null(&d_pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_two_normals
  phi_two_normals_type& phi_two_normals =
      get<::Tags::TempTensor<3, phi_two_normals_type>>(vars);
  fill_with_random_values(make_not_null(&phi_two_normals),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: three_index_constraint
  three_index_constraint_type& three_index_constraint =
      get<::Tags::TempTensor<4, three_index_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&three_index_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma2
  gamma2_type& gamma2 = get<::Tags::TempTensor<5, gamma2_type>>(vars);
  fill_with_random_values(make_not_null(&gamma2), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_one_normal
  phi_one_normal_type& phi_one_normal =
      get<::Tags::TempTensor<6, phi_one_normal_type>>(vars);
  fill_with_random_values(make_not_null(&phi_one_normal),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi_1_up
  phi_1_up_type& phi_1_up = get<::Tags::TempTensor<7, phi_1_up_type>>(vars);
  fill_with_random_values(make_not_null(&phi_1_up), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<8, lapse_type>>(vars);
  fill_with_random_values(make_not_null(&lapse), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift
  shift_type& shift = get<::Tags::TempTensor<9, shift_type>>(vars);
  fill_with_random_values(make_not_null(&shift), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: d_phi
  d_phi_type& d_phi = get<::Tags::TempTensor<10, d_phi_type>>(vars);
  fill_with_random_values(make_not_null(&d_phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_phi
  dt_phi_type& dt_phi = get<::Tags::TempTensor<0, dt_phi_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_phi), pi, d_pi, phi_two_normals,
        three_index_constraint, gamma2, phi_one_normal, phi_1_up, lapse, shift,
        d_phi);
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
