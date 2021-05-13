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
#include "Executables/Benchmark/BenchmarkedImpls.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
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
// - LHS tensor is constructed and returned by implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_return(benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1_plus_1
  const gamma1_plus_1_type gamma1_plus_1 =
      make_with_random_values<gamma1_plus_1_type>(make_not_null(&generator),
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

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  for (auto _ : state) {
    // LHS: dt_spacetime_metric
    const dt_spacetime_metric_type dt_spacetime_metric =
        BenchmarkImpl::manual_impl_lhs_return(lapse, pi, gamma1_plus_1,
                                              shift_dot_three_index_constraint,
                                              shift, phi);
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::ClobberMemory();
  }
}

// - manual implementation
// - LHS tensor is a function argument to implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_arg_without_buffer(
    benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1_plus_1
  const gamma1_plus_1_type gamma1_plus_1 =
      make_with_random_values<gamma1_plus_1_type>(make_not_null(&generator),
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

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type dt_spacetime_metric(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_spacetime_metric), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);
    benchmark::DoNotOptimize(dt_spacetime_metric);
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
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_spacetime_metric_type>,
      ::Tags::TempTensor<1, lapse_type>, ::Tags::TempTensor<2, pi_type>,
      ::Tags::TempTensor<3, gamma1_plus_1_type>,
      ::Tags::TempTensor<4, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<5, shift_type>, ::Tags::TempTensor<6, phi_type>>>
      vars{num_grid_points};

  // RHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<1, lapse_type>>(vars);
  fill_with_random_values(make_not_null(&lapse), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<2, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1 =
      get<::Tags::TempTensor<3, gamma1_plus_1_type>>(vars);
  fill_with_random_values(make_not_null(&gamma1_plus_1),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<4, shift_dot_three_index_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&shift_dot_three_index_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift
  shift_type& shift = get<::Tags::TempTensor<5, shift_type>>(vars);
  fill_with_random_values(make_not_null(&shift), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<6, phi_type>>(vars);
  fill_with_random_values(make_not_null(&phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type& dt_spacetime_metric =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(
        make_not_null(&dt_spacetime_metric), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::ClobberMemory();
  }
}

// - TensorExpression implementation
// - LHS tensor is constructed and returned by implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_tensorexpression_lhs_return(benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using dt_spacetime_metric_type =
      typename BenchmarkImpl::dt_spacetime_metric_type;
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1_plus_1
  const gamma1_plus_1_type gamma1_plus_1 =
      make_with_random_values<gamma1_plus_1_type>(make_not_null(&generator),
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

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  for (auto _ : state) {
    // LHS: dt_spacetime_metric
    const dt_spacetime_metric_type dt_spacetime_metric =
        BenchmarkImpl::tensorexpression_impl_lhs_return(
            lapse, pi, gamma1_plus_1, shift_dot_three_index_constraint, shift,
            phi);
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
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: lapse
  const lapse_type lapse = make_with_random_values<lapse_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: pi
  const pi_type pi = make_with_random_values<pi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: gamma1_plus_1
  const gamma1_plus_1_type gamma1_plus_1 =
      make_with_random_values<gamma1_plus_1_type>(make_not_null(&generator),
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

  // RHS: phi
  const phi_type phi = make_with_random_values<phi_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type dt_spacetime_metric(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_spacetime_metric), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);
    benchmark::DoNotOptimize(dt_spacetime_metric);
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
  using lapse_type = typename BenchmarkImpl::lapse_type;
  using pi_type = typename BenchmarkImpl::pi_type;
  using gamma1_plus_1_type = typename BenchmarkImpl::gamma1_plus_1_type;
  using shift_dot_three_index_constraint_type =
      typename BenchmarkImpl::shift_dot_three_index_constraint_type;
  using shift_type = typename BenchmarkImpl::shift_type;
  using phi_type = typename BenchmarkImpl::phi_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<
      ::Tags::TempTensor<0, dt_spacetime_metric_type>,
      ::Tags::TempTensor<1, lapse_type>, ::Tags::TempTensor<2, pi_type>,
      ::Tags::TempTensor<3, gamma1_plus_1_type>,
      ::Tags::TempTensor<4, shift_dot_three_index_constraint_type>,
      ::Tags::TempTensor<5, shift_type>, ::Tags::TempTensor<6, phi_type>>>
      vars{num_grid_points};

  // RHS: lapse
  lapse_type& lapse = get<::Tags::TempTensor<1, lapse_type>>(vars);
  fill_with_random_values(make_not_null(&lapse), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: pi
  pi_type& pi = get<::Tags::TempTensor<2, pi_type>>(vars);
  fill_with_random_values(make_not_null(&pi), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: gamma1_plus_1
  gamma1_plus_1_type& gamma1_plus_1 =
      get<::Tags::TempTensor<3, gamma1_plus_1_type>>(vars);
  fill_with_random_values(make_not_null(&gamma1_plus_1),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift_dot_three_index_constraint
  shift_dot_three_index_constraint_type& shift_dot_three_index_constraint =
      get<::Tags::TempTensor<4, shift_dot_three_index_constraint_type>>(vars);
  fill_with_random_values(make_not_null(&shift_dot_three_index_constraint),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: shift
  shift_type& shift = get<::Tags::TempTensor<5, shift_type>>(vars);
  fill_with_random_values(make_not_null(&shift), make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: phi
  phi_type& phi = get<::Tags::TempTensor<6, phi_type>>(vars);
  fill_with_random_values(make_not_null(&phi), make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: dt_spacetime_metric
  dt_spacetime_metric_type& dt_spacetime_metric =
      get<::Tags::TempTensor<0, dt_spacetime_metric_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&dt_spacetime_metric), lapse, pi, gamma1_plus_1,
        shift_dot_three_index_constraint, shift, phi);
    benchmark::DoNotOptimize(dt_spacetime_metric);
    benchmark::ClobberMemory();
  }
}

// ========================================================

// Each DataVector case is run with each number of grid points
constexpr std::array<long int, 4> num_grid_point_values = {8, 125, 512, 1000};

// ======= BENCHMARK_TEMPLATE INSTANTIATION HELPERS =======

template <typename DataType, size_t Dim>
void setup_manual_lhs_return() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "manual/lhs_return/without_buffer/", Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_return, DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_return, DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(num_grid_point_values[0])
        ->Arg(num_grid_point_values[1])
        ->Arg(num_grid_point_values[2])
        ->Arg(num_grid_point_values[3]);
  }
}

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

template <typename DataType, size_t Dim>
void setup_te_lhs_return() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "TE : 1/lhs_return/without_buffer/", Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_return, DataType, Dim)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_return, DataType, Dim)
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
  setup_manual_lhs_return<double, 1>();
  setup_manual_lhs_return<double, 2>();
  setup_manual_lhs_return<double, 3>();
  setup_manual_lhs_return<DataVector, 1>();
  setup_manual_lhs_return<DataVector, 2>();
  setup_manual_lhs_return<DataVector, 3>();

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

  setup_te_lhs_return<double, 1>();
  setup_te_lhs_return<double, 2>();
  setup_te_lhs_return<double, 3>();
  setup_te_lhs_return<DataVector, 1>();
  setup_te_lhs_return<DataVector, 2>();
  setup_te_lhs_return<DataVector, 3>();

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

  setup_te_lhs_arg_without_buffer<double, 1, 2>();
  setup_te_lhs_arg_without_buffer<double, 2, 2>();
  setup_te_lhs_arg_without_buffer<double, 3, 2>();
  setup_te_lhs_arg_without_buffer<DataVector, 1, 2>();
  setup_te_lhs_arg_without_buffer<DataVector, 2, 2>();
  setup_te_lhs_arg_without_buffer<DataVector, 3, 2>();

  setup_te_lhs_arg_with_buffer<double, 1, 2>();
  setup_te_lhs_arg_with_buffer<double, 2, 2>();
  setup_te_lhs_arg_with_buffer<double, 3, 2>();
  setup_te_lhs_arg_with_buffer<DataVector, 1, 2>();
  setup_te_lhs_arg_with_buffer<DataVector, 2, 2>();
  setup_te_lhs_arg_with_buffer<DataVector, 3, 2>();

  setup_te_lhs_arg_without_buffer<double, 1, 3>();
  setup_te_lhs_arg_without_buffer<double, 2, 3>();
  setup_te_lhs_arg_without_buffer<double, 3, 3>();
  setup_te_lhs_arg_without_buffer<DataVector, 1, 3>();
  setup_te_lhs_arg_without_buffer<DataVector, 2, 3>();
  setup_te_lhs_arg_without_buffer<DataVector, 3, 3>();

  setup_te_lhs_arg_with_buffer<double, 1, 3>();
  setup_te_lhs_arg_with_buffer<double, 2, 3>();
  setup_te_lhs_arg_with_buffer<double, 3, 3>();
  setup_te_lhs_arg_with_buffer<DataVector, 1, 3>();
  setup_te_lhs_arg_with_buffer<DataVector, 2, 3>();
  setup_te_lhs_arg_with_buffer<DataVector, 3, 3>();

  setup_te_lhs_arg_without_buffer<double, 1, 4>();
  setup_te_lhs_arg_without_buffer<double, 2, 4>();
  setup_te_lhs_arg_without_buffer<double, 3, 4>();
  setup_te_lhs_arg_without_buffer<DataVector, 1, 4>();
  setup_te_lhs_arg_without_buffer<DataVector, 2, 4>();
  setup_te_lhs_arg_without_buffer<DataVector, 3, 4>();

  setup_te_lhs_arg_with_buffer<double, 1, 4>();
  setup_te_lhs_arg_with_buffer<double, 2, 4>();
  setup_te_lhs_arg_with_buffer<double, 3, 4>();
  setup_te_lhs_arg_with_buffer<DataVector, 1, 4>();
  setup_te_lhs_arg_with_buffer<DataVector, 2, 4>();
  setup_te_lhs_arg_with_buffer<DataVector, 3, 4>();

  setup_te_lhs_arg_without_buffer<double, 1, 5>();
  setup_te_lhs_arg_without_buffer<double, 2, 5>();
  setup_te_lhs_arg_without_buffer<double, 3, 5>();
  setup_te_lhs_arg_without_buffer<DataVector, 1, 5>();
  setup_te_lhs_arg_without_buffer<DataVector, 2, 5>();
  setup_te_lhs_arg_without_buffer<DataVector, 3, 5>();

  setup_te_lhs_arg_with_buffer<double, 1, 5>();
  setup_te_lhs_arg_with_buffer<double, 2, 5>();
  setup_te_lhs_arg_with_buffer<double, 3, 5>();
  setup_te_lhs_arg_with_buffer<DataVector, 1, 5>();
  setup_te_lhs_arg_with_buffer<DataVector, 2, 5>();
  setup_te_lhs_arg_with_buffer<DataVector, 3, 5>();
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
