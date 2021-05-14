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
#include "Executables/Benchmark/GHTimeDerivative/gauge_constraint/BenchmarkedImpls.hpp"
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
// - LHS tensor is constructed and returned by implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_return(benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: trace_christoffel
  const trace_christoffel_type trace_christoffel =
      make_with_random_values<trace_christoffel_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  for (auto _ : state) {
    // LHS: gauge_constraint
    const gauge_constraint_type gauge_constraint =
        BenchmarkImpl::manual_impl_lhs_return(gauge_function,
                                              trace_christoffel);
    benchmark::DoNotOptimize(gauge_constraint);
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
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: trace_christoffel
  const trace_christoffel_type trace_christoffel =
      make_with_random_values<trace_christoffel_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: gauge_constraint
  gauge_constraint_type gauge_constraint(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&gauge_constraint),
                                       gauge_function, trace_christoffel);
    benchmark::DoNotOptimize(gauge_constraint);
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
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<::Tags::TempTensor<0, gauge_constraint_type>,
                        ::Tags::TempTensor<1, gauge_function_type>,
                        ::Tags::TempTensor<2, trace_christoffel_type>>>
      vars{num_grid_points};

  // RHS: gauge_function
  gauge_function_type& gauge_function =
      get<::Tags::TempTensor<1, gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<2, trace_christoffel_type>>(vars);
  fill_with_random_values(make_not_null(&trace_christoffel),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<0, gauge_constraint_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&gauge_constraint),
                                       gauge_function, trace_christoffel);
    benchmark::DoNotOptimize(gauge_constraint);
    benchmark::ClobberMemory();
  }
}

// - TensorExpression implementation
// - LHS tensor is constructed and returned by implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_tensorexpression_lhs_return(benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: trace_christoffel
  const trace_christoffel_type trace_christoffel =
      make_with_random_values<trace_christoffel_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  for (auto _ : state) {
    // LHS: gauge_constraint
    const gauge_constraint_type gauge_constraint =
        BenchmarkImpl::tensorexpression_impl_lhs_return(gauge_function,
                                                        trace_christoffel);
    benchmark::DoNotOptimize(gauge_constraint);
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
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: gauge_function
  const gauge_function_type gauge_function =
      make_with_random_values<gauge_function_type>(make_not_null(&generator),
                                                   make_not_null(&distribution),
                                                   used_for_size);

  // RHS: trace_christoffel
  const trace_christoffel_type trace_christoffel =
      make_with_random_values<trace_christoffel_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: gauge_constraint
  gauge_constraint_type gauge_constraint(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&gauge_constraint), gauge_function, trace_christoffel);
    benchmark::DoNotOptimize(gauge_constraint);
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
  using gauge_constraint_type = typename BenchmarkImpl::gauge_constraint_type;
  using gauge_function_type = typename BenchmarkImpl::gauge_function_type;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<::Tags::TempTensor<0, gauge_constraint_type>,
                        ::Tags::TempTensor<1, gauge_function_type>,
                        ::Tags::TempTensor<2, trace_christoffel_type>>>
      vars{num_grid_points};

  // RHS: gauge_function
  gauge_function_type& gauge_function =
      get<::Tags::TempTensor<1, gauge_function_type>>(vars);
  fill_with_random_values(make_not_null(&gauge_function),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<2, trace_christoffel_type>>(vars);
  fill_with_random_values(make_not_null(&trace_christoffel),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: gauge_constraint
  gauge_constraint_type& gauge_constraint =
      get<::Tags::TempTensor<0, gauge_constraint_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&gauge_constraint), gauge_function, trace_christoffel);
    benchmark::DoNotOptimize(gauge_constraint);
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
