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
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: christoffel_first_kind
  const christoffel_first_kind_type christoffel_first_kind =
      make_with_random_values<christoffel_first_kind_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: inverse_spacetime_metric
  const inverse_spacetime_metric_type inverse_spacetime_metric =
      make_with_random_values<inverse_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  for (auto _ : state) {
    // LHS: trace_christoffel
    const trace_christoffel_type trace_christoffel =
        BenchmarkImpl::manual_impl_lhs_return(christoffel_first_kind,
                                              inverse_spacetime_metric);
    benchmark::DoNotOptimize(trace_christoffel);
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
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: christoffel_first_kind
  const christoffel_first_kind_type christoffel_first_kind =
      make_with_random_values<christoffel_first_kind_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: inverse_spacetime_metric
  const inverse_spacetime_metric_type inverse_spacetime_metric =
      make_with_random_values<inverse_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: trace_christoffel
  trace_christoffel_type trace_christoffel(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&trace_christoffel),
                                       christoffel_first_kind,
                                       inverse_spacetime_metric);
    benchmark::DoNotOptimize(trace_christoffel);
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
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<::Tags::TempTensor<0, trace_christoffel_type>,
                        ::Tags::TempTensor<1, christoffel_first_kind_type>,
                        ::Tags::TempTensor<2, inverse_spacetime_metric_type>>>
      vars{num_grid_points};

  // RHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<1, christoffel_first_kind_type>>(vars);
  fill_with_random_values(make_not_null(&christoffel_first_kind),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<2, inverse_spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&inverse_spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<0, trace_christoffel_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&trace_christoffel),
                                       christoffel_first_kind,
                                       inverse_spacetime_metric);
    benchmark::DoNotOptimize(trace_christoffel);
    benchmark::ClobberMemory();
  }
}

// - TensorExpression implementation
// - LHS tensor is constructed and returned by implementation
// - equation terms are not stored in buffer
template <typename DataType, size_t Dim>
void bench_tensorexpression_lhs_return(benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: christoffel_first_kind
  const christoffel_first_kind_type christoffel_first_kind =
      make_with_random_values<christoffel_first_kind_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: inverse_spacetime_metric
  const inverse_spacetime_metric_type inverse_spacetime_metric =
      make_with_random_values<inverse_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  for (auto _ : state) {
    // LHS: trace_christoffel
    const trace_christoffel_type trace_christoffel =
        BenchmarkImpl::tensorexpression_impl_lhs_return(
            christoffel_first_kind, inverse_spacetime_metric);
    benchmark::DoNotOptimize(trace_christoffel);
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
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: christoffel_first_kind
  const christoffel_first_kind_type christoffel_first_kind =
      make_with_random_values<christoffel_first_kind_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // RHS: inverse_spacetime_metric
  const inverse_spacetime_metric_type inverse_spacetime_metric =
      make_with_random_values<inverse_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: trace_christoffel
  trace_christoffel_type trace_christoffel(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&trace_christoffel), christoffel_first_kind,
        inverse_spacetime_metric);
    benchmark::DoNotOptimize(trace_christoffel);
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
  using trace_christoffel_type = typename BenchmarkImpl::trace_christoffel_type;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using inverse_spacetime_metric_type =
      typename BenchmarkImpl::inverse_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<::Tags::TempTensor<0, trace_christoffel_type>,
                        ::Tags::TempTensor<1, christoffel_first_kind_type>,
                        ::Tags::TempTensor<2, inverse_spacetime_metric_type>>>
      vars{num_grid_points};

  // RHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<1, christoffel_first_kind_type>>(vars);
  fill_with_random_values(make_not_null(&christoffel_first_kind),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // RHS: inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<2, inverse_spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&inverse_spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: trace_christoffel
  trace_christoffel_type& trace_christoffel =
      get<::Tags::TempTensor<0, trace_christoffel_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::template tensorexpression_impl_lhs_arg<CaseNumber>(
        make_not_null(&trace_christoffel), christoffel_first_kind,
        inverse_spacetime_metric);
    benchmark::DoNotOptimize(trace_christoffel);
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
          "TE : 1/lhs_return/without_buffer", Dim);
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
