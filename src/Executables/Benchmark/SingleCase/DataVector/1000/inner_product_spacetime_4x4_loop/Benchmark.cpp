// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Executables/Benchmark/DataVector/InnerProductImpls.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
constexpr size_t seed = 17;
std::mt19937 generator(seed);
constexpr size_t num_grid_points = 1000;

void bench_loop(benchmark::State& state) {  // NOLINT
  using result_type = typename BenchmarkImpl::result_type;
  using R_type = typename BenchmarkImpl::R_type;
  using S_type = typename BenchmarkImpl::S_type;

  const DataVector used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: R
  const R_type R = make_with_random_values<R_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // RHS: S
  const S_type S = make_with_random_values<S_type>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  // LHS: result
  result_type result(used_for_size);

  for (auto _ : state) {
    // LHS: result
    BenchmarkImpl::loop_impl(make_not_null(&result), R, S);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
}

BENCHMARK(bench_loop);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
