// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
constexpr size_t seed = 17;
std::mt19937 generator(seed);
constexpr size_t num_grid_points = 1000;

void bench_prod_sum_8_plus_equals(benchmark::State& state) {  // NOLINT
  const DataVector used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: R
  const DataVector R = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  // RHS: S
  const DataVector S = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  for (auto _ : state) {
    // LHS: result
    DataVector result = R * S;
    result += R * S;
    result += R * S;
    result += R * S;
    result += R * S;
    result += R * S;
    result += R * S;
    result += R * S;
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
}

BENCHMARK(bench_prod_sum_8_plus_equals);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
