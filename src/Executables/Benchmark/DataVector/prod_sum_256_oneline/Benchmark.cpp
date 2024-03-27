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

void bench_prod_sum_256_oneline(benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));
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
    DataVector result =
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S + R * S +
        R * S + R * S + R * S + R * S;
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
}

// Cases are run with each number of grid points
constexpr std::array<long int, 4> num_grid_point_values = {8, 125, 512, 1000};

BENCHMARK(bench_prod_sum_256_oneline)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
