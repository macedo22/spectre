// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Executables/Benchmark/SingleCase/Loop/General/InnerProductImpls.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
constexpr size_t seed = 17;
std::mt19937 generator(seed);

// tensor types in tensor equation being benchmarked
using DataType = DataVector;
static constexpr size_t Dim = 3;
constexpr size_t num_grid_points = 512;

using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
using R_type = BenchmarkImpl::R_type_3x3;
using S_type = BenchmarkImpl::S_type_3x3;
using result_type = BenchmarkImpl::result_type;

void bench(benchmark::State& state) {  // NOLINT
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
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
    BenchmarkImpl::apply(make_not_null(&result), R, S);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
}

BENCHMARK(bench);
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
