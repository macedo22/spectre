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
static constexpr size_t Dim = 1;
constexpr size_t num_grid_points = 512;

using R_type =
    Tensor<DataType, Symmetry<1>,
           index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
using S_type =
    Tensor<DataType, Symmetry<1>,
           index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;
using result_type = Scalar<DataType>;

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
    TensorExpressions::evaluate(make_not_null(&result), R(ti_A) * S(ti_a));
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
