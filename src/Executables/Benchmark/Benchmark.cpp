// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <array>
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <charm++.h>
#include <cmath>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// This file is an example of how to do microbenchmark with Google Benchmark
// https://github.com/google/benchmark
// For two examples in different anonymous namespaces

namespace {
// Benchmark of push_back() in std::vector, following Chandler Carruth's talk
// at CppCon in 2015,
// https://www.youtube.com/watch?v=nXaxk27zwlk

// void bench_create(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    benchmark::DoNotOptimize(&v);
//    static_cast<void>(v);
//  }
// }
// BENCHMARK(bench_create);

// void bench_reserve(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//  }
// }
// BENCHMARK(bench_reserve);

// void bench_push_back(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//    v.push_back(42);
//    benchmark::ClobberMemory();
//  }
// }
// BENCHMARK(bench_push_back);
}  // namespace

namespace {
// In this anonymous namespace is an example of microbenchmarking the
// all_gradient routine for the GH system

template <size_t Dim>
struct Kappa : db::SimpleTag {
  using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
};
template <size_t Dim>
struct Psi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
};

template <size_t Dim>
using gh_evolution_vars_tags =
    typename gh::System<Dim>::variables_tag::tags_list;

// clang-tidy: don't pass be non-const reference
void bench_partial_derivatives(benchmark::State& state) {  // NOLINT
  const size_t num_1d_grid_points = static_cast<size_t>(state.range(0));
  constexpr size_t Dim = 3;
  // const size_t num_grid_points = pow(num_1d_grid_points, Dim);
  const Mesh<Dim> mesh{num_1d_grid_points, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
  using Map3d =
      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>;
  domain::CoordinateMap<Frame::ElementLogical, Frame::Grid, Map3d> map(
      Map3d{map1d, map1d, map1d});

  using VarTags = gh_evolution_vars_tags<Dim>;
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>
      inv_jac = map.inv_jacobian(logical_coordinates(mesh));
  const auto grid_coords = map(logical_coordinates(mesh));
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(partial_derivatives<VarTags>(vars, mesh, inv_jac));
    benchmark::ClobberMemory();
  }
}

// clang-tidy: don't pass be non-const reference
void bench_logical_partial_derivatives(benchmark::State& state) {  // NOLINT
  const size_t num_1d_grid_points = static_cast<size_t>(state.range(0));
  constexpr size_t Dim = 3;
  // const size_t num_grid_points = pow(num_1d_grid_points, Dim);
  const Mesh<Dim> mesh{num_1d_grid_points, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

  using VarTags = gh_evolution_vars_tags<Dim>;
  Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(logical_partial_derivatives<VarTags>(vars, mesh));
    benchmark::ClobberMemory();
  }
}

// Each DataVector case is run with each number of grid points
constexpr std::array<long int, 7> num_1d_grid_point_values = {2,  5,  8, 9,
                                                              10, 15, 20};

void run_benchmarks() {
  const std::string partial_derivatives_benchmark_name =
      "partial_derivatives/3D";
  BENCHMARK(bench_partial_derivatives)
      ->Name(partial_derivatives_benchmark_name)
      ->Arg(num_1d_grid_point_values[0])
      ->Arg(num_1d_grid_point_values[1])
      ->Arg(num_1d_grid_point_values[2])
      ->Arg(num_1d_grid_point_values[3])
      ->Arg(num_1d_grid_point_values[4])
      ->Arg(num_1d_grid_point_values[5])
      ->Arg(num_1d_grid_point_values[6]);

  const std::string logical_partial_derivatives_benchmark_name =
      "logical_partial_derivatives/3D";
  BENCHMARK(bench_logical_partial_derivatives)
      ->Name(logical_partial_derivatives_benchmark_name)
      ->Arg(num_1d_grid_point_values[0])
      ->Arg(num_1d_grid_point_values[1])
      ->Arg(num_1d_grid_point_values[2])
      ->Arg(num_1d_grid_point_values[3])
      ->Arg(num_1d_grid_point_values[4])
      ->Arg(num_1d_grid_point_values[5])
      ->Arg(num_1d_grid_point_values[6]);
}
// BENCHMARK(bench_all_gradient);  // NOLINT
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
  run_benchmarks();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
#pragma GCC diagnostic pop
