// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <cstddef>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkedImpls.hpp"

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
// // In this anonymous namespace is an example of microbenchmarking the
// // all_gradient routine for the GH system

// template <size_t Dim>
// struct Kappa : db::SimpleTag {
//   using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
// };
// template <size_t Dim>
// struct Psi : db::SimpleTag {
//   using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
// };

// // clang-tidy: don't pass be non-const reference
// void bench_all_gradient(benchmark::State& state) {  // NOLINT
//   constexpr const size_t pts_1d = 4;
//   constexpr const size_t Dim = 3;
//   const Mesh<Dim> mesh{pts_1d, Spectral::Basis::Legendre,
//                        Spectral::Quadrature::GaussLobatto};
//   domain::CoordinateMaps::Affine map1d(-1.0, 1.0, -1.0, 1.0);
//   using Map3d =
//       domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
//                                              domain::CoordinateMaps::Affine,
//                                              domain::CoordinateMaps::Affine>;
//   domain::CoordinateMap<Frame::Logical, Frame::Grid, Map3d> map(
//       Map3d{map1d, map1d, map1d});

//   using VarTags = tmpl::list<Kappa<Dim>, Psi<Dim>>;
//   const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Grid> inv_jac
//   =
//       map.inv_jacobian(logical_coordinates(mesh));
//   const auto grid_coords = map(logical_coordinates(mesh));
//   Variables<VarTags> vars(mesh.number_of_grid_points(), 0.0);

//   for(auto _ : state) {
//     benchmark::DoNotOptimize(partial_derivatives<VarTags>(vars, mesh,
//     inv_jac));
//   }
// }
// BENCHMARK(bench_all_gradient);  // NOLINT
}  // namespace

namespace {
// set up shared stuff
using DataType = DataVector;
constexpr size_t Dim = 3;
using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
using christoffel_second_kind_type =
    BenchmarkImpl::christoffel_second_kind_type;
using christoffel_first_kind_type = BenchmarkImpl::christoffel_first_kind_type;
using inverse_spacetime_metric_type =
    BenchmarkImpl::inverse_spacetime_metric_type;

// profile manual implementation, equation terms not in buffer
void bench_manual_tensor_equation_without_buffer(
    benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());

  // christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&christoffel_first_kind));

  // inverse_spacetime_metric
  inverse_spacetime_metric_type inverse_spacetime_metric(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&inverse_spacetime_metric));

  // LHS: christoffel_second_kind
  christoffel_second_kind_type christoffel_second_kind(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl(make_not_null(&christoffel_second_kind),
                               christoffel_first_kind,
                               inverse_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_second_kind);
    benchmark::ClobberMemory();
  }
}

// profile manual implementation, equation terms in buffer
void bench_manual_tensor_equation_with_buffer(
    benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));

  TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_second_kind_type>,
                        ::Tags::TempTensor<1, christoffel_first_kind_type>,
                        ::Tags::TempTensor<2, inverse_spacetime_metric_type>>>
      vars{num_grid_points};

  // christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<1, christoffel_first_kind_type>>(vars);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&christoffel_first_kind));

  // inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<2, inverse_spacetime_metric_type>>(vars);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&inverse_spacetime_metric));

  // LHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<0, christoffel_second_kind_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl(make_not_null(&christoffel_second_kind),
                               christoffel_first_kind,
                               inverse_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_second_kind);
    benchmark::ClobberMemory();
  }
}

// profile TE implementation, returns LHS tensor
void bench_tensorexpression_return_lhs_tensor(
    benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());

  // christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&christoffel_first_kind));

  // inverse_spacetime_metric
  inverse_spacetime_metric_type inverse_spacetime_metric(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&inverse_spacetime_metric));

  for(auto _ : state) {
    // LHS: christoffel_second_kind
    const christoffel_second_kind_type christoffel_second_kind =
        BenchmarkImpl::tensorexpression_impl_return(christoffel_first_kind,
                                                    inverse_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_second_kind);
    benchmark::ClobberMemory();
  }
}

// profile TE implementation, takes LHS as arg, equation terms not in buffer
void bench_tensorexpression_lhs_tensor_as_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      DataVector(num_grid_points, std::numeric_limits<double>::signaling_NaN());

  // christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&christoffel_first_kind));

  // inverse_spacetime_metric
  inverse_spacetime_metric_type inverse_spacetime_metric(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&inverse_spacetime_metric));

  // LHS: christoffel_second_kind
  christoffel_second_kind_type christoffel_second_kind(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&christoffel_second_kind), christoffel_first_kind,
        inverse_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_second_kind);
    benchmark::ClobberMemory();
  }
}

// profile TE implementation, takes LHS as arg, equation terms in buffer
void bench_tensorexpression_lhs_tensor_as_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
  const size_t num_grid_points = static_cast<size_t>(state.range(0));

  TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_second_kind_type>,
                        ::Tags::TempTensor<1, christoffel_first_kind_type>,
                        ::Tags::TempTensor<2, inverse_spacetime_metric_type>>>
      vars{num_grid_points};

  // christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<1, christoffel_first_kind_type>>(vars);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&christoffel_first_kind));

  // inverse_spacetime_metric
  inverse_spacetime_metric_type& inverse_spacetime_metric =
      get<::Tags::TempTensor<2, inverse_spacetime_metric_type>>(vars);
  BenchmarkHelpers::zero_initialize_tensor(
      make_not_null(&inverse_spacetime_metric));

  // LHS: christoffel_second_kind
  christoffel_second_kind_type& christoffel_second_kind =
      get<::Tags::TempTensor<0, christoffel_second_kind_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(
        make_not_null(&christoffel_second_kind), christoffel_first_kind,
        inverse_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_second_kind);
    benchmark::ClobberMemory();
  }
}

// Profile with each of these number of grid points for DataVector
const std::array<long int, 4> num_grid_point_values = {5, 100, 500, 1000};

// Profile manual implementation with and without using a buffer for tensor
// equation terms, TE implementation that returns LHS tensor, and TE
// implementation that takes LHS tensor as an argument with and without using a
// buffer for tensor equation terms
BENCHMARK(bench_manual_tensor_equation_without_buffer)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);  // NOLINT
BENCHMARK(bench_manual_tensor_equation_with_buffer)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);  // NOLINT
BENCHMARK(bench_tensorexpression_return_lhs_tensor)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);  // NOLINT
BENCHMARK(bench_tensorexpression_lhs_tensor_as_arg_without_buffer)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);  // NOLINT
BENCHMARK(bench_tensorexpression_lhs_tensor_as_arg_with_buffer)
    ->Arg(num_grid_point_values[0])
    ->Arg(num_grid_point_values[1])
    ->Arg(num_grid_point_values[2])
    ->Arg(num_grid_point_values[3]);  // NOLINT
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
