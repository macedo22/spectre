// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark.h>
#pragma GCC diagnostic pop
#include <cstddef>

#include "DataStructures/DataVector.hpp"
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
using DataType = double;
constexpr size_t Dim = 3;
using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
using L_type = typename BenchmarkImpl::L_type;
using R_type = typename BenchmarkImpl::R_type;
using S_type = typename BenchmarkImpl::S_type;
using T_type = typename BenchmarkImpl::T_type;

// benchmark TE implementation, returns LHS tensor
void bench_manual_tensor_equation_return_lhs_tensor(
    benchmark::State& state) {  // NOLINT
  const DataType used_for_size = 0.0;

  // R
  R_type R(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&R));

  // S
  S_type S(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&S));

  // T
  T_type T(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&T));

  for (auto _ : state) {
    // LHS: L
    const L_type L = BenchmarkImpl::manual_impl_return(R, S, T, used_for_size);
    benchmark::DoNotOptimize(L);
    benchmark::ClobberMemory();
  }
}

// benchmark manual implementation, equation terms not in buffer
void bench_manual_tensor_equation_lhs_tensor_as_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
  const DataType used_for_size = 0.0;

  // R
  R_type R(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&R));

  // S
  S_type S(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&S));

  // T
  T_type T(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&T));

  // LHS: L
  L_type L(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_as_arg(make_not_null(&L), R, S, T);
    benchmark::DoNotOptimize(L);
    benchmark::ClobberMemory();
  }
}

// benchmark TE implementation, returns LHS tensor
void bench_tensorexpression_return_lhs_tensor(
    benchmark::State& state) {  // NOLINT
  const DataType used_for_size = 0.0;

  // R
  R_type R(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&R));

  // S
  S_type S(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&S));

  // T
  T_type T(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&T));

  for (auto _ : state) {
    // LHS: L
    const L_type L = BenchmarkImpl::tensorexpression_impl_return(R, S, T);
    benchmark::DoNotOptimize(L);
    benchmark::ClobberMemory();
  }
}

// benchmark TE implementation, takes LHS as arg, equation terms not in buffer
void bench_tensorexpression_lhs_tensor_as_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
  const DataType used_for_size = 0.0;

  // R
  R_type R(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&R));

  // S
  S_type S(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&S));

  // T
  T_type T(used_for_size);
  BenchmarkHelpers::zero_initialize_tensor(make_not_null(&T));

  // LHS: L
  L_type L(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_as_arg(make_not_null(&L), R, S, T);
    benchmark::DoNotOptimize(L);
    benchmark::ClobberMemory();
  }
}

// Benchmark manual implementations and TensorExpression implementations that:
// (i) return LHS tensor
// (ii) take LHS tensor as an argument and do not use a buffer
BENCHMARK(bench_manual_tensor_equation_return_lhs_tensor);  // NOLINT
BENCHMARK(
    bench_manual_tensor_equation_lhs_tensor_as_arg_without_buffer);  // NOLINT
BENCHMARK(bench_tensorexpression_return_lhs_tensor);                 // NOLINT
BENCHMARK(bench_tensorexpression_lhs_tensor_as_arg_without_buffer);  // NOLINT
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
BENCHMARK_MAIN();
#pragma GCC diagnostic pop
