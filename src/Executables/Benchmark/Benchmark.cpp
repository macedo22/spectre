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

// template <size_t Dim>
// struct Kappa : db::SimpleTag {
//   using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
// };
// template <size_t Dim>
// struct Psi : db::SimpleTag {
//   using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
// };

// clang-tidy: don't pass be non-const reference
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

//   while (state.KeepRunning()) {
//     benchmark::DoNotOptimize(partial_derivatives<VarTags>(vars, mesh,
//     inv_jac));
//   }
// }
// BENCHMARK(bench_all_gradient);  // NOLINT
}  // namespace

namespace {
// set up shared stuff
constexpr size_t seed = 17;
std::mt19937 generator(seed);

// benchmark manual implementation, takes LHS as arg, equation terms not in
// buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_arg_without_buffer(
    benchmark::State& state) {
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: da_spacetime_metric
  const da_spacetime_metric_type da_spacetime_metric =
      make_with_random_values<da_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&christoffel_first_kind),
                                       da_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_first_kind);
    benchmark::ClobberMemory();
  }
}

// benchmark manual implementation, takes LHS as arg, equation terms in buffer
template <typename DataType, size_t Dim>
void bench_manual_tensor_equation_lhs_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_first_kind_type>,
                        ::Tags::TempTensor<1, da_spacetime_metric_type>>>
      vars{num_grid_points};

  // RHS: da_spacetime_metric
  da_spacetime_metric_type& da_spacetime_metric =
      get<::Tags::TempTensor<1, da_spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&da_spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<0, christoffel_first_kind_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::manual_impl_lhs_arg(make_not_null(&christoffel_first_kind),
                                       da_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_first_kind);
    benchmark::ClobberMemory();
  }
}

// benchmark TE implementation, takes LHS as arg, equation terms not in buffer
template <typename DataType, size_t Dim>
void bench_tensorexpression_lhs_arg_without_buffer(
    benchmark::State& state) {  // NOLINT
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  const DataType used_for_size =
      BenchmarkHelpers::get_used_for_size<DataType>(num_grid_points);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // RHS: da_spacetime_metric
  const da_spacetime_metric_type da_spacetime_metric =
      make_with_random_values<da_spacetime_metric_type>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  // LHS: christoffel_first_kind
  christoffel_first_kind_type christoffel_first_kind(used_for_size);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_arg(
        make_not_null(&christoffel_first_kind), da_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_first_kind);
    benchmark::ClobberMemory();
  }
}

// benchmark TE implementation, takes LHS as arg, equation terms in buffer
template <typename DataType, size_t Dim, size_t NumGridPoints>
void bench_tensorexpression_lhs_arg_with_buffer(
    benchmark::State& state) {  // NOLINT
  using BenchmarkImpl = BenchmarkImpl<DataType, Dim>;
  using christoffel_first_kind_type =
      typename BenchmarkImpl::christoffel_first_kind_type;
  using da_spacetime_metric_type =
      typename BenchmarkImpl::da_spacetime_metric_type;

  const size_t num_grid_points = static_cast<size_t>(state.range(0));
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  TempBuffer<tmpl::list<::Tags::TempTensor<0, christoffel_first_kind_type>,
                        ::Tags::TempTensor<1, da_spacetime_metric_type>>>
      vars{num_grid_points};

  // RHS: da_spacetime_metric
  da_spacetime_metric_type& da_spacetime_metric =
      get<::Tags::TempTensor<1, da_spacetime_metric_type>>(vars);
  fill_with_random_values(make_not_null(&da_spacetime_metric),
                          make_not_null(&generator),
                          make_not_null(&distribution));

  // LHS: christoffel_first_kind
  christoffel_first_kind_type& christoffel_first_kind =
      get<::Tags::TempTensor<0, christoffel_first_kind_type>>(vars);

  for (auto _ : state) {
    BenchmarkImpl::tensorexpression_impl_lhs_arg(
        make_not_null(&christoffel_first_kind), da_spacetime_metric);
    benchmark::DoNotOptimize(christoffel_first_kind);
    benchmark::ClobberMemory();
  }
}

// Benchmark with each of these number of grid points for DataVector for a
// single dimension
constexpr std::array<long int, 4> num_grid_point_values = {8, 125, 512, 1000};

// template <typename DataType, size_t Dim, size_t NumGridPoints>
// void setup_benchmarks_3() {
//   static_assert(std::is_same_v<DataType, DataVector> ^ (NumGridPoints == 0),
//                 "If the DataType is DataVector, NumGridPoints must be "
//                 "greater than 0. Otherwise, if the DataType is double, "
//                 "NumGridPoints must be 0.");
//   const std::string benchmark_name_stub =
//       BenchmarkHelpers::get_benchmark_name_stub<DataType>(Dim,
//       NumGridPoints);
//   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
//                      DataType, Dim, NumGridPoints)
//       ->Name("manual/lhs_arg/without_buffer/" + benchmark_name_stub);
//   BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_without_buffer, DataType,
//                      Dim, NumGridPoints)
//       ->Name("TensorExpression/lhs_arg/without_buffer/" +
//       benchmark_name_stub);
//   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_with_buffer,
//   DataType,
//                      Dim, NumGridPoints)
//       ->Name("manual/lhs_arg/with_buffer/" + benchmark_name_stub);
//   BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_with_buffer, DataType,
//   Dim,
//                      NumGridPoints)
//       ->Name("TensorExpression/lhs_arg/with_buffer/" + benchmark_name_stub);
// }

// template <size_t Dim>
// void setup_benchmarks_2() {
//   setup_benchmarks_3<double, Dim, 0>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[0]>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[1]>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[2]>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[3]>();
// }

// template <typename DataType, size_t Dim, size_t NumGridPoints>
// void setup_benchmarks_3() {
//   static_assert(std::is_same_v<DataType, DataVector> ^ (NumGridPoints == 0),
//                 "If the DataType is DataVector, NumGridPoints must be "
//                 "greater than 0. Otherwise, if the DataType is double, "
//                 "NumGridPoints must be 0.");
//   const std::string benchmark_name_stub =
//       BenchmarkHelpers::get_benchmark_name_stub<DataType>(Dim,
//       NumGridPoints);
//   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
//                      DataType, Dim, NumGridPoints)
//       ->Name("manual/lhs_arg/without_buffer/" + benchmark_name_stub);
//   BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_without_buffer, DataType,
//                      Dim, NumGridPoints)
//       ->Name("TensorExpression/lhs_arg/without_buffer/" +
//       benchmark_name_stub);
//   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_with_buffer,
//   DataType,
//                      Dim, NumGridPoints)
//       ->Name("manual/lhs_arg/with_buffer/" + benchmark_name_stub);
//   BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_with_buffer, DataType,
//   Dim,
//                      NumGridPoints)
//       ->Name("TensorExpression/lhs_arg/with_buffer/" + benchmark_name_stub);
// }

// template <size_t Dim>
// void setup_manual_lhs_arg_without_buffer() {
//   const std::string benchmark_name_prefix = "manual/lhs_arg/without_buffer/";
//   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
//                      double, Dim)
//       ->Name(benchmark_name_prefix +
//       BenchmarkHelpers::get_benchmark_name_suffix<double>(Dim));
//   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
//                      DataVector, Dim)
//       ->Name(benchmark_name_prefix +
//       BenchmarkHelpers::get_benchmark_name_suffix<DataVector>(Dim))
//       ->Arg(num_grid_point_values[0])
//       ->Arg(num_grid_point_values[1])
//       ->Arg(num_grid_point_values[2])
//       ->Arg(num_grid_point_values[3]);
// //   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
// //                      DataVector, Dim, num_grid_points[1])
// //       ->Name(benchmark_name_prefix +
// BenchmarkHelpers::get_benchmark_name_stub<DataVector>(Dim,
// num_grid_points[1]));
// //   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
// //                      DataVector, Dim, num_grid_points[2])
// //       ->Name(benchmark_name_prefix +
// BenchmarkHelpers::get_benchmark_name_stub<DataVector>(Dim,
// num_grid_points[2]));
// //   BENCHMARK_TEMPLATE(bench_manual_tensor_equation_lhs_arg_without_buffer,
// //                      DataVector, Dim, num_grid_points[3])
// //       ->Name(benchmark_name_prefix +
// BenchmarkHelpers::get_benchmark_name_stub<DataVector>(Dim,
// num_grid_points[3]));
// }

// template <size_t Dim>
// void setup_benchmarks_impl() {
// //   setup_benchmarks_3<double, Dim, 0>();
// //   setup_benchmarks_3<DataVector, Dim, num_grid_points[0]>();
// //   setup_benchmarks_3<DataVector, Dim, num_grid_points[1]>();
// //   setup_benchmarks_3<DataVector, Dim, num_grid_points[2]>();
// //   setup_benchmarks_3<DataVector, Dim, num_grid_points[3]>();

//   setup_manual_lhs_arg_without_buffer<Dim>();
// //   setup_manual_lhs_arg_with_buffer<Dim>();
// //   setup_te_lhs_arg_without_buffer<Dim>();
// //   setup_te_lhs_arg_with_buffer<Dim>();
// }

// template <typename Datatype>
// void setup_benchmarks_impl() {
//   setup_benchmarks_3<double, Dim, 0>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[0]>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[1]>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[2]>();
//   setup_benchmarks_3<DataVector, Dim, num_grid_points[3]>();
// }

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
void setup_te1_lhs_arg_without_buffer() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "TE - 1/lhs_arg/without_buffer/", Dim);
  if constexpr (std::is_same_v<DataType, double>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_without_buffer, DataType,
                       Dim)
        ->Name(benchmark_name)
        ->Arg(0);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    BENCHMARK_TEMPLATE(bench_tensorexpression_lhs_arg_without_buffer, DataType,
                       Dim)
        ->Name(benchmark_name)
        ->Arg(num_grid_point_values[0])
        ->Arg(num_grid_point_values[1])
        ->Arg(num_grid_point_values[2])
        ->Arg(num_grid_point_values[3]);
  }
}

template <typename DataType, size_t Dim>
void setup_te1_lhs_arg_with_buffer() {
  const std::string benchmark_name =
      BenchmarkHelpers::get_benchmark_name<DataType>(
          "TE - 1/lhs_arg/with_buffer/", Dim);
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

void setup_benchmarks() {
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

  setup_te1_lhs_arg_without_buffer<double, 1>();
  setup_te1_lhs_arg_without_buffer<double, 2>();
  setup_te1_lhs_arg_without_buffer<double, 3>();
  setup_te1_lhs_arg_without_buffer<DataVector, 1>();
  setup_te1_lhs_arg_without_buffer<DataVector, 2>();
  setup_te1_lhs_arg_without_buffer<DataVector, 3>();

  setup_te1_lhs_arg_with_buffer<double, 1>();
  setup_te1_lhs_arg_with_buffer<double, 2>();
  setup_te1_lhs_arg_with_buffer<double, 3>();
  setup_te1_lhs_arg_with_buffer<DataVector, 1>();
  setup_te1_lhs_arg_with_buffer<DataVector, 2>();
  setup_te1_lhs_arg_with_buffer<DataVector, 3>();
}

// template <size_t Case, typename Datatype, size_t Dim>
// void setup_benchmarks_core() {
// //   setup_manual_lhs_arg_without_buffer<Datatype, Dim>();
// //   setup_manual_lhs_arg_with_buffer<Datatype, Dim>();
//   SetupBenchmark<Case>::apply<Datatype, Dim>();
// }

// template <size_t Case, typename Datatype>
// void setup_benchmarks_impl() {
//   setup_benchmarks_core<Case, Datatype, 1>();
//   setup_benchmarks_core<Case, Datatype, 2>();
//   setup_benchmarks_core<Case, Datatype, 3>();
// }

// void setup_benchmarks() {
//   setup_benchmarks_impl<1>();
//   setup_benchmarks_impl<2>();
//   setup_benchmarks_impl<3>();
// }

// template <size_t Case>
// void setup_benchmarks_2() {
//   setup_benchmarks_impl<Case, double>();
//   setup_benchmarks_impl<Case, DataVector>();
// }

// void setup_benchmarks() {
//   setup_benchmarks_2<0>();
//   setup_benchmarks_2<1>();
// }

// Benchmark manual implementations and TensorExpression implementations that:
// (i) take LHS tensor as an argument and do not use a buffer
// (ii) take LHS tensor as an argument and do use a buffer
}  // namespace

// Ignore the warning about an extra ';' because some versions of benchmark
// require it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
// BENCHMARK_MAIN();
int main(int argc, char** argv) {
  setup_benchmarks();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
#pragma GCC diagnostic pop
