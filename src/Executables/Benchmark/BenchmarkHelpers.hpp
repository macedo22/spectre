// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace BenchmarkHelpers {
#define assertm(exp, msg) assert(((void)msg, exp))

template <typename DataType>
inline DataType get_used_for_size(const size_t num_grid_points);

template <>
inline double get_used_for_size<double>(const size_t /*num_grid_points*/) {
  return std::numeric_limits<double>::signaling_NaN();
}

template <>
inline DataVector get_used_for_size<DataVector>(const size_t num_grid_points) {
  return DataVector(num_grid_points,
                    std::numeric_limits<double>::signaling_NaN());
}

template <typename DataType>
std::string get_benchmark_name_suffix(const size_t dim) {
  const std::string datatype =
      std::is_same_v<DataType, DataVector> ? "DataVector" : "double";
  return datatype + "/" + std::to_string(dim) + "D/num_grid_points:";
}

template <typename DataType>
std::string get_benchmark_name(const std::string prefix, const size_t dim) {
  return prefix + get_benchmark_name_suffix<DataType>(dim);
}

template <typename... Ts>
void zero_initialize_tensor(gsl::not_null<Tensor<double, Ts...>*> tensor) {
  for (auto tensor_it = tensor->begin(); tensor_it != tensor->end();
       tensor_it++) {
    *tensor_it = 0.0;
  }
}

template <typename... Ts>
void zero_initialize_tensor(gsl::not_null<Tensor<DataVector, Ts...>*> tensor) {
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = 0.0;
    }
  }
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

template <typename... Ts>
void copy_tensor(const Tensor<Ts...>& tensor_source,
                 gsl::not_null<Tensor<Ts...>*> tensor_destination) {
  auto tensor_source_it = tensor_source.begin();
  auto tensor_destination_it = tensor_destination->begin();
  for (; tensor_source_it != tensor_source.end();
       tensor_source_it++, tensor_destination_it++) {
    *tensor_destination_it = *tensor_source_it;
  }
}

// Get a list of extents of dimension Dim with 1D grid points from 2 to 20, i.e.
// {{{{2, 2, 2}},  {{3, 3, 3}}, ..., {{20, 20, 20}}}}
template <size_t Dim, size_t num_cases>
constexpr std::array<std::array<size_t, Dim>, num_cases>
get_extents_consecutive() {
  const size_t lowest_num_grid_points = 2;

  std::array<std::array<size_t, Dim>, num_cases> extents{};

  for (size_t i = 0; i < num_cases; i++) {
    const size_t num_1d_grid_points = i + lowest_num_grid_points;
    for (size_t j = 0; j < Dim; j++) {
      gsl::at(gsl::at(extents, i), j) = num_1d_grid_points;
    }
  }

  return extents;
}

// Given a list extents with grid points per Dim, compute total grid points for
// each set of extents in the list
template <size_t Dim, size_t num_cases>
constexpr std::array<size_t, num_cases> get_total_num_grid_points(
    const std::array<std::array<size_t, Dim>, num_cases>& extents) {
  std::array<size_t, num_cases> total_num_grid_points{};

  for (size_t i = 0; i < num_cases; i++) {
    gsl::at(total_num_grid_points, i) = gsl::at(extents, i)[0];
    for (size_t j = 1; j < Dim; j++) {
      gsl::at(total_num_grid_points, i) *= gsl::at(gsl::at(extents, i), j);
    }
  }

  return total_num_grid_points;
}

template <size_t Dim>
constexpr bool total_grid_points_is_product_of_extents(
    const std::array<size_t, Dim>& extents,
    const size_t total_num_grid_points) {
  size_t product = extents[0];
  for (size_t i = 1; i < Dim; i++) {
    product *= gsl::at(extents, i);
  }
  return total_num_grid_points == product;
}

template <size_t Dim, size_t NumCases>
constexpr bool total_grid_points_is_product_of_extents(
    const std::array<std::array<size_t, Dim>, NumCases>& extents,
    const std::array<size_t, NumCases>& total_num_grid_points) {
  for (size_t i = 0; i < NumCases; i++) {
    const bool total_grid_points_is_product =
        total_grid_points_is_product_of_extents(
            gsl::at(extents, i), gsl::at(total_num_grid_points, i));
    if (not total_grid_points_is_product) {
      return false;
    }
  }
  return true;
}

// Functions benchmarked
enum class Function {
  PartialDerivatives,
  LogicalPartialDerivatives,
  TimeDerivative,
  OgtimeDerivative
};

// Dim and p-refinement benchmarking cases
enum class GridPointsListType { Consecutive, PowersOfTwo };

template <size_t Dim, GridPointsListType ListType>
struct extents_and_total_num_grid_points;

template <size_t Dim>
struct extents_and_total_num_grid_points<Dim, GridPointsListType::Consecutive> {
  // to do grid points in range [2, 20]
  static constexpr size_t num_cases = 19;
  static constexpr std::array<std::array<size_t, Dim>, num_cases> extents =
      get_extents_consecutive<Dim, num_cases>();
  static constexpr std::array<size_t, num_cases> total_num_grid_points =
      get_total_num_grid_points(extents);

  static_assert(
      total_grid_points_is_product_of_extents(extents, total_num_grid_points),
      "Total number of grid points is not equal to the product of the extents");
};

template <>
struct extents_and_total_num_grid_points<1, GridPointsListType::PowersOfTwo> {
  static constexpr size_t num_cases = 12;
  static constexpr std::array<std::array<size_t, 1>, num_cases> extents{
      {{{2}},
       {{4}},
       {{8}},
       {{16}},
       {{32}},
       {{64}},
       {{128}},
       {{256}},
       {{512}},
       {{1024}},
       {{2048}},
       {{4096}}}};

  static constexpr std::array<size_t, num_cases> total_num_grid_points =
      get_total_num_grid_points(extents);

  static_assert(
      total_grid_points_is_product_of_extents(extents, total_num_grid_points),
      "Total number of grid points is not equal to the product of the extents");
};

template <>
struct extents_and_total_num_grid_points<2, GridPointsListType::PowersOfTwo> {
  static constexpr size_t num_cases = 11;
  static constexpr std::array<std::array<size_t, 2>, num_cases> extents{{
      {{2, 2}},    // 4
      {{4, 2}},    // 8
      {{4, 4}},    // 16
      {{8, 4}},    // 32
      {{8, 8}},    // 64
      {{16, 8}},   // 128
      {{16, 16}},  // 256
      {{32, 16}},  // 512
      {{32, 32}},  // 1024
      {{64, 32}},  // 2048
      {{64, 64}}   // 4096
  }};

  static constexpr std::array<size_t, num_cases> total_num_grid_points =
      get_total_num_grid_points(extents);

  static_assert(
      total_grid_points_is_product_of_extents(extents, total_num_grid_points),
      "Total number of grid points is not equal to the product of the extents");
};

template <>
struct extents_and_total_num_grid_points<3, GridPointsListType::PowersOfTwo> {
  static constexpr size_t num_cases = 10;
  static constexpr std::array<std::array<size_t, 3>, num_cases> extents{{
      {{2, 2, 2}},    // 8
      {{4, 2, 2}},    // 16
      {{4, 4, 2}},    // 32
      {{4, 4, 4}},    // 64
      {{8, 4, 4}},    // 128
      {{8, 8, 4}},    // 256
      {{8, 8, 8}},    // 512
      {{16, 8, 8}},   // 1024
      {{16, 16, 8}},  // 2048
      {{16, 16, 16}}  // 4096
  }};

  static constexpr std::array<size_t, num_cases> total_num_grid_points =
      get_total_num_grid_points(extents);

  static_assert(
      total_grid_points_is_product_of_extents(extents, total_num_grid_points),
      "Total number of grid points is not equal to the product of the extents");
};

template <size_t Dim, GridPointsListType ListType>
inline constexpr auto extents =
    extents_and_total_num_grid_points<Dim, ListType>::extents;
template <size_t Dim, GridPointsListType ListType>
inline constexpr auto total_num_grid_points =
    extents_and_total_num_grid_points<Dim, ListType>::total_num_grid_points;
template <size_t Dim, GridPointsListType ListType>
inline constexpr size_t num_cases =
    extents_and_total_num_grid_points<Dim, ListType>::num_cases;
}  // namespace BenchmarkHelpers
