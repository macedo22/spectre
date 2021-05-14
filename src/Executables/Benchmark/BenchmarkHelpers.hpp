// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace BenchmarkHelpers {
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
void zero_initialize_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  for (auto tensor_it = tensor->begin(); tensor_it != tensor->end();
       tensor_it++) {
    *tensor_it = 0.0;
  }
}

template <typename... Ts>
void zero_initialize_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = 0.0;
    }
  }
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
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
                 gsl::not_null<Tensor<Ts...>*> tensor_destination) noexcept {
  auto tensor_source_it = tensor_source.begin();
  auto tensor_destination_it = tensor_destination->begin();
  for (; tensor_source_it != tensor_source.end();
       tensor_source_it++, tensor_destination_it++) {
    *tensor_destination_it = *tensor_source_it;
  }
}
}  // namespace BenchmarkHelpers
