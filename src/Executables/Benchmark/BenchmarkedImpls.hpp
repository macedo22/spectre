// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace BenchmarkHelpers {
template <typename DataType>
DataType get_used_for_size(const size_t num_grid_points);

template <>
double get_used_for_size<double>(const size_t /*num_grid_points*/) {
  return std::numeric_limits<double>::signaling_NaN();
}

template <>
DataVector get_used_for_size<DataVector>(const size_t num_grid_points) {
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

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using trace_christoffel_type = tnsr::a<DataType, Dim>;
  using christoffel_first_kind_type = tnsr::abb<DataType, Dim>;
  using inverse_spacetime_metric_type = tnsr::AA<DataType, Dim>;

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static trace_christoffel_type manual_impl_lhs_return(
      const christoffel_first_kind_type christoffel_first_kind,
      const inverse_spacetime_metric_type inverse_spacetime_metric) noexcept {
    return trace_last_indices(christoffel_first_kind, inverse_spacetime_metric);
  }

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type christoffel_first_kind,
      const inverse_spacetime_metric_type inverse_spacetime_metric) noexcept {
    trace_last_indices(trace_christoffel, christoffel_first_kind,
                       inverse_spacetime_metric);
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static trace_christoffel_type
  tensorexpression_impl_lhs_return(
      const christoffel_first_kind_type christoffel_first_kind,
      const inverse_spacetime_metric_type inverse_spacetime_metric) noexcept {
    return TensorExpressions::evaluate<ti_a>(
        christoffel_first_kind(ti_a, ti_b, ti_c) *
        inverse_spacetime_metric(ti_B, ti_C));
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type christoffel_first_kind,
      const inverse_spacetime_metric_type inverse_spacetime_metric) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type christoffel_first_kind,
      const inverse_spacetime_metric_type inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_a>(trace_christoffel,
                                      christoffel_first_kind(ti_a, ti_b, ti_c) *
                                          inverse_spacetime_metric(ti_B, ti_C));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type christoffel_first_kind,
      const inverse_spacetime_metric_type inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_a>(trace_christoffel,
                                      christoffel_first_kind(ti_a, ti_b, ti_c) *
                                          inverse_spacetime_metric(ti_C, ti_B));
  }
};
