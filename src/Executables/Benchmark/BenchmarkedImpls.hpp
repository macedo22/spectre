// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace BenchmarkHelpers {
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
  using christoffel_first_kind_type = tnsr::abb<DataType, Dim>;
  using da_spacetime_metric_type = tnsr::abb<DataType, Dim>;

  // Note: TE can't determine correct LHS symmetry
  // manual implementation benchmarked that returns LHS tensor
  // SPECTRE_ALWAYS_INLINE static christoffel_first_kind_type
  // manual_impl_lhs_return(
  //     const da_spacetime_metric_type da_spacetime_metric) noexcept {
  //   return gr::christoffel_first_kind(da_spacetime_metric);
  // }

  // manual implementation benchmarked that takes LHS tensor as argument
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const da_spacetime_metric_type da_spacetime_metric) noexcept {
    gr::christoffel_first_kind(christoffel_first_kind, da_spacetime_metric);
  }

  // Note: TE can't determine correct LHS symmetry from adding first two terms
  // TensorExpression implementation benchmarked returns LHS tensor
  // SPECTRE_ALWAYS_INLINE static christoffel_first_kind_type
  // tensorexpression_impl_lhs_return(
  //     const da_spacetime_metric_type da_spacetime_metric) noexcept {
  //   return TensorExpressions::evaluate<ti_c, ti_a, ti_b>(
  //       0.5 * (da_spacetime_metric(ti_a, ti_b, ti_c) +
  //              da_spacetime_metric(ti_b, ti_a, ti_c) -
  //              da_spacetime_metric(ti_c, ti_a, ti_b)));
  // }

  // TensorExpression implementation benchmarked that takes LHS tensor as
  // argument
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const da_spacetime_metric_type da_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_c, ti_a, ti_b>(
        christoffel_first_kind, 0.5 * (da_spacetime_metric(ti_a, ti_b, ti_c) +
                                       da_spacetime_metric(ti_b, ti_a, ti_c) -
                                       da_spacetime_metric(ti_c, ti_a, ti_b)));
  }
};
