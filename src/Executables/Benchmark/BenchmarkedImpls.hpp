// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace BenchmarkHelpers {
template <typename... Ts>
void zero_initialize_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
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
}  // namespace BenchmarkHelpers

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being profiles
  using christoffel_second_kind_type = tnsr::Abb<DataType, Dim>;
  using christoffel_first_kind_type = tnsr::abb<DataType, Dim>;
  using inverse_spacetime_metric_type = tnsr::AA<DataType, Dim>;

  // manual implementation profiled
  SPECTRE_ALWAYS_INLINE static void manual_impl(
      const gsl::not_null<christoffel_second_kind_type*>
          christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    raise_or_lower_first_index(christoffel_second_kind, christoffel_first_kind,
                               inverse_spacetime_metric);
  }

  // TensorExpression implementation profiled that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static christoffel_second_kind_type
  tensorexpression_impl_return(
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    return TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_first_kind(ti_d, ti_b, ti_c) *
        inverse_spacetime_metric(ti_A, ti_D));
  }

  // TensorExpression implementation profiled that takes LHS tensor as argument
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_as_arg(
      const gsl::not_null<christoffel_second_kind_type*>
          christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_second_kind, christoffel_first_kind(ti_d, ti_b, ti_c) *
                                     inverse_spacetime_metric(ti_A, ti_D));
  }
};
