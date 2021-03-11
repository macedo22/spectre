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

namespace BenchmarkImpl {
constexpr size_t Dim = 3;
constexpr size_t num_grid_points = 5;

using phi_1_up_type = tnsr::Iaa<DataVector, Dim>;
using inverse_spatial_metric_type = tnsr::II<DataVector, Dim>;
using phi_type = tnsr::iaa<DataVector, Dim>;

SPECTRE_ALWAYS_INLINE void manual_impl(
    const gsl::not_null<phi_1_up_type*> phi_1_up,
    const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
    const phi_type& phi) noexcept {
  for (size_t m = 0; m < Dim; ++m) {  // *
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        phi_1_up->get(m, mu, nu) =
            inverse_spatial_metric->get(m, 0) * phi.get(0, mu, nu);
        for (size_t n = 1; n < Dim; ++n) {
          phi_1_up->get(m, mu, nu) +=
              inverse_spatial_metric->get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }
}

SPECTRE_ALWAYS_INLINE tnsr::Iaa<DataVector, Dim> tensorexpression_impl_return(
    const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
    const tnsr::iaa<DataVector, Dim>& phi) noexcept {
  return TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
      (*inverse_spatial_metric)(ti_I, ti_J) * phi(ti_j, ti_a, ti_b));
}

SPECTRE_ALWAYS_INLINE void tensorexpression_impl_lhs_as_arg(
    const gsl::not_null<phi_1_up_type*> phi_1_up,
    const gsl::not_null<inverse_spatial_metric_type*> inverse_spatial_metric,
    const phi_type& phi) noexcept {
  TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
      phi_1_up, (*inverse_spatial_metric)(ti_I, ti_J) * phi(ti_j, ti_a, ti_b));
}
}  // namespace BenchmarkImpl
