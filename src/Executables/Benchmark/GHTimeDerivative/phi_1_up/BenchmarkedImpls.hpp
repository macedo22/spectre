// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using phi_1_up_type = tnsr::Iaa<DataType, Dim>;
  using inverse_spatial_metric_type = tnsr::II<DataType, Dim>;
  using phi_type = tnsr::iaa<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = a; b < Dim + 1; b++) {
          phi_1_up->get(i, a, b) =
              inverse_spatial_metric.get(i, 0) * phi.get(0, a, b);
          for (size_t j = 1; j < Dim; ++j) {
            phi_1_up->get(i, a, b) +=
                inverse_spatial_metric.get(i, j) * phi.get(j, a, b);
          }
        }
      }
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static phi_1_up_type manual_impl_lhs_return(
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    phi_1_up_type phi_1_up =
        make_with_value<phi_1_up_type>(inverse_spatial_metric, 0.);
    manual_impl_lhs_arg(make_not_null(&phi_1_up), inverse_spatial_metric, phi);
    return phi_1_up;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
        phi_1_up, inverse_spatial_metric(ti_I, ti_J) * phi(ti_j, ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
        phi_1_up, inverse_spatial_metric(ti_J, ti_I) * phi(ti_j, ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<3>(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
        phi_1_up, phi(ti_j, ti_a, ti_b) * inverse_spatial_metric(ti_I, ti_J));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<4>(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
        phi_1_up, phi(ti_j, ti_a, ti_b) * inverse_spatial_metric(ti_J, ti_I));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<5>(
      gsl::not_null<phi_1_up_type*> phi_1_up,
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
        phi_1_up, phi(ti_j, ti_b, ti_a) * inverse_spatial_metric(ti_I, ti_J));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static phi_1_up_type tensorexpression_impl_lhs_return(
      const inverse_spatial_metric_type& inverse_spatial_metric,
      const phi_type& phi) noexcept {
    return TensorExpressions::evaluate<ti_I, ti_a, ti_b>(
        inverse_spatial_metric(ti_I, ti_J) * phi(ti_j, ti_a, ti_b));
  }
};
