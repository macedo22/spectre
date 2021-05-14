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
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using trace_christoffel_type = tnsr::a<DataType, Dim>;
  using christoffel_first_kind_type = tnsr::abb<DataType, Dim>;
  using inverse_spacetime_metric_type = tnsr::AA<DataType, Dim>;

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static trace_christoffel_type manual_impl_lhs_return(
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    return trace_last_indices(christoffel_first_kind, inverse_spacetime_metric);
  }

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    trace_last_indices(trace_christoffel, christoffel_first_kind,
                       inverse_spacetime_metric);
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static trace_christoffel_type
  tensorexpression_impl_lhs_return(
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    return TensorExpressions::evaluate<ti_a>(
        christoffel_first_kind(ti_a, ti_b, ti_c) *
        inverse_spacetime_metric(ti_B, ti_C));
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_a>(trace_christoffel,
                                      christoffel_first_kind(ti_a, ti_b, ti_c) *
                                          inverse_spacetime_metric(ti_B, ti_C));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<trace_christoffel_type*> trace_christoffel,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_a>(trace_christoffel,
                                      christoffel_first_kind(ti_a, ti_b, ti_c) *
                                          inverse_spacetime_metric(ti_C, ti_B));
  }
};
