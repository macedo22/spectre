// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
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
  using gauge_constraint_type = tnsr::a<DataType, Dim>;
  using gauge_function_type = tnsr::a<DataType, Dim>;
  using trace_christoffel_type = tnsr::a<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<gauge_constraint_type*> gauge_constraint,
      const gauge_function_type& gauge_function,
      const trace_christoffel_type& trace_christoffel) noexcept {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      gauge_constraint->get(nu) =
          gauge_function.get(nu) + trace_christoffel.get(nu);
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static gauge_constraint_type manual_impl_lhs_return(
      const gauge_function_type& gauge_function,
      const trace_christoffel_type& trace_christoffel) noexcept {
    gauge_constraint_type gauge_constraint =
        make_with_value<gauge_constraint_type>(gauge_function, 0.);
    manual_impl_lhs_arg(make_not_null(&gauge_constraint), gauge_function,
                        trace_christoffel);
    return gauge_constraint;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<gauge_constraint_type*> gauge_constraint,
      const gauge_function_type& gauge_function,
      const trace_christoffel_type& trace_christoffel) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<gauge_constraint_type*> gauge_constraint,
      const gauge_function_type& gauge_function,
      const trace_christoffel_type& trace_christoffel) noexcept {
    TensorExpressions::evaluate<ti_a>(
        gauge_constraint, gauge_function(ti_a) + trace_christoffel(ti_a));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static gauge_constraint_type
  tensorexpression_impl_lhs_return(
      const gauge_function_type& gauge_function,
      const trace_christoffel_type& trace_christoffel) noexcept {
    return TensorExpressions::evaluate<ti_a>(gauge_function(ti_a) +
                                             trace_christoffel(ti_a));
  }
};
