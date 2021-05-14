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
  using three_index_constraint_type = tnsr::iaa<DataType, Dim>;
  using d_spacetime_metric_type = tnsr::iaa<DataType, Dim>;
  using phi_type = tnsr::iaa<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t a = 0; a < Dim + 1; ++a) {
        for (size_t b = a; b < Dim + 1; ++b) {
          three_index_constraint->get(i, a, b) =
              d_spacetime_metric.get(i, a, b) - phi.get(i, a, b);
        }
      }
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static three_index_constraint_type
  manual_impl_lhs_return(const d_spacetime_metric_type& d_spacetime_metric,
                         const phi_type& phi) noexcept {
    three_index_constraint_type three_index_constraint =
        make_with_value<three_index_constraint_type>(d_spacetime_metric, 0.);
    manual_impl_lhs_arg(make_not_null(&three_index_constraint),
                        d_spacetime_metric, phi);
    return three_index_constraint;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
        three_index_constraint,
        d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
        three_index_constraint,
        d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_b, ti_a));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<3>(
      gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
        three_index_constraint,
        d_spacetime_metric(ti_i, ti_b, ti_a) - phi(ti_i, ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<4>(
      gsl::not_null<three_index_constraint_type*> three_index_constraint,
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
        three_index_constraint,
        d_spacetime_metric(ti_i, ti_b, ti_a) - phi(ti_i, ti_b, ti_a));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static three_index_constraint_type
  tensorexpression_impl_lhs_return(
      const d_spacetime_metric_type& d_spacetime_metric,
      const phi_type& phi) noexcept {
    return TensorExpressions::evaluate<ti_i, ti_a, ti_b>(
        d_spacetime_metric(ti_i, ti_a, ti_b) - phi(ti_i, ti_a, ti_b));
  }
};
