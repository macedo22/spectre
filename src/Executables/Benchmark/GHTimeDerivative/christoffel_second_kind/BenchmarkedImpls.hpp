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
  using christoffel_second_kind_type = tnsr::Abb<DataType, Dim>;
  using christoffel_first_kind_type = tnsr::abb<DataType, Dim>;
  using inverse_spacetime_metric_type = tnsr::AA<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    raise_or_lower_first_index(christoffel_second_kind, christoffel_first_kind,
                               inverse_spacetime_metric);
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static christoffel_second_kind_type
  manual_impl_lhs_return(
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    return raise_or_lower_first_index(christoffel_first_kind,
                                      inverse_spacetime_metric);
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_second_kind, christoffel_first_kind(ti_d, ti_b, ti_c) *
                                     inverse_spacetime_metric(ti_A, ti_D));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_second_kind, christoffel_first_kind(ti_d, ti_b, ti_c) *
                                     inverse_spacetime_metric(ti_D, ti_A));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<3>(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_second_kind, inverse_spacetime_metric(ti_A, ti_D) *
                                     christoffel_first_kind(ti_d, ti_b, ti_c));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<4>(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_second_kind, inverse_spacetime_metric(ti_D, ti_A) *
                                     christoffel_first_kind(ti_d, ti_b, ti_c));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<5>(
      gsl::not_null<christoffel_second_kind_type*> christoffel_second_kind,
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_second_kind, christoffel_first_kind(ti_d, ti_c, ti_b) *
                                     inverse_spacetime_metric(ti_A, ti_D));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static christoffel_second_kind_type
  tensorexpression_impl_lhs_return(
      const christoffel_first_kind_type& christoffel_first_kind,
      const inverse_spacetime_metric_type& inverse_spacetime_metric) noexcept {
    return TensorExpressions::evaluate<ti_A, ti_b, ti_c>(
        christoffel_first_kind(ti_d, ti_b, ti_c) *
        inverse_spacetime_metric(ti_A, ti_D));
  }
};
