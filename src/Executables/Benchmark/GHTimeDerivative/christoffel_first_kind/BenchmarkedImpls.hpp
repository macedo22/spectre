// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

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
  //     const da_spacetime_metric_type& da_spacetime_metric) noexcept {
  //   return gr::christoffel_first_kind(da_spacetime_metric);
  // }

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const da_spacetime_metric_type& da_spacetime_metric) noexcept {
    gr::christoffel_first_kind(christoffel_first_kind, da_spacetime_metric);
  }

  // Note: TE can't determine correct LHS symmetry from adding first two terms
  // TensorExpression implementation benchmarked that returns LHS tensor
  // SPECTRE_ALWAYS_INLINE static christoffel_first_kind_type
  // tensorexpression_impl_lhs_return(
  //     const da_spacetime_metric_type& da_spacetime_metric) noexcept {
  //   return tenex::evaluate<ti::c, ti::a, ti::b>(
  //       0.5 * (da_spacetime_metric(ti::a, ti::b, ti::c) +
  //              da_spacetime_metric(ti::b, ti::a, ti::c) -
  //              da_spacetime_metric(ti::c, ti::a, ti::b)));
  // }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const da_spacetime_metric_type& da_spacetime_metric) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const da_spacetime_metric_type& da_spacetime_metric) noexcept {
    tenex::evaluate<ti::c, ti::a, ti::b>(
        christoffel_first_kind, 0.5 * (da_spacetime_metric(ti::a, ti::b, ti::c) +
                                       da_spacetime_metric(ti::b, ti::a, ti::c) -
                                       da_spacetime_metric(ti::c, ti::a, ti::b)));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<christoffel_first_kind_type*> christoffel_first_kind,
      const da_spacetime_metric_type& da_spacetime_metric) noexcept {
    tenex::evaluate<ti::c, ti::a, ti::b>(
        christoffel_first_kind, (da_spacetime_metric(ti::a, ti::b, ti::c) +
                                 da_spacetime_metric(ti::b, ti::a, ti::c) -
                                 da_spacetime_metric(ti::c, ti::a, ti::b)) *
                                    0.5);
  }
};
