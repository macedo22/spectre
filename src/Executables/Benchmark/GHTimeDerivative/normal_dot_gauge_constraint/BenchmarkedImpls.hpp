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
  using normal_dot_gauge_constraint_type = Scalar<DataType>;
  using normal_spacetime_vector_type = tnsr::A<DataType, Dim>;
  using gauge_constraint_type = tnsr::a<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<normal_dot_gauge_constraint_type*>
          normal_dot_gauge_constraint,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const gauge_constraint_type& gauge_constraint) noexcept {
    get(*normal_dot_gauge_constraint) =
        get<0>(normal_spacetime_vector) * get<0>(gauge_constraint);
    for (size_t a = 1; a < Dim + 1; ++a) {
      get(*normal_dot_gauge_constraint) +=
          normal_spacetime_vector.get(a) * gauge_constraint.get(a);
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static normal_dot_gauge_constraint_type
  manual_impl_lhs_return(
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const gauge_constraint_type& gauge_constraint) noexcept {
    normal_dot_gauge_constraint_type normal_dot_gauge_constraint =
        make_with_value<normal_dot_gauge_constraint_type>(
            normal_spacetime_vector, 0.);
    manual_impl_lhs_arg(make_not_null(&normal_dot_gauge_constraint),
                        normal_spacetime_vector, gauge_constraint);
    return normal_dot_gauge_constraint;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<normal_dot_gauge_constraint_type*>
          normal_dot_gauge_constraint,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const gauge_constraint_type& gauge_constraint) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<normal_dot_gauge_constraint_type*>
          normal_dot_gauge_constraint,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const gauge_constraint_type& gauge_constraint) noexcept {
    TensorExpressions::evaluate(
        normal_dot_gauge_constraint,
        normal_spacetime_vector(ti_A) * gauge_constraint(ti_a));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static normal_dot_gauge_constraint_type
  tensorexpression_impl_lhs_return(
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const gauge_constraint_type& gauge_constraint) noexcept {
    return TensorExpressions::evaluate(normal_spacetime_vector(ti_A) *
                                       gauge_constraint(ti_a));
  }
};
