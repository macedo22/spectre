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
  using pi_one_normal_type = tnsr::a<DataType, Dim>;
  using normal_spacetime_vector_type = tnsr::A<DataType, Dim>;
  using pi_type = tnsr::aa<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    for (size_t a = 0; a < Dim + 1; ++a) {
      pi_one_normal->get(a) = get<0>(normal_spacetime_vector) * pi.get(0, a);
      for (size_t b = 1; b < Dim + 1; ++b) {
        pi_one_normal->get(a) += normal_spacetime_vector.get(b) * pi.get(b, a);
      }
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static pi_one_normal_type manual_impl_lhs_return(
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    pi_one_normal_type pi_one_normal =
        make_with_value<pi_one_normal_type>(normal_spacetime_vector, 0.);
    manual_impl_lhs_arg(make_not_null(&pi_one_normal), normal_spacetime_vector,
                        pi);
    return pi_one_normal;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    tenex::evaluate<ti::a>(
        pi_one_normal, normal_spacetime_vector(ti::B) * pi(ti::b, ti::a));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    tenex::evaluate<ti::a>(
        pi_one_normal, normal_spacetime_vector(ti::B) * pi(ti::a, ti::b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<3>(
      gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    tenex::evaluate<ti::a>(
        pi_one_normal, pi(ti::b, ti::a) * normal_spacetime_vector(ti::B));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<4>(
      gsl::not_null<pi_one_normal_type*> pi_one_normal,
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    tenex::evaluate<ti::a>(
        pi_one_normal, pi(ti::a, ti::b) * normal_spacetime_vector(ti::B));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static pi_one_normal_type
  tensorexpression_impl_lhs_return(
      const normal_spacetime_vector_type& normal_spacetime_vector,
      const pi_type& pi) noexcept {
    return tenex::evaluate<ti::a>(normal_spacetime_vector(ti::B) *
                                             pi(ti::b, ti::a));
  }
};
