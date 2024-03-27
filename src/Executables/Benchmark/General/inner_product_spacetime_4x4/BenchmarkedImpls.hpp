// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Executables/Benchmark/BenchmarkHelpers.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using result_type = Scalar<DataType>;
  using R_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Up, Frame::Inertial>>>;
  using S_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<Dim, UpLo::Lo, Frame::Inertial>>>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    destructive_resize_components(result, get_size(get<0, 0, 0, 0>(R)));
    get(*result) = 0.0;

    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        for (size_t c = 0; c < Dim + 1; c++) {
          for (size_t d = 0; d < Dim + 1; d++) {
            get(*result) += R.get(a, b, c, d) * S.get(a, b, c, d);
          }
        }
      }
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static result_type manual_impl_lhs_return(
      const R_type& R, const S_type& S) {
    result_type result{};
    manual_impl_lhs_arg(make_not_null(&result), R, S);
    return result;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S);

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<result_type*> result, const R_type& R, const S_type& S) {
    tenex::evaluate(
        result, R(ti::A, ti::B, ti::C, ti::D) * S(ti::a, ti::b, ti::c, ti::d));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static result_type tensorexpression_impl_lhs_return(
      const R_type& R, const S_type& S) {
    return tenex::evaluate(R(ti::A, ti::B, ti::C, ti::D) *
                                       S(ti::a, ti::b, ti::c, ti::d));
  }
};
