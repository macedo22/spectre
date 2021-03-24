// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
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

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}
}  // namespace BenchmarkHelpers

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  using frame = Frame::Inertial;

  // index types in tensor equation being benchmarked
  using A_index = SpacetimeIndex<Dim, UpLo::Up, frame>;
  using a_index = SpacetimeIndex<Dim, UpLo::Lo, frame>;
  using B_index = SpacetimeIndex<Dim, UpLo::Up, frame>;
  using b_index = SpacetimeIndex<Dim, UpLo::Lo, frame>;
  using C_index = B_index;
  using d_index = SpacetimeIndex<Dim, UpLo::Lo, frame>;
  using I_index = SpatialIndex<Dim, UpLo::Up, frame>;
  using i_index = SpatialIndex<Dim, UpLo::Lo, frame>;
  using J_index = SpatialIndex<Dim, UpLo::Up, frame>;
  using j_index = i_index;
  using k_index = SpatialIndex<Dim, UpLo::Lo, frame>;
  using l_index = SpatialIndex<Dim, UpLo::Lo, frame>;

  // tensor types in tensor equation being benchmarked
  using L_type = Tensor<DataType, Symmetry<4, 3, 2, 1>,
                        index_list<C_index, d_index, k_index, l_index>>;
  using R_type = Tensor<DataType, Symmetry<3, 3, 2, 1>,
                        index_list<i_index, j_index, b_index, A_index>>;
  using S_type = Tensor<DataType, Symmetry<3, 2, 1, 1>,
                        index_list<d_index, a_index, B_index, C_index>>;
  using T_type = Tensor<DataType, Symmetry<4, 3, 2, 1>,
                        index_list<J_index, k_index, l_index, I_index>>;

  // manual implementation benchmarked that takes LHS tensor as argument
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_as_arg(
      const gsl::not_null<L_type*> L, const R_type& R, const S_type& S,
      const T_type& T) noexcept {
    BenchmarkHelpers::zero_initialize_tensor(L);

    for (size_t c = 0; c < C_index::dim; c++) {
      for (size_t d = 0; d < d_index::dim; d++) {
        for (size_t k = 0; k < k_index::dim; k++) {
          for (size_t l = 0; l < l_index::dim; l++) {
            for (size_t i = 0; i < i_index::dim; i++) {
              for (size_t j = 0; j < j_index::dim; j++) {
                for (size_t b = 0; b < b_index::dim; b++) {
                  for (size_t a = 0; a < a_index::dim; a++) {
                    L->get(c, d, k, l) += R.get(i, j, b, a) *
                                          S.get(d, a, b, c) * T.get(j, k, l, i);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static L_type manual_impl_return(
      const R_type& R, const S_type& S, const T_type& T,
      const DataType& used_for_size) noexcept {
    L_type L(used_for_size);
    manual_impl_lhs_as_arg(make_not_null(&L), R, S, T);
    return L;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as
  // argument
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_as_arg(
      const gsl::not_null<L_type*> L, const R_type& R, const S_type& S,
      const T_type& T) noexcept {
    TensorExpressions::evaluate<ti_C, ti_d, ti_k, ti_l>(
        L, R(ti_i, ti_j, ti_b, ti_A) *
               (S(ti_d, ti_a, ti_B, ti_C) * T(ti_J, ti_k, ti_l, ti_I)));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static L_type tensorexpression_impl_return(
      const R_type& R, const S_type& S, const T_type& T) noexcept {
    return TensorExpressions::evaluate<ti_C, ti_d, ti_k, ti_l>(
        R(ti_i, ti_j, ti_b, ti_A) *
        (S(ti_d, ti_a, ti_B, ti_C) * T(ti_J, ti_k, ti_l, ti_I)));
  }
};
