// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace BenchmarkHelpers {
template <typename DataType>
DataType get_used_for_size(const size_t num_grid_points);

template <>
double get_used_for_size<double>(const size_t /*num_grid_points*/) {
  return std::numeric_limits<double>::signaling_NaN();
}

template <>
DataVector get_used_for_size<DataVector>(const size_t num_grid_points) {
  return DataVector(num_grid_points,
                    std::numeric_limits<double>::signaling_NaN());
}

template <typename DataType>
std::string get_benchmark_name_suffix(const size_t dim) {
  const std::string datatype =
      std::is_same_v<DataType, DataVector> ? "DataVector" : "double";
  return datatype + "/" + std::to_string(dim) + "D/num_grid_points:";
}

template <typename DataType>
std::string get_benchmark_name(const std::string prefix, const size_t dim) {
  return prefix + get_benchmark_name_suffix<DataType>(dim);
}

template <typename... Ts>
void zero_initialize_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
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

template <typename... Ts>
void copy_tensor(const Tensor<Ts...>& tensor_source,
                 gsl::not_null<Tensor<Ts...>*> tensor_destination) noexcept {
  auto tensor_source_it = tensor_source.begin();
  auto tensor_destination_it = tensor_destination->begin();
  for (; tensor_source_it != tensor_source.end();
       tensor_source_it++, tensor_destination_it++) {
    *tensor_destination_it = *tensor_source_it;
  }
}
}  // namespace BenchmarkHelpers

// Implementations benchmarked
template <typename DataType, size_t Dim>
struct BenchmarkImpl {
  // tensor types in tensor equation being benchmarked
  using dt_spacetime_metric_type = tnsr::aa<DataType, Dim>;
  using lapse_type = Scalar<DataType>;
  using pi_type = tnsr::aa<DataType, Dim>;
  using gamma1_plus_1_type = Scalar<DataType>;
  using shift_dot_three_index_constraint_type = tnsr::aa<DataType, Dim>;
  using shift_type = tnsr::I<DataType, Dim>;
  using phi_type = tnsr::iaa<DataType, Dim>;

  // manual implementation benchmarked that takes LHS tensor as arg
  SPECTRE_ALWAYS_INLINE static void manual_impl_lhs_arg(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    for (size_t a = 0; a < Dim + 1; ++a) {
      for (size_t b = a; b < Dim + 1; ++b) {
        dt_spacetime_metric->get(a, b) = -get(lapse) * pi.get(a, b);
        dt_spacetime_metric->get(a, b) +=
            get(gamma1_plus_1) * shift_dot_three_index_constraint.get(a, b);
        for (size_t i = 0; i < Dim; ++i) {
          dt_spacetime_metric->get(a, b) += shift.get(i) * phi.get(i, a, b);
        }
      }
    }
  }

  // manual implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static dt_spacetime_metric_type manual_impl_lhs_return(
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    dt_spacetime_metric_type dt_spacetime_metric =
        make_with_value<dt_spacetime_metric_type>(lapse, 0.);
    manual_impl_lhs_arg(make_not_null(&dt_spacetime_metric), lapse, pi,
                        gamma1_plus_1, shift_dot_three_index_constraint, shift,
                        phi);
    return dt_spacetime_metric;
  }

  // TensorExpression implementation benchmarked that takes LHS tensor as arg
  template <size_t CaseNumber>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept;

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<1>(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_a, ti_b>(
        dt_spacetime_metric,
        -1.0 * lapse() * pi(ti_a, ti_b) +
            gamma1_plus_1() * shift_dot_three_index_constraint(ti_a, ti_b) +
            shift(ti_I) * phi(ti_i, ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<2>(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_a, ti_b>(
        dt_spacetime_metric,
        shift(ti_I) * phi(ti_i, ti_a, ti_b) + -1.0 * lapse() * pi(ti_a, ti_b) +
            gamma1_plus_1() * shift_dot_three_index_constraint(ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<3>(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_a, ti_b>(
        dt_spacetime_metric,
        -1.0 * lapse() * pi(ti_b, ti_a) +
            gamma1_plus_1() * shift_dot_three_index_constraint(ti_b, ti_a) +
            shift(ti_I) * phi(ti_i, ti_b, ti_a));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<4>(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_a, ti_b>(
        dt_spacetime_metric,
        -1.0 * lapse() * pi(ti_a, ti_b) +
            gamma1_plus_1() * shift_dot_three_index_constraint(ti_b, ti_a) +
            shift(ti_I) * phi(ti_i, ti_a, ti_b));
  }

  template <>
  SPECTRE_ALWAYS_INLINE static void tensorexpression_impl_lhs_arg<5>(
      gsl::not_null<dt_spacetime_metric_type*> dt_spacetime_metric,
      const lapse_type& lapse, const pi_type& pi,
      const gamma1_plus_1_type& gamma1_plus_1,
      const shift_dot_three_index_constraint_type&
          shift_dot_three_index_constraint,
      const shift_type& shift, const phi_type& phi) noexcept {
    TensorExpressions::evaluate<ti_a, ti_b>(
        dt_spacetime_metric,
        -1.0 * lapse() * pi(ti_a, ti_b) +
            gamma1_plus_1() * shift_dot_three_index_constraint(ti_a, ti_b) +
            phi(ti_i, ti_a, ti_b) * shift(ti_I));
  }

  // TensorExpression implementation benchmarked that returns LHS tensor
  SPECTRE_ALWAYS_INLINE static dt_spacetime_metric_type
  tensorexpression_impl_lhs_return(const lapse_type& lapse, const pi_type& pi,
                                   const gamma1_plus_1_type& gamma1_plus_1,
                                   const shift_dot_three_index_constraint_type&
                                       shift_dot_three_index_constraint,
                                   const shift_type& shift,
                                   const phi_type& phi) noexcept {
    return TensorExpressions::evaluate<ti_a, ti_b>(
        -1.0 * lapse() * pi(ti_a, ti_b) +
        gamma1_plus_1() * shift_dot_three_index_constraint(ti_a, ti_b) +
        shift(ti_I) * phi(ti_i, ti_a, ti_b));
  }
};
