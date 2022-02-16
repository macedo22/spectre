// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ET for tensor division by scalars

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/NumberAsExpression.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the quotient of one tensor
/// expression divided by another tensor expression that evaluates to a rank 0
/// tensor
///
/// \tparam T1 the numerator operand expression of the division expression
/// \tparam T2 the denominator operand expression of the division expression
/// \tparam Args2 the generic indices of the denominator expression
template <typename T1, typename T2, typename... Args2>
struct Divide : public TensorExpression<
                    Divide<T1, T2, Args2...>,
                    typename std::conditional_t<
                        std::is_same<typename T1::type, DataVector>::value or
                            std::is_same<typename T2::type, DataVector>::value,
                        DataVector, double>,
                    typename T1::symmetry, typename T1::index_list,
                    typename T1::args_list> {
  static_assert(std::is_same<typename T1::type, typename T2::type>::value or
                    std::is_same<T1, NumberAsExpression>::value,
                "Cannot divide TensorExpressions holding different data types");
  static_assert((... and tt::is_time_index<Args2>::value),
                "Can only divide a tensor expression by a double or a tensor "
                "expression that evaluates to "
                "a rank 0 tensor.");

  using type =
      std::conditional_t<std::is_same<typename T1::type, DataVector>::value or
                             std::is_same<typename T2::type, DataVector>::value,
                         DataVector, double>;
  using symmetry = typename T1::symmetry;
  using index_list = typename T1::index_list;
  using args_list = typename T1::args_list;
  static constexpr auto num_tensor_indices =
      tmpl::size<typename T1::index_list>::value;
  static constexpr auto op2_num_tensor_indices =
      tmpl::size<typename T2::index_list>::value;
  // the denominator has no indices or all time indices
  static constexpr auto op2_multi_index =
      make_array<op2_num_tensor_indices, size_t>(0);

  static constexpr size_t num_ops_left_child = T1::num_ops_subtree;
  static constexpr size_t num_ops_right_child = T2::num_ops_subtree;
  static constexpr size_t num_ops_subtree =
      num_ops_left_child + num_ops_right_child + 1;

  static constexpr bool is_primary_end = T1::is_primary_start;
  static constexpr size_t num_ops_to_evaluate_primary_left_child =
      is_primary_end ? 0 : T1::num_ops_to_evaluate_primary_subtree;
  static constexpr size_t num_ops_to_evaluate_primary_right_child =
      num_ops_right_child;
  static constexpr size_t num_ops_to_evaluate_primary_subtree =
      num_ops_to_evaluate_primary_left_child +
      num_ops_to_evaluate_primary_right_child + 1;
  static constexpr bool is_primary_start =
      num_ops_to_evaluate_primary_subtree >=
      detail::max_num_ops_in_sub_expression<type>;
  static constexpr bool evaluate_children_separately =
      is_primary_start and (num_ops_to_evaluate_primary_left_child >=
                                detail::max_num_ops_in_sub_expression<type> or
                            num_ops_to_evaluate_primary_right_child >=
                                detail::max_num_ops_in_sub_expression<type>);

  static constexpr bool primary_child_subtree_contains_primary_start =
      T1::primary_subtree_contains_primary_start;
  static constexpr bool primary_subtree_contains_primary_start =
      is_primary_start or primary_child_subtree_contains_primary_start;

  Divide(T1 t1, T2 t2) : t1_(std::move(t1)), t2_(std::move(t2)) {}
  ~Divide() override = default;

  /// \brief Assert that the LHS tensor of the equation does not also appear in
  /// this expression's subtree
  template <typename LhsTensor>
  SPECTRE_ALWAYS_INLINE void assert_lhs_tensor_not_in_rhs_expression(
      const gsl::not_null<LhsTensor*> lhs_tensor) const {
    if constexpr (not std::is_base_of_v<NumberAsExpression, T1>) {
      t1_.assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
    }
    if constexpr (not std::is_base_of_v<NumberAsExpression, T2>) {
      t2_.assert_lhs_tensor_not_in_rhs_expression(lhs_tensor);
    }
  }

  SPECTRE_ALWAYS_INLINE auto get_used_for_size() const {
    if constexpr (not std::is_base_of_v<NumberAsExpression, T2>) {
      return t2_.get_used_for_size();
    } else {
      return t1_.get_used_for_size();
    }
  }

  /// \brief Return the value of the component of the quotient tensor at a given
  /// multi-index
  ///
  /// \param result_multi_index the multi-index of the component of the quotient
  //// tensor to retrieve
  /// \return the value of the component in the quotient tensor at
  /// `result_multi_index`
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    return t1_.get(result_multi_index) / t2_.get(op2_multi_index);
  }

  SPECTRE_ALWAYS_INLINE void evaluate_primary_children(
      type& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    // don't send result_component down right branch because we are at a * and
    // shouldn't edit result_component in right child
    if constexpr (is_primary_end) {
      (void)result_multi_index;
      result_component /= t2_.get(op2_multi_index);
    } else {
      if constexpr (primary_child_subtree_contains_primary_start) {
        result_component =
            t1_.get_primary(result_component, result_multi_index);
      } else {
        result_component = t1_.get(result_multi_index);
      }
      result_component /= t2_.get(op2_multi_index);
    }
  }

  SPECTRE_ALWAYS_INLINE decltype(auto) get_primary(
      const type& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    if constexpr (is_primary_end) {
      (void)result_multi_index;
      return result_component / t2_.get(op2_multi_index);
    } else {
      return t1_.get_primary(result_component, result_multi_index) /
             t2_.get(op2_multi_index);
    }
  }

  SPECTRE_ALWAYS_INLINE void evaluate_primary_subtree(
      type& result_component,
      const std::array<size_t, num_tensor_indices>& result_multi_index) const {
    if constexpr (primary_child_subtree_contains_primary_start) {
      t1_.evaluate_primary_subtree(result_component, result_multi_index);
    }
    if constexpr (is_primary_start) {
      if constexpr (evaluate_children_separately) {
        evaluate_primary_children(result_component, result_multi_index);
      } else {
        result_component = get_primary(result_component, result_multi_index);
      }
    }
  }

 private:
  T1 t1_;
  T2 t2_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of one tensor
/// expression over another tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t2` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t2` could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
/// - `R(ti_t, ti_t) + 1.0`
///
/// \param t1 the tensor expression numerator
/// \param t2 the rank 0 tensor expression denominator
template <typename T1, typename T2, typename... Args2>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T1, typename T1::type, typename T1::symmetry,
                           typename T1::index_list, typename T1::args_list>& t1,
    const TensorExpression<T2, typename T2::type, typename T2::symmetry,
                           typename T2::index_list, tmpl::list<Args2...>>& t2) {
  return TensorExpressions::Divide<T1, T2, Args2...>(~t1, ~t2);
}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a tensor
/// expression over a `double`
///
/// \note The implementation instead uses the operation, `t * (1.0 / number)`
///
/// \param t the tensor expression operand of the quotient
/// \param number the `double` operand of the quotient
/// \return the tensor expression representing the quotient of a tensor
/// expression and a `double`
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator/(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t,
    const double number) {
  return t * TensorExpressions::NumberAsExpression(1.0 / number);
}

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the quotient of a `double`
/// over a tensor expression that evaluates to a rank 0 tensor
///
/// \param number the `double` numerator of the quotient
/// \param t the tensor expression denominator of the quotient
/// \return the tensor expression representing the quotient of a `double` over a
/// tensor expression that evaluates to a rank 0 tensor
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator/(
    const double number,
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return TensorExpressions::NumberAsExpression(number) / t;
}
