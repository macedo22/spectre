// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Expressions/TimeIndex.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a `TensorExpressions::SquareRoot`
///
/// \details
/// The empty base class provides a simple means for checking if a type is a
/// `TensorExpressions::SquareRoot`.
struct MarkAsSquareRoot {};

/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details The expression can have a non-zero number of indices as long as
/// all indices are concrete time indices, as this represents a rank 0 tensor.
///
/// \tparam T the type of the tensor expression of which to take the square
/// root
/// \tparam Args the TensorIndexs of the expression
template <typename T, typename... Args>
struct SquareRoot
    : public TensorExpression<SquareRoot<T, Args...>, typename T::type,
                              typename T::symmetry, typename T::index_list,
                              tmpl::list<Args...>>,
      MarkAsSquareRoot {
  static_assert(
      (... and tt::is_time_index<Args>::value),
      "Can only take the square root of a tensor expression that evaluates to "
      "a rank 0 tensor.");

  using type = typename T::type;
  using symmetry = typename T::symmetry;
  using index_list = typename T::index_list;
  using args_list = tmpl::list<Args...>;
  static constexpr bool is_binary_op = false;
  static constexpr auto num_tensor_indices = sizeof...(Args);
  static constexpr size_t num_ops_left = T::num_ops_subtree;
  static constexpr size_t num_ops_right = 0;
  static constexpr size_t num_ops_subtree = T::num_ops_subtree + 1;
  static constexpr size_t num_addsub_ops_subtree = T::num_addsub_ops_subtree;
  static constexpr bool is_main_end = T::is_main_beg;
  static constexpr size_t num_ops_to_evaluate_main_left =
      is_main_end ? 0 : T::num_ops_to_evaluate_main_subtree;
  static constexpr size_t num_ops_to_evaluate_main_subtree =
      num_ops_to_evaluate_main_left + 1;
  static constexpr bool is_main_beg =
      num_ops_to_evaluate_main_subtree >= detail::max_num_ops_in_sub_expression;

  SquareRoot(T t) : t_(std::move(t)) {}
  ~SquareRoot() override = default;

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_main(
      const ResultType& result_component) const {
    if constexpr (is_main_end) {
      return sqrt(result_component);
    } else {
      return sqrt(t_.get_main(result_component));
    }
  }

  /// \brief Returns the square root of the component of the tensor evaluated
  /// from the contained tensor expression
  ///
  /// \details
  /// SquareRoot only supports tensor expressions that evaluate to a rank 0
  /// Tensor. This is why `multi_index` is always an array of size 0.
  ///
  /// \param multi_index the multi-index of the component of which to take the
  /// square root
  /// \return the square root of the component of the tensor evaluated from the
  /// contained tensor expression
  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_main(
      const ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (is_main_end) {
      // TODO : better error message
      // static_assert(not is_main_beg, "Shouldn't happen.");
      (void)multi_index;
      return sqrt(result_component);
    } else {
      return sqrt(t_.get_main(result_component, multi_index));
    }
  }

  SPECTRE_ALWAYS_INLINE decltype(auto) get_main(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    // TODO : implement to do branch things
    return sqrt(t_.get_main(multi_index));
  }

  SPECTRE_ALWAYS_INLINE decltype(auto) get_branch(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    // TODO : implement to do branch things
    return sqrt(t_.get_branch(multi_index));
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit_main(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) {
    current_multi_index = multi_index;
    t_.visit_main(result_component, multi_index);
    // TODO : better error message; move up with member variables instead
    // instead function?
    // static_assert(not(is_main_beg and is_main_end), "Shouldn't happen.");
    if constexpr (is_main_beg) {
      result_component = get_main(result_component);
    }
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit_branch(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    // TODO : implement to do branch things
    t_.visit_branch(result_component, multi_index);
  }

  type get_used_for_size() const { return t_.get_used_for_size(); }

 private:
  T t_;
  std::array<size_t, num_tensor_indices> current_multi_index{};
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the square root of a
/// tensor expression that evaluates to a rank 0 tensor
///
/// \details
/// `t` must be an expression that, when evaluated, would be a rank 0 tensor.
/// For example, if `R` and `S` are Tensors, here is a non-exhaustive list of
/// some of the acceptable forms that `t` could take:
/// - `R()`
/// - `R(ti_A, ti_a)`
/// - `(R(ti_A, ti_B) * S(ti_a, ti_b))`
/// - `R(ti_t, ti_t) + 1.0`
///
/// \param t the tensor expression of which to take the square root
template <typename T, typename X, typename Symm, typename IndexList,
          typename... Args>
SPECTRE_ALWAYS_INLINE auto sqrt(
    const TensorExpression<T, X, Symm, IndexList, tmpl::list<Args...>>& t) {
  return TensorExpressions::SquareRoot<T, Args...>(~t);
}
