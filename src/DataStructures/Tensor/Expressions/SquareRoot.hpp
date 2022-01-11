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
  static constexpr auto num_tensor_indices = sizeof...(Args);
  static constexpr size_t num_ops_left = T::num_ops_subtree;
  static constexpr size_t num_ops_right = 0;
  static constexpr size_t num_ops_subtree = T::num_ops_subtree + 1;
  static constexpr size_t num_addsub_ops_subtree = T::num_addsub_ops_subtree;

  SquareRoot(T t) : t_(std::move(t)) {}
  ~SquareRoot() override = default;

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
  SPECTRE_ALWAYS_INLINE decltype(auto) get(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return sqrt(t_.get(multi_index));
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit_main(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    t_.visit_main(result_component, multi_index);
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit_branch(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    t_.visit_branch(result_component, multi_index);
  }

 private:
  T t_;
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
