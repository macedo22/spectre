// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the tensor expression representing negation

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
/// \ingroup TensorExpressionsGroup
/// \brief Marks a class as being a `TensorExpressions::Negate`
///
/// \details
/// The empty base class provides a simple means for checking if a type is a
/// `TensorExpressions::Negate`.
struct MarkAsNegate {};

/// \ingroup TensorExpressionsGroup
/// \brief Defines the tensor expression representing the negation of a tensor
/// expression
///
/// \tparam T the type of tensor expression being negated
template <typename T>
struct Negate
    : public TensorExpression<Negate<T>, typename T::type, typename T::symmetry,
                              typename T::index_list, typename T::args_list>,
      MarkAsNegate {
  using type = typename T::type;
  using symmetry = typename T::symmetry;
  using index_list = typename T::index_list;
  using args_list = typename T::args_list;
  static constexpr bool is_binary_op = false;
  static constexpr auto num_tensor_indices = tmpl::size<index_list>::value;
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

  static constexpr size_t consecutive_branch_ops_left = num_ops_left;
  static constexpr size_t consecutive_branch_ops_right = 0;
  static constexpr size_t consecutive_branch_ops =
      consecutive_branch_ops_left + 1;

  Negate(T t) : t_(std::move(t)) {}
  ~Negate() override = default;

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE decltype(auto) get_main(
      const ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    if constexpr (is_main_end) {
      (void)multi_index;
      return -result_component;
    } else {
      return -t_.get_main(result_component, multi_index);
    }
  }

  /// \brief Return the value of the component of the negated tensor expression
  /// at a given multi-index
  ///
  /// \param multi_index the multi-index of the component to retrieve from the
  /// negated tensor expression
  /// \return the value of the component at `multi_index` in the negated tensor
  /// expression
  SPECTRE_ALWAYS_INLINE decltype(auto) get_branch(
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    return -t_.get_branch(multi_index);
  }

  template <typename ResultType>
  SPECTRE_ALWAYS_INLINE void visit_main(
      ResultType& result_component,
      const std::array<size_t, num_tensor_indices>& multi_index) const {
    t_.visit_main(result_component, multi_index);
    if constexpr (is_main_beg) {
      result_component = get_main(result_component, multi_index);
    }
  }

  type get_used_for_size() const { return t_.get_used_for_size(); }

 private:
  T t_;
};
}  // namespace TensorExpressions

/// \ingroup TensorExpressionsGroup
/// \brief Returns the tensor expression representing the negation of a tensor
/// expression
///
/// \param t the tensor expression
/// \return the tensor expression representing the negation of `t`
template <typename T>
SPECTRE_ALWAYS_INLINE auto operator-(
    const TensorExpression<T, typename T::type, typename T::symmetry,
                           typename T::index_list, typename T::args_list>& t) {
  return TensorExpressions::Negate<T>(~t);
}
