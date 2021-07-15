// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \file
/// Defines functions and metafunctions used for helping evaluate
/// TensorExpression equations where generic spatial indices are used for
/// spacetime indices

namespace tt {
/*!
 * \ingroup TypeTraitsGroup TensorExpressionsGroup
 * \brief Check if a type `T` is a TensorIndex representing a concrete time
 * index
 */
template <typename T>
struct is_concrete_time_index : std::false_type {};
template <>
struct is_concrete_time_index<std::decay_t<decltype(ti_t)>> : std::true_type {};
template <>
struct is_concrete_time_index<std::decay_t<decltype(ti_T)>> : std::true_type {};
}  // namespace tt

// TODO: remove
template <typename... T>
struct td;

namespace TensorExpressions {
namespace detail {
template <typename State, typename Element>
struct remove_concrete_time_indices_impl {
  using type = typename std::conditional_t<
      not tt::is_concrete_time_index<Element>::value,
      tmpl::push_back<State, Element>, State>;
};

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
template <typename TensorIndexList>
struct remove_concrete_time_indices {
  using type = tmpl::fold<
      TensorIndexList, tmpl::list<>,
      remove_concrete_time_indices_impl<tmpl::_state, tmpl::_element>>;
};

template <typename State, typename Element, typename Iteration>
struct concrete_time_index_positions_impl {
  using type =
      typename std::conditional_t<tt::is_concrete_time_index<Element>::value,
                                  tmpl::push_back<State, Iteration>, State>;
};

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
template <typename TensorIndexList>
using concrete_time_index_positions = tmpl::enumerated_fold<
    TensorIndexList, tmpl::list<>,
    concrete_time_index_positions_impl<tmpl::_state, tmpl::_element, tmpl::_3>,
    tmpl::size_t<0>>;

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
/// \return the list of positions where the generic index is spatial and the
/// tensor index is spacetime
template <typename TensorIndexList>
constexpr auto get_concrete_time_index_positions() noexcept {
  using concrete_time_index_positions_ =
      concrete_time_index_positions<TensorIndexList>;
  // TODO: update ConstantEpressions to accept a type for if empty list?
  using make_list_type =
      std::conditional_t<tmpl::size<concrete_time_index_positions_>::value == 0,
                         size_t, concrete_time_index_positions_>;
  return make_array_from_list<make_list_type>();
}

constexpr bool is_concrete_time_index_value(const size_t& value) {
  return (value == ti_t.value or value == ti_T.value);
}
}  // namespace detail
}  // namespace TensorExpressions
