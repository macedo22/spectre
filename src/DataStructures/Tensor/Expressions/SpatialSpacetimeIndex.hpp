// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
// TODO: try to make this an alias instead of a struct
template <typename State, typename Element, typename Iteration,
          typename TensorIndexList>
struct spatial_spacetime_index_positions_impl {
  using type = typename std::conditional<
      Element::index_type == IndexType::Spacetime and
          not tmpl::at<TensorIndexList, Iteration>::is_spacetime,
      tmpl::push_back<State, Iteration>, State>::type;
};

template <typename TensorIndexTypeList, typename TensorIndexList>
using spatial_spacetime_index_positions = tmpl::enumerated_fold<
    TensorIndexTypeList, tmpl::list<>,
    spatial_spacetime_index_positions_impl<
        tmpl::_state, tmpl::_element, tmpl::_3, tmpl::pin<TensorIndexList>>,
    tmpl::size_t<0>>;

template <typename TensorIndexTypeList, typename TensorIndexList>
constexpr auto get_spatial_spacetime_index_positions() noexcept {
  using spatial_spacetime_index_positions_ =
      spatial_spacetime_index_positions<TensorIndexTypeList, TensorIndexList>;
  using make_list_type = std::conditional_t<
      tmpl::size<spatial_spacetime_index_positions_>::value == 0, size_t,
      spatial_spacetime_index_positions_>;
  return make_array_from_list<make_list_type>();
}

// TODO: this needs to also handle antisymmetry...
template <size_t NumIndices, size_t NumSpatialSpacetimeIndices>
constexpr std::array<std::int32_t, NumIndices>
get_spatial_spacetime_index_symmetry(
    const std::array<std::int32_t, NumIndices>& symmetry,
    const std::array<size_t, NumSpatialSpacetimeIndices>&
        spatial_spacetime_index_positions) noexcept {
  std::array<std::int32_t, NumIndices> spatial_spacetime_index_symmetry{};
  const std::int32_t max_symm_value =
      static_cast<std::int32_t>(*alg::max_element(symmetry));
  for (size_t i = 0; i < NumIndices; i++) {
    spatial_spacetime_index_symmetry[i] = symmetry[i];
  }
  for (size_t i = 0; i < NumSpatialSpacetimeIndices; i++) {
    spatial_spacetime_index_symmetry[spatial_spacetime_index_positions[i]] +=
        max_symm_value;
  }

  return spatial_spacetime_index_symmetry;
}

template <typename S, typename E>
struct replace_spatial_spacetime_indices_helper {
  using type = tmpl::replace_at<S, E, change_indextype<tmpl::at<S, E>>>;
};

template <typename TensorIndexTypeList, typename SpatialSpacetimeIndexPositions>
using replace_spatial_spacetime_indices = tmpl::fold<
    SpatialSpacetimeIndexPositions, TensorIndexTypeList,
    replace_spatial_spacetime_indices_helper<tmpl::_state, tmpl::_element>>;
}  // namespace TensorExpressions
