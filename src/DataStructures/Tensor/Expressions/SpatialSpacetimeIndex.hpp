// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"

/// \file
/// Defines functions and metafunctions used for helping evaluate tensor
/// expression equations where generic spatial indices are used for spacetime
/// indices

namespace TensorExpressions {
namespace detail {
template <typename State, typename Element, typename Iteration,
          typename TensorIndexList>
struct spatial_spacetime_index_positions_impl {
  using type = typename std::conditional_t<
      Element::index_type == IndexType::Spacetime and
          not tmpl::at<TensorIndexList, Iteration>::is_spacetime,
      tmpl::push_back<State, Iteration>, State>;
};

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
/// \tparam TensorIndexTypeList the list of
/// \ref SpacetimeIndex "TensorIndexType"s
template <typename TensorIndexTypeList, typename TensorIndexList>
using spatial_spacetime_index_positions = tmpl::enumerated_fold<
    TensorIndexTypeList, tmpl::list<>,
    spatial_spacetime_index_positions_impl<
        tmpl::_state, tmpl::_element, tmpl::_3, tmpl::pin<TensorIndexList>>,
    tmpl::size_t<0>>;

/// \brief Given a generic index list and tensor index list, returns the list of
/// positions where the generic index is spatial and the tensor index is
/// spacetime
///
/// \tparam TensorIndexList the generic index list
/// \tparam TensorIndexTypeList the list of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \return the list of positions where the generic index is spatial and the
/// tensor index is spacetime
template <typename TensorIndexTypeList, typename TensorIndexList>
constexpr auto get_spatial_spacetime_index_positions() noexcept {
  using spatial_spacetime_index_positions_ =
      spatial_spacetime_index_positions<TensorIndexTypeList, TensorIndexList>;
  using make_list_type = std::conditional_t<
      tmpl::size<spatial_spacetime_index_positions_>::value == 0, size_t,
      spatial_spacetime_index_positions_>;
  return make_array_from_list<make_list_type>();
}

/// \brief Given a tensor symmetry and the positions of indices where a generic
/// spatial index is used for a spacetime index, this returns the symmetry
/// after making those indices nonsymmetric with others
///
/// \details
/// Example: If `symmetry` is `[2, 1, 1, 1]` and
/// `spatial_spacetime_index_positions` is `[1]`, then position 1 is the only
/// position where a generic spatial index is used for a spacetime index. The
/// resulting symmetry will make the index at position 1 no longer be symmetric
/// with the indices at positions 2 and 3. Therefore, the resulting symmetry
/// will be equivalent to the form of `[3, 2, 1, 1]`.
///
/// Note: the symmetry returned by this function is not in the canonical form
/// specified by ::Symmetry. In reality, for the example above, this function
/// would return `[2, 3, 1, 1]`.
///
/// \param symmetry the input tensor symmetry to transform
/// \param spatial_spacetime_index_positions the positions of the indices of the
/// tensor where a generic spatial index is used for a spacetime index
/// \return the symmetry after making the `spatial_spacetime_index_positions` of
/// `symmetry` nonsymmetric with other indices
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

// The list of indices resulting from taking `TensorIndexTypeList` and
// replacing the spacetime indices at positions `SpatialSpacetimeIndexPositions`
// with spatial indices
template <typename TensorIndexTypeList, typename SpatialSpacetimeIndexPositions>
using replace_spatial_spacetime_indices = tmpl::fold<
    SpatialSpacetimeIndexPositions, TensorIndexTypeList,
    replace_spatial_spacetime_indices_helper<tmpl::_state, tmpl::_element>>;
}  // namespace detail
}  // namespace TensorExpressions
