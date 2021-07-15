// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

#include "DataStructures/Tensor/Expressions/ConcreteTimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/SpatialSpacetimeIndex.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorExpressions {
// TODO: review below documentation
/*!
 * \ingroup TensorExpressionsGroup
 * \brief Determines and stores a LHS tensor's symmetry and index list from a
 * RHS tensor expression and desired LHS index order
 *
 * \details Given the generic index order of a RHS TensorExpression and the
 * generic index order of the desired LHS Tensor, this creates a mapping between
 * the two that is used to determine the (potentially reordered) ordering of the
 * elements of the desired LHS Tensor`s ::Symmetry and typelist of
 * \ref SpacetimeIndex "TensorIndexType"s.
 *
 * Note: If a generic spatial index is used for a spacetime index in the RHS
 * tensor, its corresponding index in the LHS tensor type will be a spatial
 * index with the same valence, frame, and number of spatial dimensions.
 *
 * @tparam RhsTensorIndexList the typelist of TensorIndex of the RHS
 * TensorExpression
 * @tparam LhsTensorIndexList the typelist of TensorIndexs of the desired LHS
 * tensor
 * @tparam RhsSymmetry the ::Symmetry of the RHS indices
 * @tparam RhsTensorIndexTypeList the RHS TensorExpression's typelist of
 * \ref SpacetimeIndex "TensorIndexType"s
 */
template <typename RhsTensorIndexList, typename LhsTensorIndexList,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t NumLhsIndices = tmpl::size<LhsTensorIndexList>::value,
          size_t NumRhsIndices = tmpl::size<RhsTensorIndexList>::value,
          typename LhsIndexSequence = std::make_index_sequence<NumLhsIndices>>
struct LhsTensorSymmAndIndices;

template <typename... RhsTensorIndices, typename... LhsTensorIndices,
          typename RhsSymmetry, typename RhsTensorIndexTypeList,
          size_t NumLhsIndices, size_t NumRhsIndices, size_t... LhsInts>
struct LhsTensorSymmAndIndices<tmpl::list<RhsTensorIndices...>,
                               tmpl::list<LhsTensorIndices...>, RhsSymmetry,
                               RhsTensorIndexTypeList, NumLhsIndices,
                               NumRhsIndices, std::index_sequence<LhsInts...>> {
  static_assert((... and
                 (not tt::is_concrete_time_index<LhsTensorIndices>::value)),
                "LHS generic indices cannot contain the concrete time index.");
  // LHS generic indices, RHS generic indices, and the mapping between them
  static constexpr std::array<size_t, NumLhsIndices> lhs_tensorindex_values = {
      {LhsTensorIndices::value...}};
  static constexpr std::array<size_t, NumRhsIndices> rhs_tensorindex_values = {
      {RhsTensorIndices::value...}};
  static constexpr std::array<size_t, NumLhsIndices> rhs_to_lhs_map =
      compute_tensorindex_transformation(rhs_tensorindex_values,
                                         lhs_tensorindex_values);

  // Compute symmetry of RHS after spacetime indices using generic spatial
  // indices are swapped for spatial indices
  static constexpr std::array<std::int32_t, NumRhsIndices> rhs_symmetry = {
      {tmpl::at_c<RhsSymmetry, LhsInts>::value...}};
  using rhs_spatial_spacetime_index_positions_ =
      detail::spatial_spacetime_index_positions<
          RhsTensorIndexTypeList, tmpl::list<RhsTensorIndices...>>;
  using make_list_type = std::conditional_t<
      tmpl::size<rhs_spatial_spacetime_index_positions_>::value == 0, size_t,
      rhs_spatial_spacetime_index_positions_>;
  static constexpr auto rhs_spatial_spacetime_index_positions =
      make_array_from_list<make_list_type>();
  static constexpr std::array<std::int32_t, NumRhsIndices>
      rhs_spatial_spacetime_index_symmetry =
          detail::get_spatial_spacetime_index_symmetry(
              rhs_symmetry, rhs_spatial_spacetime_index_positions);

  // Compute index list of RHS after spacetime indices using generic spatial
  // indices are made nonsymmetric to other indices
  using rhs_spatial_spacetime_tensorindextype_list =
      detail::replace_spatial_spacetime_indices<
          RhsTensorIndexTypeList, rhs_spatial_spacetime_index_positions_>;

  // Desired LHS Tensor's Symmetry, typelist of TensorIndexTypes, and Structure
  using symmetry = Symmetry<
      rhs_spatial_spacetime_index_symmetry[rhs_to_lhs_map[LhsInts]]...>;
  using tensorindextype_list =
      tmpl::list<tmpl::at_c<rhs_spatial_spacetime_tensorindextype_list,
                            rhs_to_lhs_map[LhsInts]>...>;
  using structure =
      Tensor_detail::Structure<symmetry,
                               tmpl::at_c<tensorindextype_list, LhsInts>...>;
};
}  // namespace TensorExpressions
